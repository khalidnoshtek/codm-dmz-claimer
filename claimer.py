#!/usr/bin/env python3
"""One-shot claim attempt.

Loads config, attaches to the AVD via ADB, foregrounds CODM, walks the
DMZ LST Hunt flow, taps every available CLAIM, and exits. Run from cron
or driven by daemon.py for the continuous loop.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from lib.adb import AdbDevice, ensure_avd_running, LOCKED_AVD
from lib.flow import DEFAULT_STEPS, run_flow
from lib.ocr import read_cooldowns_with_retry

ROOT = Path(__file__).resolve().parent
TEMPLATES = ROOT / "templates"
LOGS = ROOT / "logs"


def setup_logging(level: int = logging.INFO) -> None:
    LOGS.mkdir(exist_ok=True)
    handler_console = logging.StreamHandler(sys.stdout)
    handler_console.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    handler_file = logging.FileHandler(LOGS / "claimer.log")
    handler_file.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers[:] = [handler_console, handler_file]


# Dedicated user-friendly log: one line per cycle, plain English, no debug
# spam. Separate file so the verbose log stays available for debugging but
# isn't what the user has to read day-to-day.
def log_status(message: str) -> None:
    LOGS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp}  {message}\n"
    with (LOGS / "status.log").open("a") as f:
        f.write(line)


def load_config() -> dict:
    cfg_path = ROOT / "config.yaml"
    with cfg_path.open() as f:
        return yaml.safe_load(f) or {}


def _ocr_region(img, y1: float, y2: float, x1: float, x2: float, upscale: int = 2) -> str:
    """OCR one fraction-addressed region of the screen, upscaled.

    Whole-frame OCR is unreliable here: the AVD renders at 3120x1440, so
    dialog and button text is tiny relative to the frame and tesseract drops
    it. The logged-out screen, for instance, OCR'd to nothing but the version
    string -- which is why a logged-out cycle was reported as "CODM crashed"
    for four runs straight. Cropping to the region of interest and upscaling
    makes the same text read cleanly. Fractions (not pixels) so this survives
    a resolution change."""
    import cv2
    import pytesseract
    h, w = img.shape[:2]
    crop = img[int(h * y1):int(h * y2), int(w * x1):int(w * x2)]
    crop = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    return pytesseract.image_to_string(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)).upper()


def _signed_out_reason(img) -> str | None:
    """Why CODM is showing its sign-in screen, or None if it isn't.

    Three distinct states land here and they need different responses:
      - the account-picker screen (GUEST / CALL OF DUTY / Facebook / Google),
        identified by the TERMS OF USE & PRIVACY POLICY footer;
      - "Authorization error. (2B11)", CODM invalidating the session -- the
        footer is partly hidden behind the dialog's OK button, so the footer
        check alone misses it;
      - the Activision password form, which was the only case handled before.
    """
    footer = _ocr_region(img, 0.68, 0.95, 0.10, 0.90)
    # Two body bands on purpose. A wide crop reads a long wrapped message but
    # loses a single short line -- tesseract's page segmentation gets lost in
    # the surrounding key art -- while a tight centre crop reads the short line
    # and clips the wrapped one. "Authorization error. (2B11)" needs the tight
    # one; "Download configuration failed..." needs the wide one.
    body = _ocr_region(img, 0.28, 0.62, 0.10, 0.90) + " " + _ocr_region(img, 0.40, 0.60, 0.10, 0.90)
    if "AUTHORIZATION ERROR" in body:
        return "CODM signed you out (authorization error) — sign in again on the AVD"
    if "PRIVACY POLICY" in footer or "TERMS OF USE" in footer:
        return "signed out — CODM is on the sign-in screen (pick your account on the AVD)"
    full = _ocr_region(img, 0.0, 1.0, 0.0, 1.0, upscale=1)
    keys = ("LOGIN NOW", "ACTIVISION ACCOUNT", "FORGOT YOUR PASSWORD",
            "I'M NOT A ROBOT", "RECAPTCHA", "DON'T HAVE AN ACTIVISION")
    if any(k in full for k in keys):
        return "needs login — sign in to CODM (Activision account)"
    return None


def _on_login_screen(device, confirm_seconds: float = 120.0) -> bool:
    """True if CODM is genuinely stuck on a sign-in state.

    One look is not enough. CODM renders the sign-in layout -- the account
    buttons and the TERMS OF USE & PRIVACY POLICY footer -- while it is
    AUTO-logging in, so a single glance cannot tell "signed out" from
    "signing in", and calling it early aborts a perfectly good cycle with a
    bogus needs_login. So re-look until the screen moves on or the window
    expires; only a sign-in screen that is still there at the end counts.

    The authorization-error dialog is the exception: it is a definitive
    rejection, not a transient state, so it returns immediately.
    """
    try:
        deadline = time.time() + confirm_seconds
        while True:
            reason = _signed_out_reason(device.screencap())
            if reason is None:
                return False               # moved on -> it was signing itself in
            if "authorization error" in reason:
                return True                # definitive, no point waiting
            if time.time() >= deadline:
                return True                # still sitting there -> genuinely out
            time.sleep(5.0)
    except Exception:
        return False


SESSION_CONFLICT_DETAIL = ("you were playing CODM elsewhere — the AVD session was "
                           "kicked (account accessed from another location)")


def _session_conflict(img) -> bool:
    """True for CODM's "Account accessed from another location. (0E100004)"
    dialog.

    CODM allows one live session per account, so this appears whenever the
    claimer connects while you are playing on your phone. The dialog's only
    button is QUIT GAME -- there is nothing to dismiss, which is why the
    popup handler sat tapping its close-X six times in a row and the cycle
    hung until the AVD stopped responding.

    It matters that this is never retried: each retry signs in again and
    kicks YOU off your phone mid-match.
    """
    body = _ocr_region(img, 0.28, 0.62, 0.10, 0.90) + " " + _ocr_region(img, 0.40, 0.60, 0.10, 0.90)
    return "ACCESSED FROM ANOTHER" in body or "0E100004" in body or "OE100004" in body


def _classify_failure(device, pkg: str) -> str:
    """Human-readable reason the flow couldn't reach the claim screen, by
    inspecting whatever CODM is actually showing when it gave up. Turns the
    useless generic 'dmz_lobby_check not found' into something actionable."""
    try:
        import cv2
        import pytesseract
        foreground = device.is_app_foreground(pkg)
        img = device.screencap()
        txt = pytesseract.image_to_string(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).upper()
    except Exception:
        return "couldn't reach the DMZ lobby (screen unreadable)"
    # Identify the screen BEFORE falling back to "not in foreground". A
    # logged-out CODM is still CODM, and "signed you out" is far more
    # actionable than "crashed" -- which is what four runs reported while the
    # AVD sat on the sign-in screen.
    if _session_conflict(img):
        return SESSION_CONFLICT_DETAIL
    signed_out = _signed_out_reason(img)
    if signed_out:
        return signed_out
    if "DOWNLOAD CONFIGURATION" in (_ocr_region(img, 0.28, 0.62, 0.10, 0.90)
                                    + " " + _ocr_region(img, 0.40, 0.60, 0.10, 0.90)):
        return "CODM couldn't download its config — the AVD had no working internet"
    if not foreground:
        return "CODM crashed / closed (not in foreground)"
    if "UNSTABLE" in txt or "CHECK YOUR CONNECTION" in txt or "NETWORK" in txt:
        return "CODM network error during load (connection unstable)"
    if "EXPIRED" in txt:
        return "blocked by a 'time-limited item expired' popup"
    if "STORAGE" in txt:
        return "blocked by a 'device storage' popup"
    if "UPDATE" in txt or "DOWNLOAD" in txt:
        return "CODM wants to update/download resources"
    # Mostly-empty splash = the logo with little else -> stuck connecting.
    words = txt.split()
    if "CALL" in txt and "DUTY" in txt and len(words) < 16:
        return "stuck on CODM loading screen — likely needs an app update (open the AVD via scrcpy, update in Play Store)"
    return "reached CODM but couldn't find the DMZ lobby (unexpected screen or slow load)"


def claim_once(cfg: dict, dry_run_override: bool | None = None) -> dict:
    """Run the claim flow once. Returns a summary dict that's also written
    to logs/<timestamp>_summary.json."""
    dry_run = dry_run_override if dry_run_override is not None else bool(cfg.get("dry_run", False))
    log = logging.getLogger("claim_once")

    # Auto-start the locked AVD if it's down. Lets the daemon survive AVD
    # restarts / Mac reboots without manual intervention — only the Mac itself
    # has to be on. The AVD name is hard-locked in lib/adb.py (LOCKED_AVD);
    # config.yaml is ignored to prevent accidental redirection.
    try:
        ensure_avd_running(
            LOCKED_AVD,
            boot_timeout=float(cfg.get("avd_boot_timeout_seconds", 240)),
            headless=bool(cfg.get("emulator_headless", True)),
            gpu_mode=str(cfg.get("emulator_gpu", "host")),
            virtio_wifi=bool(cfg.get("emulator_virtio_wifi", False)),
        )
    except Exception as e:
        log.error("Could not start/find locked AVD %s: %s", LOCKED_AVD, e)
        return {"ok": False, "reason": "avd_not_running",
                "fail_detail": "the AVD (emulator) wouldn't start", "target_avd": LOCKED_AVD, "error": str(e)}

    device = AdbDevice.auto()
    log.info("ADB device: %s (dry_run=%s)", device.serial, dry_run)

    # Keep the display awake so headless rendering stays fresh and screencaps
    # don't come back frozen (idempotent — persists in the AVD's userdata).
    device.keep_awake()

    # Lock screen: the AVD's pattern lock keeps reappearing across cold boots,
    # and while the keyguard is up CODM's launcher activity won't resolve
    # (monkey/am start fail with 'No activities found' / exit 252). So clear
    # the lock every cycle via `locksettings` — reliable and idempotent, unlike
    # the old swipe-the-pattern gesture. The credential is the pattern dots
    # joined into digits (e.g. [5,3,6,9] -> "5369").
    lock_cfg = cfg.get("lockscreen") or {}
    pattern = lock_cfg.get("pattern") or []
    credential = "".join(str(d) for d in pattern) or None
    device.disable_lock(old_credential=credential)

    # Legacy pattern-drawing unlock, kept only if explicitly re-enabled.
    if lock_cfg.get("enabled"):
        if not device.ensure_unlocked(lock_cfg):
            log.error("Could not get past the lock screen — aborting cycle")
            return {"ok": False, "reason": "lockscreen",
                    "fail_detail": "couldn't get past the AVD lock screen", "target_avd": LOCKED_AVD}

    pkg = cfg["package"]
    activity = cfg.get("activity") or None

    # Launch with retries. CODM cold-launches are sometimes flaky on the AVD
    # (foregrounds briefly during splash, then crashes back to the launcher).
    # We poll for foreground throughout the cold-launch settle; if CODM dies,
    # we relaunch up to a few times before giving up.
    cold_launch = not device.is_app_foreground(pkg)
    if cold_launch:
        max_attempts = int(cfg.get("cold_launch_max_attempts", 3))
        # CODM is typically ready ~15s after foreground on this AVD; 25s gives
        # 10s of buffer for shader compilation variance without dragging the
        # cycle out by 45s of unnecessary polling.
        cold_settle = float(cfg.get("cold_launch_settle_seconds", 25))
        succeeded = False
        for attempt in range(1, max_attempts + 1):
            log.info("Launching %s (cold, attempt %d/%d) ...", pkg, attempt, max_attempts)
            device.launch_app(pkg, activity)
            if not device.wait_app_foreground(pkg, timeout=30.0):
                log.warning("App %s never reached foreground within 30s — retrying", pkg)
                continue
            # Poll throughout the cold settle. If CODM drops out of foreground,
            # something killed it (crash, OOM, splash crash) — break and retry.
            log.info("Cold-launch settle: polling for %.0fs that %s stays foregrounded", cold_settle, pkg)
            deadline = time.time() + cold_settle
            crashed = False
            while time.time() < deadline:
                time.sleep(3.0)
                if not device.is_app_foreground(pkg):
                    log.warning("App %s dropped out of foreground mid-settle — retrying", pkg)
                    crashed = True
                    break
            if not crashed:
                succeeded = True
                break
        if not succeeded:
            log.error("App %s failed to stay foregrounded after %d cold-launch attempts", pkg, max_attempts)
            return {"ok": False, "reason": "app_unstable_on_cold_launch",
                    "fail_detail": "CODM wouldn't stay open on launch (slow or crashing cold start)",
                    "package": pkg, "attempts": max_attempts}
    else:
        log.info("App %s already foregrounded (warm)", pkg)
        time.sleep(float(cfg.get("screen_settle_seconds", 2.5)) * 2)

    # We need OCR to run while we're still on the LST Hunt screen — that's
    # the only screen where the "Remaining HH:MM:SS" badges are visible.
    # Splitting the flow: run everything up to (but not including) the
    # back_to_lobby step, capture the screen + OCR, then run the back step.
    nav_steps = [s for s in DEFAULT_STEPS if s.name != "back_to_lobby"]
    back_step = next((s for s in DEFAULT_STEPS if s.name == "back_to_lobby"), None)

    # The login-screen popup is temporary — once it's gone, set
    # login_popup_enabled: false so we don't burn the step's 120s timeout
    # waiting for a popup that will never show.
    if not bool(cfg.get("login_popup_enabled", True)):
        nav_steps = [s for s in nav_steps if s.name != "login_popup_confirm"]
        # With the popup gone, its long wait no longer absorbs a slow launch /
        # small update. extra_launch_settle_seconds gives that time back if
        # updates start causing dmz_lobby_check aborts again.
        extra = float(cfg.get("extra_launch_settle_seconds", 0))
        if extra > 0 and cold_launch and not dry_run:
            log.info("Extra post-launch settle: %.0fs (login popup disabled)", extra)
            time.sleep(extra)

    # If CODM has dropped its session it lands on the Activision login form
    # (password + "I'm not a robot" captcha) — no automation can pass that.
    # Report it clearly and stop, WITHOUT running the flow or pressing BACK
    # (which would quit the game). Leaves CODM on the login screen so the user
    # can sign in.
    if not dry_run and _on_login_screen(device):
        log.error("CODM is on the Activision login screen — a manual sign-in is required.")
        log_status("NEEDS LOGIN — open the AVD and sign in to CODM (Activision account)")
        if bool(cfg.get("shutdown_between_cycles", True)):
            device.home()
            time.sleep(2.0)
            try:
                subprocess.run(["adb", "-s", device.serial, "emu", "kill"],
                               check=False, capture_output=True, timeout=10)
            except Exception:
                pass
        stamp = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        summary = {"ok": False, "reason": "needs_login", "stamp": stamp,
                   "claims_attempted": 0, "steps_run": [], "steps_skipped": [],
                   "aborted_at": "login", "abort_reason": "CODM logged out — manual sign-in required",
                   "final_screenshot": None, "cooldowns_seconds": [], "min_cooldown_seconds": None}
        (LOGS / f"{stamp}_summary.json").write_text(json.dumps(summary, indent=2))
        log.info("Summary: %s", summary)
        return summary

    # CODM throws promo popups (battle-pass, "FINAL WEEK FOR CP", event banners)
    # at unpredictable times — often AFTER the early dismiss_popups step, right
    # on the main lobby where they cover the DMZ:RECON tile and block navigation
    # (the classic dmz_lobby_check failure). Before each lobby->DMZ nav step,
    # close any popup whose X matches 10_popup_close_x.png. Scoped to the nav
    # steps only, so it never taps a close-X on the claim screens.
    _close_x = TEMPLATES / "10_popup_close_x.png"
    _dismiss_before = {"tap_home_icon", "enter_dmz_mode", "dmz_lobby_check", "tap_black_market"}

    # How long each step's hook will WAIT for its target to appear while
    # dismissing popups. enter_dmz_mode gets the big budget because the lobby
    # takes ~40s to render after login ("Getting Version Info"); rushing past
    # that was the real cause of the dmz_lobby_check failures.
    # enter_dmz_mode carries the cold-load wait now that login_popup_confirm no
    # longer does (see lib/flow.py). This budget is a ceiling, not a cost: the
    # hook returns as soon as the lobby renders, so raising it only helps a
    # slow load and never slows a fast one.
    _hook_budget = {"enter_dmz_mode": 150.0, "dmz_lobby_check": 25.0,
                    "tap_black_market": 15.0, "tap_home_icon": 6.0}

    def _dismiss_hook(step) -> None:
        if dry_run or step.name not in _dismiss_before:
            return
        try:
            from lib.vision import find_all, find_template, find_close_button
            import cv2 as _cv2
            import pytesseract as _pt
            thr = float(cfg.get("match_threshold", 0.82))
            tpl = TEMPLATES / step.template
            deadline = time.time() + _hook_budget.get(step.name, 8.0)
            backs = 0          # cap generic BACKs so we can't spiral

            def _quit_dialog_up(img) -> bool:
                """True if the 'Are you sure you want to quit the game?' dialog
                is on screen. Full-frame OCR misses this roughly half the time
                (the title is tiny against a 3120x1440 frame, and the lobby
                shows through behind it), which used to leave the dialog up and
                block every later step. Cropping to the dialog's title band and
                upscaling 2x makes detection reliable."""
                h, w = img.shape[:2]
                band = img[int(h * 0.16):int(h * 0.27), int(w * 0.15):int(w * 0.85)]
                band = _cv2.resize(band, None, fx=2, fy=2, interpolation=_cv2.INTER_CUBIC)
                t = _pt.image_to_string(_cv2.cvtColor(band, _cv2.COLOR_BGR2GRAY)).upper()
                return "QUIT THE GAME" in t or "WANT TO QUIT" in t
            while time.time() < deadline:
                screen = device.screencap()
                txt = _pt.image_to_string(_cv2.cvtColor(screen, _cv2.COLOR_BGR2GRAY)).upper()
                # 0) "Quit the game?" dialog — cancel it with BACK. This MUST be
                #    checked before the target-visible guard below: the lobby (and
                #    the DMZ tile) render straight through this dialog, so the
                #    guard would think we're clear and the flow would tap into the
                #    dialog. BACK cancels; we never tap OK.
                if _quit_dialog_up(screen):
                    log.info("popup dismiss: quit dialog — cancelling with BACK")
                    device.back()
                    time.sleep(1.1)
                    continue
                # Target visible -> lobby/screen is ready and clear, stop.
                if step.template and find_template(screen, tpl, threshold=thr):
                    return
                # 1) Any modal with a close X — promos, event banners, friend
                #    match invites. Masked matching finds the X whatever header
                #    it is drawn on; the old plain template matched none of the
                #    real popups, so these used to fall through to the BACK
                #    branch below. Closing by the X also avoids the buttons
                #    inside: a match invite's ACCEPT would drop the account into
                #    a live game.
                x_hit = find_close_button(screen, TEMPLATES)
                if x_hit:
                    log.info("popup dismiss: close-X at (%d,%d) score=%.3f",
                             x_hit.x, x_hit.y, x_hit.score)
                    device.tap(x_hit.x, x_hit.y)
                    time.sleep(1.0)
                    continue
                hits = find_all(screen, _close_x, threshold=thr)
                if hits:
                    log.info("popup dismiss: legacy close-X at (%d,%d)", hits[0].x, hits[0].y)
                    device.tap(hits[0].x, hits[0].y)
                    time.sleep(0.9)
                    continue
                # 2) Network-error dialog ("connection unstable" / "please check
                #    your connection") — buttons are QUIT GAME (left) + RETRY
                #    (right of the centered pair). Tap RETRY, NEVER quit. The
                #    hook loops, so repeated RETRYs give the connection time to
                #    settle during the cold load.
                if ("UNSTABLE" in txt or "CHECK YOUR CONNECTION" in txt
                        or ("NETWORK" in txt and "QUIT GAME" in txt)):
                    log.info("popup dismiss: network-error dialog — tapping RETRY (1770,953)")
                    device.tap(1770, 953)
                    time.sleep(2.5)
                    continue
                # 3) Any modal covering the lobby: if we can see lobby elements
                #    (RANKED/MULTIPLAYER/...) but not the DMZ tile, a popup is on
                #    top — WEB PURCHASE COMPLETE, EXPIRED item, DEVICE STORAGE,
                #    events, announcements, etc. CODM modals close on BACK.
                #    Capped at 2: if the target still isn't matching after that,
                #    it's more likely a borderline template match than a popup,
                #    and further BACKs would just toggle the Quit dialog.
                lobby_words = ("RANKED", "MULTIPLAYER", "BATTLE ROYALE", "LOADOUT",
                               "TOURNAMENT", "ZOMBIES", "DMZ")
                if sum(1 for w in lobby_words if w in txt) >= 2 and backs < 2:
                    backs += 1
                    log.info("popup dismiss: modal over lobby — BACK to clear (%d/2)", backs)
                    device.back()
                    time.sleep(1.3)
                    # A clean lobby looks identical to "modal over lobby" here:
                    # both show >=2 lobby words, and the DMZ tile can miss the
                    # template for unrelated reasons (e.g. the lobby is on BR
                    # RANKED). The difference only shows AFTER the BACK: on a
                    # clean lobby BACK has nothing to close, so CODM opens the
                    # quit dialog instead. Treat that as proof there was no
                    # modal — cancel it and stop pressing BACK, rather than
                    # ping-ponging the dialog open/closed for the rest of the
                    # budget (which is what stalled dmz_lobby_check).
                    if _quit_dialog_up(device.screencap()):
                        log.info("popup dismiss: BACK opened the quit dialog — "
                                 "lobby was already clear; cancelling and waiting")
                        device.back()
                        backs = 99          # disable further generic BACKs
                        time.sleep(1.1)
                    continue
                # Otherwise it's the login/loading splash still connecting — do
                # NOT press BACK (that would walk out of CODM); just wait.
                time.sleep(1.8)
        except Exception:
            pass

    result = run_flow(
        device,
        nav_steps,
        TEMPLATES,
        threshold=float(cfg.get("match_threshold", 0.82)),
        step_timeout=float(cfg.get("step_timeout_seconds", 20)),
        screen_settle=float(cfg.get("screen_settle_seconds", 2.5)),
        tap_settle=float(cfg.get("tap_settle_seconds", 1.2)),
        dry_run=dry_run,
        pre_step_hook=_dismiss_hook,
    )

    # Recovery: if the flow aborted, check whether we even have CODM in
    # foreground. The most common abort cause is CODM crashing to Android
    # home (network error dialog dismissed by user, ANR, asset-check error,
    # etc.). In that case relaunch CODM cold and retry the flow once.
    if not result.ok() and not dry_run:
        log.warning("Flow aborted at %s — attempting recovery", result.aborted_at)
        codm_alive = device.is_app_foreground(pkg)
        if not codm_alive:
            log.info("Recovery: CODM not foregrounded (probably crashed). Relaunching cold.")
            # Best-effort force_stop first to clean residual state, then relaunch
            try:
                device.force_stop(pkg)
            except Exception:
                pass
            time.sleep(2.0)
            device.launch_app(pkg, activity)
            if device.wait_app_foreground(pkg, timeout=60.0):
                time.sleep(float(cfg.get("cold_launch_settle_seconds", 25)))
            else:
                log.error("Recovery: CODM didn't come back after relaunch")
        else:
            # CODM is foregrounded but on an unknown screen — most often the
            # lobby still loading or a popup covering it. Do NOT blind-BACK
            # (that walks out of CODM). The retry's pre-step hook already waits
            # for the lobby and dismisses popups (close-X + recognized modals),
            # so just retry.
            log.info("Recovery: CODM foregrounded — retrying (hook handles load + popups)")
        log.info("Recovery: retrying navigation flow once")
        result = run_flow(
            device,
            nav_steps,
            TEMPLATES,
            threshold=float(cfg.get("match_threshold", 0.82)),
            step_timeout=float(cfg.get("step_timeout_seconds", 20)),
            screen_settle=float(cfg.get("screen_settle_seconds", 2.5)),
            tap_settle=float(cfg.get("tap_settle_seconds", 1.2)),
            dry_run=dry_run,
            pre_step_hook=_dismiss_hook,
        )
        if result.ok():
            log.info("Recovery: SUCCESS — flow completed on retry")
        else:
            log.warning("Recovery: retry also aborted at %s — giving up this cycle", result.aborted_at)

    # Final screenshot + cooldown OCR — captured while still on the LST Hunt
    # page. Uses retry-with-fresh-screencaps because the post-claim animation
    # can briefly hide a badge, making a single OCR pass under-count. The
    # daemon depends on these timers for adaptive scheduling — missing one
    # means the next wake is too late.
    stamp = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    final_path = LOGS / f"{stamp}_final.png"
    cooldowns_seconds: list[int] = []
    try:
        # Save one screenshot for audit, then run OCR with retries
        screen = device.screencap()
        import cv2
        cv2.imwrite(str(final_path), screen)
        cds = read_cooldowns_with_retry(
            device,
            max_attempts=int(cfg.get("ocr_max_attempts", 3)),
            inter_attempt_seconds=float(cfg.get("ocr_inter_attempt_seconds", 4.0)),
        )
        cooldowns_seconds = [c.seconds for c in cds]
    except Exception as e:
        log.warning("Could not save final screenshot / OCR cooldowns: %s", e)

    # Now that OCR is done, run the back-to-lobby step (tap < arrow) so the
    # cleanup BACK keystrokes land on the main lobby instead of overshooting
    # into the "Quit the game?" dialog.
    if back_step and result.ok() and not dry_run:
        from lib.flow import run_step
        run_step(
            device, back_step, TEMPLATES,
            threshold=float(cfg.get("match_threshold", 0.82)),
            step_timeout=float(cfg.get("step_timeout_seconds", 20)),
            settle_default=float(cfg.get("screen_settle_seconds", 2.5)),
            tap_settle=float(cfg.get("tap_settle_seconds", 1.2)),
            dry_run=dry_run,
        )

    # Cleanup — controlled by `shutdown_between_cycles` in config.yaml.
    # Graceful path: send HOME key (CODM gets the normal Android lifecycle
    # signals: onPause -> onStop, saves state), wait briefly, then emu kill
    # the AVD. NO force_stop — that triggers CODM's repair sequence on next
    # cold launch (30-90s wasted per cycle).
    shutdown_between = bool(cfg.get("shutdown_between_cycles", True))
    if dry_run:
        log.info("Cleanup: skipped (dry-run)")
    elif shutdown_between:
        log.info("Cleanup: HOME key (CODM saves state) then AVD shutdown")
        device.home()
        time.sleep(2.5)  # give CODM time to run onPause/onStop and flush state
        try:
            subprocess.run(
                ["adb", "-s", device.serial, "emu", "kill"],
                check=False, capture_output=True, timeout=10,
            )
            log.info("Sent emu kill to %s — AVD will exit", device.serial)
        except Exception as e:
            log.warning("emu kill failed: %s", e)
    else:
        log.info("Cleanup: pressing BACK 3x to return to main CODM lobby (AVD + CODM stay running)")
        for _ in range(3):
            device.back()
            time.sleep(0.8)

    # On failure, work out WHY (login form / stuck loading / network / popup /
    # crash) so the status + dashboard show something actionable, not just
    # "dmz_lobby_check not found".
    fail_detail = None
    fail_reason = None
    if not result.ok() and not dry_run:
        fail_detail = _classify_failure(device, pkg)
        if fail_detail == SESSION_CONFLICT_DETAIL:
            # Retrying would sign in again and kick the player off their phone.
            fail_reason = "session_conflict"

    summary = {
        "ok": result.ok(),
        "stamp": stamp,
        "claims_attempted": result.claims_attempted,
        "steps_run": result.steps_run,
        "steps_skipped": result.steps_skipped,
        "aborted_at": result.aborted_at,
        "abort_reason": result.abort_reason,
        "fail_detail": fail_detail,
        "reason": fail_reason,
        "final_screenshot": str(final_path) if final_path.exists() else None,
        "cooldowns_seconds": cooldowns_seconds,  # remaining seconds per card that's locked
        "min_cooldown_seconds": min(cooldowns_seconds) if cooldowns_seconds else None,
    }
    (LOGS / f"{stamp}_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Summary: %s", summary)

    # One-line human-readable summary for the user-facing status log.
    if not summary.get("ok"):
        msg = f"FAILED: {fail_detail or (summary.get('aborted_at') or 'unknown')}"
    else:
        claimed = summary.get("claims_attempted") or 0
        cds = summary.get("cooldowns_seconds") or []
        cd_str = ", ".join(f"{s/3600:.1f}h" for s in sorted(cds)) if cds else "(none read)"
        if claimed:
            msg = f"CLAIMED {claimed} reward(s) | cooldowns: {cd_str}"
        else:
            msg = f"nothing claimable | cooldowns: {cd_str}"
    log_status(msg)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Single DMZ LST Hunt claim attempt.")
    parser.add_argument("--dry-run", action="store_true", help="Log taps instead of performing them.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    cfg = load_config()
    summary = claim_once(cfg, dry_run_override=True if args.dry_run else None)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
