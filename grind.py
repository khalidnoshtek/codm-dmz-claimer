#!/usr/bin/env python3
"""Join a match, sit out the match, come back, join the next one. Repeat.

CODM pays Battle Pass XP for *time in a match*, not for what you do in it, so
there is nothing to simulate here: the loop only has to press START, recognise
that a match is running, recognise when it has ended, and clear whatever the
results screens put in the way.

Deliberately NOT part of the daemon. This is a manual, attended tool you start
and stop yourself:

    .venv/bin/python grind.py --minutes 90
    .venv/bin/python grind.py --matches 10 --watch

It drives the same locked AVD as the claimer, so the first thing it does is
tell the claimer to stand down (the dashboard's "I'm playing" hold), and the
last thing it does is hand the AVD back.

One template has to be captured before the first run -- the lobby's START
button, which differs per mode and per CODM version:

    .venv/bin/python capture.py shot --name 20_start_match

Everything else reuses templates the claimer already has.
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from claimer import claim_once, load_config          # noqa: E402
from lib import cooldowns                            # noqa: E402
from lib.adb import LOCKED_AVD, AdbDevice, ensure_avd_running   # noqa: E402
from lib.status import record_run                                # noqa: E402
from lib.vision import find_close_button, find_template          # noqa: E402

TEMPLATES = ROOT / "templates"
START_TEMPLATE = TEMPLATES / "20_start_match.png"
# The small house icon on every CODM menu screen. Its presence is the cheapest
# reliable "we are in the menus, not in a match" signal, and it is already in
# the template library.
MENU_TEMPLATE = TEMPLATES / "08_home_icon.png"

log = logging.getLogger("grind")

_stopping = False


def _handle_stop(signum, frame) -> None:
    global _stopping
    _stopping = True
    log.info("Received signal %s — finishing up and handing the AVD back.", signum)


# --------------------------------------------------------------------------
# Keeping out of the claimer's way
# --------------------------------------------------------------------------
def _git(*args: str, timeout: float = 60) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(ROOT),
                          capture_output=True, text=True, timeout=timeout)


def set_claimer_hold(until_epoch: int) -> bool:
    """Tell the claimer to leave the AVD alone until `until_epoch`.

    Reuses the dashboard's "I'm playing" control rather than inventing a second
    mechanism: the daemon already honours play_until, already frees the AVD
    when it sees one, and already survives its own restart with the hold
    intact. Writing it here means the dashboard shows the pause too, so there
    is one place that says what the AVD is doing.

    The daemon reads the control from origin/main, so this has to be pushed,
    not just written. Returns False if that didn't work -- the caller warns
    rather than silently racing the claimer for the emulator.
    """
    path = ROOT / "docs" / "trigger.json"
    try:
        control = json.loads(path.read_text())
    except Exception:
        control = {}
    control["play_until"] = int(until_epoch)
    path.write_text(json.dumps(control, indent=2) + "\n")

    if _git("add", "docs/trigger.json").returncode != 0:
        return False
    if _git("diff", "--cached", "--quiet").returncode == 0:
        return True          # already said the same thing; nothing to push
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    if _git("commit", "-m", f"grind: hold the claimer until {stamp}").returncode != 0:
        return False
    if _git("push", "origin", "HEAD").returncode != 0:
        # Someone else (the daemon's own status push) got there first.
        _git("pull", "--rebase", "origin", "main")
        return _git("push", "origin", "HEAD").returncode == 0
    return True


# --------------------------------------------------------------------------
# Screen reading
# --------------------------------------------------------------------------
def _in_menus(device: AdbDevice, threshold: float) -> bool:
    """True if CODM is showing a menu screen (lobby, mode select, results)."""
    try:
        return find_template(device.screencap(), MENU_TEMPLATE, threshold=threshold) is not None
    except Exception:
        return False


def _results_words(screen) -> bool:
    """True if this looks like a post-match results/summary screen.

    Checked by OCR rather than a template because CODM shows several different
    end-of-match panels (match summary, BP progress, rank change, crate drop)
    and capturing one template per panel per season is a losing game. We only
    need to know "something is between us and the lobby" before it is safe to
    start tapping -- tapping blind during a live match is how you accidentally
    open the quit menu.
    """
    try:
        import cv2
        import pytesseract
        txt = pytesseract.image_to_string(cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)).upper()
    except Exception:
        return False
    hits = ("VICTORY", "DEFEAT", "MATCH SUMMARY", "RESULTS", "CONTINUE", "EXIT",
            "BATTLE PASS", "RANK UP", "ELIMINATED", "SPECTATE", "RETURN TO LOBBY",
            "PLACEMENT", "EXP ", "TAP TO CONTINUE")
    return sum(1 for h in hits if h in txt) >= 1


# --------------------------------------------------------------------------
# Claiming in the gaps between matches
# --------------------------------------------------------------------------
def claim_due() -> bool:
    """True if at least one LST Hunt card has come off cooldown.

    Reads the same remembered expiries the daemon schedules from, so the two
    agree about when something is actually claimable. No file (or an unreadable
    one) counts as due: not knowing is a reason to go and look, and the claim
    flow is cheap here -- the AVD is already up and we are already standing in
    the lobby.
    """
    try:
        d = json.loads((ROOT / "logs" / "cooldowns.json").read_text())
    except Exception:
        return True
    expiries = [int(e) for e in d.get("expiries", [])]
    if not expiries:
        return True
    return min(expiries) <= int(time.time())


def claim_between_matches(cfg: dict) -> None:
    """Run one claim cycle on the AVD that is already running.

    This is the reason the grind loop is worth having beyond XP: the daemon
    pays ~2 minutes of cold boot for every claim, and has to take the CODM
    session to do it. Here the emulator is already up and already signed in,
    so a claim costs the ~40 seconds of menu navigation and lands in the gap
    between two matches, where it cannot interrupt anything.

    shutdown_between_cycles is forced off for exactly that reason -- the
    claimer's normal cleanup would kill the emulator out from under the next
    match. With it off, cleanup walks BACK to the main lobby instead, which is
    where start_match expects to be.
    """
    if not claim_due():
        return
    log.info("A DMZ card is off cooldown — claiming before the next match")
    try:
        summary = claim_once({**cfg, "shutdown_between_cycles": False})
    except Exception as e:
        log.warning("Claim attempt failed (%s) — carrying on with the next match", e)
        return

    cds = summary.get("cooldowns_seconds") or []
    claims = int(summary.get("claims_attempted") or 0)
    if cds:
        cooldowns.save(cds, ROOT)
    elif claims:
        # Same post-claim blind spot the daemon has: the claim landed but the
        # page hid the badges, so rewind the claimed entries rather than let
        # the cache keep saying they are due.
        cooldowns.note_claim(claims, ROOT,
                             int(cfg.get("post_claim_cooldown_seconds", 14400)))
    record_run(summary, ROOT, retention_days=int(cfg.get("history_retention_days", 7)))
    log.info("Claim cycle: ok=%s claimed=%d cooldowns=%s",
             summary.get("ok"), claims, [round(c / 3600, 1) for c in cds])


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------
def wait_for_lobby(device: AdbDevice, cfg: dict, deadline: float) -> bool:
    """Get back to a menu screen, clearing whatever is on the way.

    Only acts once a screen looks like results -- see _results_words for why
    blind tapping is unsafe while a match may still be live.
    """
    thr = float(cfg.get("match_threshold", 0.82))
    taps = 0
    while time.time() < deadline and not _stopping:
        try:
            screen = device.screencap()
        except Exception as e:
            log.warning("screencap failed (%s) — AVD may be gone", e)
            return False
        if find_template(screen, MENU_TEMPLATE, threshold=thr):
            return True
        if _results_words(screen):
            x_hit = find_close_button(screen, TEMPLATES)
            if x_hit:
                log.info("results: close-X at (%d,%d)", x_hit.x, x_hit.y)
                device.tap(x_hit.x, x_hit.y)
            elif taps < 12:
                # CODM's summary panels advance on BACK; it is only reachable
                # here because we have already confirmed the match is over.
                device.back()
                taps += 1
            time.sleep(2.0)
            continue
        time.sleep(3.0)
    return False


def start_match(device: AdbDevice, cfg: dict) -> bool:
    """Press START on the lobby. False if the button never appeared."""
    thr = float(cfg.get("match_threshold", 0.82))
    if not START_TEMPLATE.exists():
        log.error("Missing %s — capture it first with:\n"
                  "    .venv/bin/python capture.py shot --name 20_start_match",
                  START_TEMPLATE.name)
        return False
    deadline = time.time() + 30
    while time.time() < deadline and not _stopping:
        try:
            hit = find_template(device.screencap(), START_TEMPLATE, threshold=thr)
        except Exception:
            return False
        if hit:
            log.info("START at (%d,%d) score=%.3f", hit.x, hit.y, hit.score)
            device.tap(hit.x, hit.y)
            return True
        time.sleep(1.5)
    log.warning("START button never appeared on the lobby")
    return False


def wait_until_in_match(device: AdbDevice, cfg: dict, timeout: float) -> bool:
    """Wait out matchmaking + map load. In-match == the menu icon is gone.

    Requires two consecutive menu-free reads: a single one fires on loading
    screens and transitions, which would make every match look like it ended
    the moment it started.
    """
    thr = float(cfg.get("match_threshold", 0.82))
    deadline = time.time() + timeout
    clear = 0
    while time.time() < deadline and not _stopping:
        if _in_menus(device, thr):
            clear = 0
        else:
            clear += 1
            if clear >= 2:
                return True
        time.sleep(4.0)
    return False


def run(args) -> int:
    cfg = load_config()
    thr = float(cfg.get("match_threshold", 0.82))
    pkg = str(cfg.get("package", "com.activision.callofduty.shooter"))

    session_end = time.time() + args.minutes * 60 if args.minutes else float("inf")
    # Hold the claimer for the whole session plus a margin, so it does not
    # wake into the middle of the last match.
    hold_until = int((session_end if args.minutes else time.time() + 6 * 3600) + 600)
    if set_claimer_hold(hold_until):
        log.info("Claimer held off until %s",
                 time.strftime("%H:%M:%S", time.localtime(hold_until)))
    else:
        log.warning("Could NOT push the claimer hold — it may boot the AVD mid-match. "
                    "Press 'I'm playing' on the dashboard, or stop the daemon.")

    ensure_avd_running(
        LOCKED_AVD,
        boot_timeout=float(cfg.get("avd_boot_timeout_seconds", 240)),
        headless=not args.watch,
        gpu_mode=str(cfg.get("emulator_gpu", "host")),
        virtio_wifi=bool(cfg.get("emulator_virtio_wifi", False)),
    )
    device = AdbDevice.auto()
    log.info("Driving %s", device.serial)
    device.keep_awake()
    lock_cfg = cfg.get("lockscreen") or {}
    device.disable_lock(old_credential="".join(str(d) for d in (lock_cfg.get("pattern") or [])) or None)

    if not device.is_app_foreground(pkg):
        log.info("Launching CODM…")
        device.launch_app(pkg, cfg.get("activity"))
        device.wait_app_foreground(pkg, timeout=120.0)
        time.sleep(float(cfg.get("cold_launch_settle_seconds", 25)))

    played = 0
    try:
        while not _stopping:
            if args.matches and played >= args.matches:
                log.info("Done — %d match(es) played.", played)
                break
            if time.time() >= session_end:
                log.info("Done — session time is up after %d match(es).", played)
                break

            if not wait_for_lobby(device, cfg, deadline=time.time() + 300):
                log.error("Couldn't get back to the lobby — stopping rather than "
                          "tapping blind.")
                break

            # Standing in the lobby between two matches is the one moment when
            # claiming is free: the AVD is up, the session is ours, and nothing
            # is interrupted. Checked every lap, so it covers both "before the
            # next match" and "after the one that just ended".
            if not args.no_claim:
                claim_between_matches(cfg)
                if not wait_for_lobby(device, cfg, deadline=time.time() + 120):
                    log.warning("Not back on the lobby after claiming — skipping a lap")
                    continue

            if not start_match(device, cfg):
                break
            if not wait_until_in_match(device, cfg, timeout=args.queue_timeout):
                log.warning("Match never started (still in the menus after %ds) — "
                            "trying again.", args.queue_timeout)
                continue

            played += 1
            log.info("Match %s started — waiting it out (cap %d min)",
                     played if not args.matches else f"{played}/{args.matches}",
                     args.match_timeout // 60)
            # The whole point is time in the match, so there is nothing to do
            # here but wait. The cap only exists so a match that never resolves
            # (disconnect, stuck spectate) cannot hang the session forever.
            if not wait_for_lobby(device, cfg, deadline=time.time() + args.match_timeout):
                log.warning("Match didn't end within the cap — restarting CODM to "
                            "get back to a known screen.")
                device.force_stop(pkg)
                time.sleep(3.0)
                device.launch_app(pkg, cfg.get("activity"))
                device.wait_app_foreground(pkg, timeout=120.0)
                time.sleep(float(cfg.get("cold_launch_settle_seconds", 25)))
    except KeyboardInterrupt:
        log.info("Interrupted.")
    finally:
        log.info("Played %d match(es).", played)
        if args.leave_avd_up:
            log.info("Leaving the AVD running (--leave-avd-up).")
        else:
            subprocess.run(["adb", "-s", device.serial, "emu", "kill"],
                           check=False, capture_output=True, timeout=10)
            log.info("AVD shut down.")
        # Hand the claimer back straight away rather than making it sit out the
        # rest of a hold that was sized for a session we are no longer in.
        if set_claimer_hold(int(time.time()) + 120):
            log.info("Claimer released — it resumes in ~2 minutes.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--minutes", type=int, default=0,
                   help="stop after this many minutes (0 = until --matches or Ctrl-C)")
    p.add_argument("--matches", type=int, default=0,
                   help="stop after this many matches (0 = no limit)")
    p.add_argument("--match-timeout", type=int, default=1800,
                   help="give up on a match that hasn't ended after this many seconds")
    p.add_argument("--queue-timeout", type=int, default=240,
                   help="seconds to wait for matchmaking + map load")
    p.add_argument("--watch", action="store_true",
                   help="show the emulator window instead of running headless")
    p.add_argument("--leave-avd-up", action="store_true",
                   help="don't shut the AVD down on exit")
    p.add_argument("--no-claim", action="store_true",
                   help="just grind; don't claim DMZ rewards between matches")
    args = p.parse_args()
    if not args.minutes and not args.matches:
        print("Refusing to run unbounded — pass --minutes and/or --matches.", file=sys.stderr)
        return 2

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
