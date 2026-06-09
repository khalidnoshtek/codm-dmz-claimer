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
from lib.ocr import read_cooldowns, read_cooldowns_with_retry

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


def load_config() -> dict:
    cfg_path = ROOT / "config.yaml"
    with cfg_path.open() as f:
        return yaml.safe_load(f) or {}


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
        ensure_avd_running(LOCKED_AVD, boot_timeout=float(cfg.get("avd_boot_timeout_seconds", 240)))
    except Exception as e:
        log.error("Could not start/find locked AVD %s: %s", LOCKED_AVD, e)
        return {"ok": False, "reason": "avd_not_running", "target_avd": LOCKED_AVD, "error": str(e)}

    device = AdbDevice.auto()
    log.info("ADB device: %s (dry_run=%s)", device.serial, dry_run)

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
            return {"ok": False, "reason": "app_unstable_on_cold_launch", "package": pkg, "attempts": max_attempts}
    else:
        log.info("App %s already foregrounded (warm)", pkg)
        time.sleep(float(cfg.get("screen_settle_seconds", 2.5)) * 2)

    # We need OCR to run while we're still on the LST Hunt screen — that's
    # the only screen where the "Remaining HH:MM:SS" badges are visible.
    # Splitting the flow: run everything up to (but not including) the
    # back_to_lobby step, capture the screen + OCR, then run the back step.
    nav_steps = [s for s in DEFAULT_STEPS if s.name != "back_to_lobby"]
    back_step = next((s for s in DEFAULT_STEPS if s.name == "back_to_lobby"), None)

    result = run_flow(
        device,
        nav_steps,
        TEMPLATES,
        threshold=float(cfg.get("match_threshold", 0.82)),
        step_timeout=float(cfg.get("step_timeout_seconds", 20)),
        screen_settle=float(cfg.get("screen_settle_seconds", 2.5)),
        tap_settle=float(cfg.get("tap_settle_seconds", 1.2)),
        dry_run=dry_run,
    )

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

    summary = {
        "ok": result.ok(),
        "stamp": stamp,
        "claims_attempted": result.claims_attempted,
        "steps_run": result.steps_run,
        "steps_skipped": result.steps_skipped,
        "aborted_at": result.aborted_at,
        "abort_reason": result.abort_reason,
        "final_screenshot": str(final_path) if final_path.exists() else None,
        "cooldowns_seconds": cooldowns_seconds,  # remaining seconds per card that's locked
        "min_cooldown_seconds": min(cooldowns_seconds) if cooldowns_seconds else None,
    }
    (LOGS / f"{stamp}_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Summary: %s", summary)
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
