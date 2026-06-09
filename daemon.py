#!/usr/bin/env python3
"""Continuous claim loop.

Sleeps `loop_period_seconds` (+/- jitter) between attempts. Designed to be
kept running as a background process (nohup / launchd / `tmux`). Each
iteration re-reads config.yaml so you can adjust intervals without restarting.

Why 3h default: two of the DMZ LST Hunt rewards refresh every 3-4h, the
bigger one every 8h. Running every 3h means each reward is claimed within
at most ~1h of becoming available, with no missed cycles.
"""
from __future__ import annotations

import logging
import random
import signal
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

from claimer import claim_once, load_config, setup_logging
from lib.wake import schedule_wake, cancel_wakes

ROOT = Path(__file__).resolve().parent


_stopping = False


def _handle_stop(signum, _frame):
    global _stopping
    logging.getLogger("daemon").info("Received signal %s — finishing current cycle then exiting.", signum)
    _stopping = True


def loop_forever() -> int:
    setup_logging()
    log = logging.getLogger("daemon")
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    consecutive_failures = 0
    while not _stopping:
        cfg = load_config()  # re-read each cycle so config changes take effect
        period = float(cfg.get("loop_period_seconds", 10800))
        jitter = float(cfg.get("loop_jitter_seconds", 600))

        try:
            log.info("--- Cycle start ---")
            summary = claim_once(cfg)
            if summary["ok"]:
                consecutive_failures = 0
                log.info("Cycle ok — claims_attempted=%s", summary.get("claims_attempted"))
            else:
                consecutive_failures += 1
                log.warning("Cycle aborted at %s: %s (consecutive_failures=%d)",
                            summary.get("aborted_at"), summary.get("abort_reason"), consecutive_failures)
        except Exception as e:
            consecutive_failures += 1
            log.error("Cycle threw: %s\n%s", e, traceback.format_exc())

        if _stopping:
            break

        # Back off a bit on repeated failures so we don't hammer a broken AVD.
        backoff = min(consecutive_failures, 4) * 300  # +5min per fail, capped at +20min

        # Adaptive next-wake: prefer the soonest cooldown the OCR found on screen.
        # That way, as rarity tiers raise the cooldown (3h -> 4h -> 5h+), we
        # auto-track it without changing config. The config period is the
        # *fallback* used when OCR couldn't read any timers (e.g. all claimed,
        # nav failed, or popup blocking the cards).
        ocr_seconds = None
        ocr_count = 0
        all_cooldowns: list[int] = []
        try:
            if isinstance(summary, dict):
                ocr_seconds = summary.get("min_cooldown_seconds")
                all_cooldowns = sorted(summary.get("cooldowns_seconds") or [])
                ocr_count = len(all_cooldowns)
        except NameError:
            pass
        buffer = 120  # 2-minute safety so we don't arrive a few seconds early
        if all_cooldowns:
            # Cluster nearby cooldowns: starting from the soonest, include any
            # subsequent cooldown within BURST_WINDOW seconds of the running
            # max. Wake at the MAX of the cluster, so all cards in the burst
            # are claimable in one cycle.
            BURST_WINDOW = float(cfg.get("burst_window_seconds", 600))  # 10min default
            cluster_max = all_cooldowns[0]
            for cd in all_cooldowns[1:]:
                if cd - cluster_max <= BURST_WINDOW:
                    cluster_max = cd
                else:
                    break
            base = cluster_max + buffer
            if cluster_max == all_cooldowns[0]:
                source = f"OCR (min cooldown {ocr_seconds}s, found {ocr_count}/3)"
            else:
                in_cluster = [c for c in all_cooldowns if c <= cluster_max]
                source = (f"OCR (burst cluster of {len(in_cluster)} cooldowns, "
                          f"min={all_cooldowns[0]}s max={cluster_max}s, found {ocr_count}/3)")
        else:
            base = period
            source = "config period"
        # Two-tier cap based on OCR confidence:
        #   - Found all 3 timers -> trust the schedule, cap at max_sleep_seconds
        #   - Found fewer than 3 -> can't trust min(), use the tighter
        #     low_confidence_sleep_seconds so we re-check soon. This handles
        #     the case where OCR missed a short cooldown entirely and the
        #     read-out min was actually from a much-longer-cooldown card.
        max_sleep = float(cfg.get("max_sleep_seconds", 7200))            # 2h default
        low_conf_sleep = float(cfg.get("low_confidence_sleep_seconds", 1200))  # 20m default
        if ocr_count < 3 and ocr_seconds:
            tighter = low_conf_sleep
            if base > tighter:
                log.info("Low-confidence OCR (%d/3 timers): capping sleep at %.0fs instead of %.0fs",
                         ocr_count, tighter, max_sleep)
                base = tighter
                source = f"{source} -> low-confidence cap"
        elif base > max_sleep:
            log.info("Capping sleep: OCR said %ds but max_sleep_seconds=%.0f", int(base), max_sleep)
            base = max_sleep
            source = f"{source} -> capped at max_sleep_seconds"
        wait = max(60.0, base + random.uniform(-jitter, jitter) + backoff)
        wake_at = time.time() + wait
        log.info("Sleeping %.0fs (source=%s, jitter=±%.0f, backoff=%ds) -> next ~ %s",
                 wait, source, jitter, backoff,
                 time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(wake_at)))

        # Schedule macOS to wake itself just before the sleep ends. Only worth
        # the round-trip if the wait is long enough (short waits leave no time
        # for the Mac to actually sleep). The pmset wake is scheduled ~30s
        # BEFORE the daemon's wake time so the Mac has time to come up and
        # services to settle before Python resumes.
        if cfg.get("sleep_mac_between_cycles", True) and wait > 600:
            pmset_wake = datetime.now() + timedelta(seconds=wait - 30)
            schedule_wake(pmset_wake)

        # Sleep in small chunks so SIGTERM is responsive. Use an absolute
        # wall-clock deadline rather than a counter — Python's time.sleep()
        # accumulates ~80ms of overhead per call, which over 1500+ iterations
        # of 5s chunks drifts the daemon ~130s late on a multi-hour wait.
        deadline = time.time() + wait
        while not _stopping:
            now = time.time()
            if now >= deadline:
                break
            chunk = min(5.0, deadline - now)
            time.sleep(chunk)

    # Cancel any pending wake so we don't leave a stale alarm queued.
    cancel_wakes()
    log.info("Daemon exiting cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(loop_forever())
