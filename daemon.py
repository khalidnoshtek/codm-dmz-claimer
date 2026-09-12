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
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

from claimer import claim_once, load_config, setup_logging
from lib.adb import AdbDevice, LOCKED_AVD
from lib.wake import schedule_wake, cancel_wakes
from lib.status import publish_status, read_remote_control, record_run


_last_diagnosis: dict | None = None  # {"epoch": int, "text": str} shown on the dashboard


def _diagnosis_prompt(summary: dict | None, extra: str = "") -> str:
    """Context bundle for Claude: what failed, the stuck screenshot, recent log."""
    detail = (summary or {}).get("fail_detail") or (summary or {}).get("reason") or "unknown"
    shot = (summary or {}).get("final_screenshot")
    if not shot:
        try:
            shots = sorted((ROOT / "logs").glob("*_final.png"), key=lambda p: p.stat().st_mtime)
            shot = str(shots[-1]) if shots else "(none)"
        except Exception:
            shot = "(none)"
    try:
        logtail = "\n".join((ROOT / "logs" / "claimer.log").read_text(errors="ignore").splitlines()[-60:])
    except Exception:
        logtail = "(log unavailable)"
    try:
        statustail = "\n".join((ROOT / "logs" / "status.log").read_text(errors="ignore").splitlines()[-15:])
    except Exception:
        statustail = ""
    return (
        "You are maintaining the CODM DMZ auto-claimer in this repo (claimer.py drives an Android "
        "AVD through CODM to claim DMZ LST Hunt rewards). " + extra +
        f"\nMost recent failure reason: '{detail}'.\n"
        f"Screenshot of the stuck screen: {shot} (open it with Read).\n\n"
        "Investigate: read the screenshot and the relevant code (claimer.py popup hook, "
        "lib/flow.py steps). Identify what is actually blocking it and the specific fix. "
        "Do NOT edit code.\n\n"
        f"Recent status log:\n{statustail}\n\nRecent daemon log tail:\n{logtail}\n\n"
        "End with one line starting 'DIAGNOSIS: ' summarizing the root cause in plain English."
    )


def _run_diagnosis_sync(cfg: dict, summary: dict | None) -> str | None:
    """On-demand (dashboard button): run Claude Code's analysis and return the
    diagnosis text so it can be shown on the dashboard. Report-only."""
    import shutil
    global _last_diagnosis
    claude = shutil.which("claude")
    log = logging.getLogger("daemon")
    if not claude:
        _last_diagnosis = {"epoch": int(time.time()), "text": "claude CLI not found on PATH"}
        return None
    prompt = _diagnosis_prompt(summary, extra="The user pressed 'Analyze via Claude' on the dashboard. ")
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = ROOT / "logs" / f"diagnosis-{ts}.txt"
    log.info("on-demand diagnosis: running Claude Code (report-only)...")
    try:
        r = subprocess.run([claude, "-p", prompt, "--model", "sonnet"], cwd=str(ROOT),
                           capture_output=True, text=True,
                           timeout=float(cfg.get("diagnose_timeout_seconds", 420)))
        text = (r.stdout or "").strip() or (r.stderr or "").strip()
    except subprocess.TimeoutExpired:
        text = "analysis timed out"
    except Exception as e:
        text = f"analysis failed: {e}"
    try:
        out.write_text(text)
    except Exception:
        pass
    # Prefer the explicit DIAGNOSIS line; else the tail of the reply.
    line = next((l for l in reversed(text.splitlines()) if l.strip().upper().startswith("DIAGNOSIS:")), "")
    short = (line.strip() or text.strip()[-400:]) or "no output"
    _last_diagnosis = {"epoch": int(time.time()), "text": short[:600]}
    log.info("on-demand diagnosis complete -> %s", out.name)
    return short


def _auto_diagnose(cfg: dict, summary: dict | None) -> str | None:
    """After a cycle fully fails (all retries), optionally invoke Claude Code
    headlessly to look at the failure screenshot + logs and diagnose (and, if
    auto_diagnose_apply_fixes is on, fix) the problem. Rate-limited and
    best-effort. Returns a short note for the dashboard, or None."""
    import shutil
    if not bool(cfg.get("auto_diagnose_on_failure", True)):
        return None
    claude = shutil.which("claude")
    if not claude:
        return None
    log = logging.getLogger("daemon")
    marker = ROOT / "logs" / ".last_diagnose"
    cooldown = float(cfg.get("auto_diagnose_cooldown_hours", 6)) * 3600
    try:
        if marker.exists() and (time.time() - marker.stat().st_mtime) < cooldown:
            return None  # already diagnosed recently — don't spam / burn tokens
    except Exception:
        pass

    detail = (summary or {}).get("fail_detail") or (summary or {}).get("reason") or "unknown"
    shot = (summary or {}).get("final_screenshot") or "(none)"
    try:
        logtail = "\n".join((ROOT / "logs" / "claimer.log").read_text(errors="ignore").splitlines()[-60:])
    except Exception:
        logtail = "(log unavailable)"
    # REPORT-ONLY: Claude reads the screenshot + logs and writes a diagnosis.
    # It is NOT given autonomous edit/commit powers (no skip-permissions) — an
    # unattended daemon rewriting its own live code is unsafe. Actual fixes stay
    # a reviewed, manual step (read logs/diagnosis-*.txt, then apply).
    prompt = (
        "You are maintaining the CODM DMZ auto-claimer in this repo. Its daemon just failed a "
        f"full claim cycle after all retries. Reported reason: '{detail}'. A screenshot of the "
        f"stuck screen is at: {shot} (open it with Read). Investigate (read the screenshot and "
        "relevant code) and write a concise diagnosis + the specific fix you'd suggest. Do NOT "
        "edit code.\n\nRecent daemon log tail:\n" + logtail + "\n\n"
        "End with one line starting 'DIAGNOSIS: ' summarizing the root cause."
    )
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = ROOT / "logs" / f"diagnosis-{ts}.txt"
    try:
        with out.open("w") as f:
            subprocess.Popen([claude, "-p", prompt, "--model", "sonnet"],
                             cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT,
                             start_new_session=True)
        marker.parent.mkdir(exist_ok=True)
        marker.touch()
        log.info("auto-diagnose: launched Claude Code (report-only) -> %s", out.name)
        return f"auto-diagnosis (report) running — see logs/{out.name}"
    except Exception as e:
        log.warning("auto-diagnose failed to launch: %s", e)
        return None


def _clean_boot_reset() -> None:
    """Kill the locked AVD so the next attempt cold-boots a fresh, healthy
    emulator. Recovers from a wedged state (SystemUI ANR, stuck launch) that a
    warm reuse or in-app retry can't fix. Best-effort; waits for the port to
    free before the next launch."""
    log = logging.getLogger("daemon")
    try:
        for d in AdbDevice.list_devices():
            if d.startswith("emulator-") and AdbDevice.emulator_avd_name(d) == LOCKED_AVD:
                subprocess.run(["adb", "-s", d, "emu", "kill"],
                               check=False, capture_output=True, timeout=10)
                log.info("clean-boot reset: killed %s", d)
    except Exception as e:
        log.warning("clean-boot reset failed: %s", e)
    time.sleep(8)

ROOT = Path(__file__).resolve().parent


def _result_text(summary: dict | None) -> str:
    """One-line human-readable outcome, mirroring claimer's status.log line."""
    if not isinstance(summary, dict):
        return "unknown"
    if summary.get("reason") == "needs_login":
        return "NEEDS LOGIN"
    if not summary.get("ok"):
        return "Failed"
    claimed = summary.get("claims_attempted") or 0
    return f"CLAIMED {claimed} reward(s)" if claimed else "nothing claimable"


def _publish(cfg: dict, state: str, *, summary: dict | None = None,
             wake_at: float | None = None, source: str | None = None) -> None:
    """Best-effort push of the current status to docs/status.json for the
    GitHub Pages dashboard. Controlled by config `publish_status` (default on)."""
    if not bool(cfg.get("publish_status", True)):
        return
    status: dict = {"state": state}
    if summary is not None:
        cds = sorted(summary.get("cooldowns_seconds") or [])
        status["last_run"] = {
            "epoch": int(time.time()),
            "ok": bool(summary.get("ok")),
            "result": _result_text(summary),
            "detail": summary.get("fail_detail"),  # why it failed, in plain English
            "claims": summary.get("claims_attempted") or 0,
            "cooldowns_hours": [round(s / 3600, 1) for s in cds],
        }
    if wake_at is not None:
        status["next_run"] = {"epoch": int(wake_at), "source": source or ""}
    if _last_diagnosis:
        status["diagnosis"] = _last_diagnosis
    publish_status(status, ROOT, push=bool(cfg.get("publish_status_push", True)))


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
    prev_summary: dict | None = None
    # Baseline the trigger/delay markers to whatever's already on the remote so
    # we don't act on a stale request the moment the daemon (re)starts.
    _ctl0 = read_remote_control(ROOT)
    last_trigger = _ctl0["requested_at"]
    last_delay = _ctl0["delay_until"]
    last_run_at = _ctl0["run_at"]
    last_fix_at = _ctl0.get("fix_requested_at", 0)
    while not _stopping:
        cfg = load_config()  # re-read each cycle so config changes take effect
        period = float(cfg.get("loop_period_seconds", 10800))
        jitter = float(cfg.get("loop_jitter_seconds", 600))

        # Tell the dashboard a cycle is starting (carry the previous run's
        # result so it stays visible while this one runs).
        _publish(cfg, "running", summary=prev_summary)

        # Self-healing: try the cycle up to cycle_max_attempts. On failure,
        # clean-boot the AVD (fresh emulator) and retry — recovers transient
        # issues (SystemUI ANR, slow/stuck cold launch, a blocking popup, a
        # network blip) without waiting a whole sleep period. We do NOT retry a
        # needs-login failure — that needs a manual sign-in, and churning the
        # AVD would just ANR it.
        summary = None  # bound even if claim_once throws (used below + in _publish)
        max_attempts = max(1, int(cfg.get("cycle_max_attempts", 3)))
        for attempt in range(1, max_attempts + 1):
            try:
                log.info("--- Cycle start (attempt %d/%d) ---", attempt, max_attempts)
                summary = claim_once(cfg)
            except Exception as e:
                log.error("Cycle threw: %s\n%s", e, traceback.format_exc())
                summary = None
            if isinstance(summary, dict) and summary.get("ok"):
                consecutive_failures = 0
                log.info("Cycle ok — claims_attempted=%s", summary.get("claims_attempted"))
                break
            reason = (summary or {}).get("fail_detail") or (summary or {}).get("reason") or "unknown"
            if isinstance(summary, dict) and summary.get("reason") == "needs_login":
                consecutive_failures += 1
                log.warning("Needs login — manual sign-in required, not retrying")
                break
            if attempt < max_attempts and not _stopping:
                log.warning("Attempt %d/%d failed (%s) — clean-booting AVD and retrying",
                            attempt, max_attempts, reason)
                _publish(cfg, "running", summary=prev_summary)
                _clean_boot_reset()
            else:
                consecutive_failures += 1
                log.warning("Cycle failed after %d attempt(s): %s (consecutive_failures=%d)",
                            attempt, reason, consecutive_failures)

        if _stopping:
            break

        # Cycle fully failed (all retries) and it's not a needs-login case ->
        # optionally call Claude Code to look at the screenshot + logs and
        # diagnose/fix. Rate-limited inside _auto_diagnose.
        if not (isinstance(summary, dict) and summary.get("ok")) \
                and not (isinstance(summary, dict) and summary.get("reason") == "needs_login"):
            _auto_diagnose(cfg, summary)

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

        # Append this run to the rolling history, then publish last-run result
        # + next-run time for the dashboard countdown.
        record_run(summary, ROOT, retention_days=int(cfg.get("history_retention_days", 7)))
        _publish(cfg, "sleeping", summary=summary, wake_at=wake_at, source=source)
        prev_summary = summary if isinstance(summary, dict) else prev_summary

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
        # Poll the repo for a manual "run now" trigger from the dashboard button
        # while we sleep. A newer trigger.json timestamp breaks the sleep and
        # runs a cycle immediately.
        trigger_enabled = bool(cfg.get("trigger_enabled", True))
        trigger_poll = max(15.0, float(cfg.get("trigger_poll_seconds", 45)))
        last_poll = 0.0
        deadline = time.time() + wait
        while not _stopping:
            now = time.time()
            if now >= deadline:
                break
            if trigger_enabled and now - last_poll >= trigger_poll:
                last_poll = now
                ctl = read_remote_control(ROOT)
                # "Analyze via Claude" button: run the diagnosis now and publish
                # the result so it shows on the dashboard. Report-only.
                if ctl.get("fix_requested_at", 0) > last_fix_at:
                    last_fix_at = ctl["fix_requested_at"]
                    log.info("Dashboard requested a Claude analysis — running now")
                    _publish(cfg, "sleeping", summary=summary, wake_at=deadline, source=source)
                    _run_diagnosis_sync(cfg, summary)
                    _publish(cfg, "sleeping", summary=summary, wake_at=deadline, source=source)
                    last_poll = time.time()
                    continue
                if ctl["requested_at"] > last_trigger:
                    last_trigger = ctl["requested_at"]
                    log.info("Manual trigger received (requested_at=%d) — running a cycle now",
                             ctl["requested_at"])
                    break
                # Reschedule request: move the next run to an absolute time —
                # earlier (prepone) OR later (delay). run_at <= now means "now".
                if ctl["run_at"] > last_run_at:
                    last_run_at = ctl["run_at"]
                    if ctl["run_at"] <= now + 5:
                        log.info("Reschedule to now — running a cycle")
                        break
                    deadline = float(ctl["run_at"])
                    log.info("Next run rescheduled to %s",
                             time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(deadline)))
                    if cfg.get("sleep_mac_between_cycles", True) and deadline - now > 600:
                        schedule_wake(datetime.now() + timedelta(seconds=deadline - now - 30))
                    _publish(cfg, "sleeping", summary=summary, wake_at=deadline, source="manual reschedule")
                    continue
                # Delay ("snooze") request: push the wake time out to at least
                # delay_until, so a scheduled run won't interrupt a game.
                if ctl["delay_until"] > last_delay:
                    last_delay = ctl["delay_until"]
                    if ctl["delay_until"] > deadline:
                        deadline = float(ctl["delay_until"])
                        log.info("Delay requested — next run pushed to %s",
                                 time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(deadline)))
                        if cfg.get("sleep_mac_between_cycles", True) and deadline - now > 600:
                            schedule_wake(datetime.now() + timedelta(seconds=deadline - now - 30))
                        # Republish so the dashboard countdown reflects the delay.
                        _publish(cfg, "sleeping", summary=summary, wake_at=deadline,
                                 source="manual delay")
            chunk = min(5.0, deadline - now)
            time.sleep(chunk)

    # Cancel any pending wake so we don't leave a stale alarm queued.
    cancel_wakes()
    log.info("Daemon exiting cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(loop_forever())
