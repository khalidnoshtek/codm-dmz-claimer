"""macOS auto-wake scheduling via `pmset schedule wake`.

Lets the daemon sleep the Mac between cycles and wake it back up just before
the next cooldown expires. Requires a NOPASSWD sudoers rule for:
    /usr/bin/pmset schedule *

Install with:
    echo 'khalid ALL=(root) NOPASSWD: /usr/bin/pmset schedule *' > /tmp/x
    sudo install -o root -g wheel -m 440 /tmp/x /etc/sudoers.d/codm-claimer

If the rule isn't installed, schedule_wake() returns False and the daemon
falls back to plain Python sleep (which means the Mac must be kept on).
"""
from __future__ import annotations

import logging
import subprocess
from datetime import datetime

log = logging.getLogger(__name__)


def _can_run_pmset_without_password() -> bool:
    """Returns True if our sudoers rule is set up. Tested by attempting
    `sudo -n pmset -g sched` — non-interactive sudo fails fast if a
    password would be required."""
    try:
        # `pmset -g sched` is a read-only listing — safe to probe. But our
        # NOPASSWD rule covers `pmset schedule *`, not `pmset -g`. Use the
        # same syntax the daemon actually runs: a no-op `pmset schedule
        # cancelall` on an already-empty queue is harmless.
        result = subprocess.run(
            ["sudo", "-n", "/usr/bin/pmset", "schedule", "cancelall"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def schedule_wake(when: datetime) -> bool:
    """Tell macOS to wake the Mac from sleep at `when`. Returns True on
    success. Cancels any previously-scheduled wake first so we don't pile
    up entries across cycles.

    Caller is responsible for choosing a sane wake time — we don't validate
    that `when` is in the future. macOS will silently drop wakes scheduled
    for the past.
    """
    if not _can_run_pmset_without_password():
        log.warning(
            "Mac auto-wake is unavailable: sudoers rule for 'pmset schedule *' "
            "is missing. Falling back to plain sleep — the Mac must stay on."
        )
        return False
    # Format pmset expects: "MM/dd/yy HH:mm:ss"
    when_str = when.strftime("%m/%d/%y %H:%M:%S")
    # Cancel previous wakes first so they don't queue up. cancelall is
    # idempotent — no harm if queue is already empty.
    try:
        subprocess.run(
            ["sudo", "-n", "/usr/bin/pmset", "schedule", "cancelall"],
            check=False, capture_output=True, text=True, timeout=5,
        )
    except Exception as e:
        log.warning("pmset cancelall failed: %s (continuing)", e)
    try:
        result = subprocess.run(
            ["sudo", "-n", "/usr/bin/pmset", "schedule", "wake", when_str],
            check=False, capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            log.warning("pmset schedule wake failed (rc=%d): %s", result.returncode, result.stderr.strip())
            return False
        log.info("Scheduled Mac auto-wake at %s", when_str)
        return True
    except Exception as e:
        log.warning("pmset schedule wake threw: %s", e)
        return False


def sleep_now() -> None:
    """Trigger an immediate sleep. Not currently called by the daemon —
    macOS will idle-sleep on its own after the configured idle timeout
    (System Settings -> Battery -> Sleep). Manual sleep is here in case
    we want it later."""
    try:
        subprocess.run(["pmset", "sleepnow"], check=False, capture_output=True, timeout=5)
    except Exception as e:
        log.warning("pmset sleepnow failed: %s", e)


def cancel_wakes() -> None:
    """Cancel any pending wake events. Called on daemon shutdown so we
    don't leave a stale wake scheduled."""
    try:
        subprocess.run(
            ["sudo", "-n", "/usr/bin/pmset", "schedule", "cancelall"],
            check=False, capture_output=True, text=True, timeout=5,
        )
    except Exception:
        pass
