"""Remember cooldown expiries across cycles.

OCR reads the LST Hunt timers as *relative* seconds ("Remaining 03:44:28"),
which are only meaningful at the moment they were read. Storing them as
*absolute* epochs makes them reusable: a reading taken an hour ago still
tells us exactly when that card unlocks.

This matters because the one moment OCR reliably fails is the moment right
after a claim, when the post-claim state hides the badges — so the cycle
that most needs a schedule is the one that can't read one. Previously that
fell back to a blind fixed period (capped at max_sleep_seconds), waking the
emulator every ~90 minutes regardless of when anything was actually
claimable. With expiries persisted, the failed read falls back to the last
good one instead.

Cards that were claimable (and so have already expired) drop out naturally:
their expiry is in the past, so they're filtered on load.

Best-effort throughout — scheduling must never break because a cache file
is missing or corrupt.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)

_FILE = "cooldowns.json"


def _path(repo_root: Path) -> Path:
    return repo_root / "logs" / _FILE


def save(seconds_list: list[int], repo_root: Path) -> None:
    """Record a fresh OCR reading as absolute expiry epochs. No-op on an
    empty reading, so a failed cycle never overwrites good data."""
    if not seconds_list:
        return
    try:
        now = int(time.time())
        p = _path(repo_root)
        p.parent.mkdir(exist_ok=True)
        p.write_text(json.dumps({
            "saved_at": now,
            "count_at_save": len(seconds_list),
            "expiries": sorted(now + int(s) for s in seconds_list),
        }, indent=2))
    except Exception as e:
        log.debug("could not persist cooldowns: %s", e)


def load(repo_root: Path) -> tuple[list[int], int]:
    """Return (remaining seconds for still-future expiries, how many timers
    the saved reading contained). ([], 0) if there's nothing usable.

    count_at_save is returned so the caller can apply the same OCR-confidence
    rules it would to a live reading: a cache built from a partial 2-of-3
    read is no more trustworthy now than it was then.
    """
    try:
        d = json.loads(_path(repo_root).read_text())
        now = int(time.time())
        remaining = [int(e) - now for e in d.get("expiries", []) if int(e) > now]
        if not remaining:
            return [], 0
        return sorted(remaining), int(d.get("count_at_save", 0) or 0)
    except Exception:
        return [], 0
