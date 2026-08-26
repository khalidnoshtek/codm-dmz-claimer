"""Publish daemon status to docs/status.json for the GitHub Pages dashboard.

The dashboard (docs/index.html, served via GitHub Pages from /docs on main)
fetches status.json to show the last run and a live countdown to the next run,
so you can see at a glance when the claimer is about to take over the AVD and
avoid getting kicked out of a game mid-match.

Publishing is best-effort: any failure (git offline, no creds, etc.) is logged
and swallowed so it can never break a claim cycle.
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path

log = logging.getLogger(__name__)


def _run(args: list[str], cwd: Path, timeout: float = 30) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)


def write_status(status: dict, repo_root: Path) -> Path:
    """Write status.json into docs/. Always safe (local file write)."""
    docs = repo_root / "docs"
    docs.mkdir(exist_ok=True)
    status = {**status, "updated_at_epoch": int(time.time())}
    path = docs / "status.json"
    path.write_text(json.dumps(status, indent=2))
    return path


def publish_status(status: dict, repo_root: Path, push: bool = True) -> None:
    """Write docs/status.json and (best-effort) commit + push it so GitHub
    Pages serves the fresh status. Never raises."""
    try:
        write_status(status, repo_root)
    except Exception as e:
        log.warning("Could not write status.json: %s", e)
        return
    if not push:
        return
    try:
        # Stage status.json + history.json (the two dashboard data files) so
        # they ship in one commit. We never `git add -A`, to avoid sweeping up
        # unrelated working changes.
        add = _run(["git", "add", "docs/status.json", "docs/history.json"], repo_root)
        if add.returncode != 0:
            log.warning("git add dashboard files failed: %s", add.stderr.strip())
            return
        # Nothing staged (unchanged) -> skip the commit quietly.
        diff = _run(["git", "diff", "--cached", "--quiet"], repo_root)
        if diff.returncode == 0:
            return
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        commit = _run(["git", "commit", "-m", f"chore(status): {stamp}"], repo_root)
        if commit.returncode != 0:
            log.warning("git commit status failed: %s", commit.stderr.strip())
            return
        pushed = _run(["git", "push", "origin", "HEAD"], repo_root, timeout=60)
        if pushed.returncode != 0:
            # Remote likely moved (e.g. a manual-trigger commit written via the
            # GitHub API). Rebase our status commit on top and retry once.
            # status.json and trigger.json are different files, so this never
            # conflicts.
            log.info("status push rejected; rebasing on origin/main and retrying")
            _run(["git", "pull", "--rebase", "origin", "main"], repo_root, timeout=60)
            retry = _run(["git", "push", "origin", "HEAD"], repo_root, timeout=60)
            if retry.returncode != 0:
                log.warning("git push status failed after rebase: %s", retry.stderr.strip())
    except Exception as e:
        log.warning("publish_status push step failed: %s", e)


def _result_line(summary: dict | None) -> tuple[bool, int, str]:
    """(ok, claims, short result string) from a cycle summary."""
    if not isinstance(summary, dict):
        return False, 0, "FAILED (error)"
    if summary.get("reason") == "needs_login":
        return False, 0, "NEEDS LOGIN"
    ok = bool(summary.get("ok"))
    claims = summary.get("claims_attempted") or 0
    if not ok:
        return ok, claims, f"FAILED at {summary.get('aborted_at') or 'unknown'}"
    return ok, claims, (f"CLAIMED {claims}" if claims else "nothing claimable")


def record_run(summary: dict | None, repo_root: Path, retention_days: int = 7) -> None:
    """Append this cycle to docs/history.json, pruning entries older than
    retention_days. Keeps a rolling window the dashboard can total (claims/week)
    and that you can skim before deciding to trigger a manual run. Best-effort
    (write only — publish_status pushes it alongside status.json). Never raises."""
    try:
        docs = repo_root / "docs"
        docs.mkdir(exist_ok=True)
        path = docs / "history.json"
        try:
            hist = json.loads(path.read_text())
            if not isinstance(hist, list):
                hist = []
        except Exception:
            hist = []
        ok, claims, result = _result_line(summary)
        cds = sorted((summary or {}).get("cooldowns_seconds") or [])
        hist.append({
            "epoch": int(time.time()),
            "ok": ok,
            "result": result,
            "claims": claims,
            "cooldowns_hours": [round(s / 3600, 1) for s in cds],
        })
        cutoff = int(time.time()) - int(retention_days) * 86400
        hist = [h for h in hist if int(h.get("epoch", 0)) >= cutoff]
        path.write_text(json.dumps(hist, indent=2))
    except Exception as e:
        log.warning("record_run failed (non-fatal): %s", e)


def read_remote_control(repo_root: Path) -> dict:
    """Fetch origin/main and return the dashboard control state from
    docs/trigger.json: {"requested_at": <epoch>, "delay_until": <epoch>}
    (0s on any error / missing file). `requested_at` is a manual "run now"
    request; `delay_until` is a "push the next run to at least this time"
    request. Uses git rather than the raw CDN URL so the values are fresh (no
    ~5-min Fastly cache) and unauthenticated-rate-limit free. Also advances the
    origin/main tracking ref, so the next status push rebases cleanly onto any
    button commit."""
    try:
        f = _run(["git", "fetch", "origin", "main", "-q"], repo_root, timeout=30)
        if f.returncode != 0:
            return {"requested_at": 0, "delay_until": 0}
        show = _run(["git", "show", "origin/main:docs/trigger.json"], repo_root, timeout=10)
        if show.returncode != 0:
            return {"requested_at": 0, "delay_until": 0}
        d = json.loads(show.stdout)
        return {
            "requested_at": int(d.get("requested_at", 0) or 0),
            "delay_until": int(d.get("delay_until", 0) or 0),
        }
    except Exception:
        return {"requested_at": 0, "delay_until": 0}
