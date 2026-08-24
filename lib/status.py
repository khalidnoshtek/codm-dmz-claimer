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
        # Stage only status.json so we never sweep up unrelated working changes.
        add = _run(["git", "add", "docs/status.json"], repo_root)
        if add.returncode != 0:
            log.warning("git add status.json failed: %s", add.stderr.strip())
            return
        # Nothing staged (status unchanged) -> skip the commit quietly.
        diff = _run(["git", "diff", "--cached", "--quiet", "docs/status.json"], repo_root)
        if diff.returncode == 0:
            return
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        commit = _run(["git", "commit", "-m", f"chore(status): {stamp}", "docs/status.json"], repo_root)
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


def read_remote_trigger(repo_root: Path) -> int:
    """Fetch origin/main and return docs/trigger.json's `requested_at` epoch
    (0 on any error / missing file). Uses git rather than the raw CDN URL so
    the value is fresh (no ~5-min Fastly cache) and unauthenticated-rate-limit
    free. Also advances the origin/main tracking ref, which lets the next
    status push rebase cleanly onto any trigger commit."""
    try:
        f = _run(["git", "fetch", "origin", "main", "-q"], repo_root, timeout=30)
        if f.returncode != 0:
            return 0
        show = _run(["git", "show", "origin/main:docs/trigger.json"], repo_root, timeout=10)
        if show.returncode != 0:
            return 0
        return int(json.loads(show.stdout).get("requested_at", 0) or 0)
    except Exception:
        return 0
