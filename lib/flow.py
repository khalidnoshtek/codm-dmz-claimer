"""High-level claim sequence for DMZ LST Hunt rewards.

The flow is just a list of `Step`s. Each step waits for a template to appear
on screen, taps it (or all instances of it), and moves on. If a step's
template never shows up, we either retry, skip, or abort depending on the
step's `on_missing` policy.

The list below is the EXPECTED shape of the navigation — you'll need to
capture templates for each step from your own AVD running CODM. See README
"Capturing templates" for the workflow.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from .adb import AdbDevice
from .vision import find_template, find_all

log = logging.getLogger(__name__)


OnMissing = Literal["abort", "skip", "retry_back"]


@dataclass
class Step:
    name: str
    template: str                       # filename inside templates/
    on_missing: OnMissing = "abort"     # what to do if the template never appears
    tap_all: bool = False               # tap every match instead of just the first
    tap: bool = True                    # tap the match (False = verify-only / "we're on the right screen")
    settle_after: float | None = None   # override the default per-step settle delay
    timeout_override: float | None = None  # per-step timeout, overrides config.step_timeout_seconds
    threshold_override: float | None = None  # per-step template-match threshold; loosen for distinctive text-only templates
    optional: bool = False              # alias for on_missing="skip", reads nicer in the list
    pre_swipe: tuple[int, int, int, int, int] | None = None  # (x1,y1,x2,y2,duration_ms) before the search
    # After each tap, send an additional "anywhere" tap to dismiss a transient popup
    # (the LST Hunt 'You got X' modal closes on any tap; rather than template-match the
    # OK button, we just blast a tap at a known-safe spot like the screen center).
    dismiss_after_tap: tuple[int, int] | None = None
    dismiss_pause_seconds: float = 1.5  # how long to wait for the popup to appear before dismissing
    notes: str = ""                     # human-readable context, ignored at runtime


# Navigation path derived from your recorded flow (DMZ lobby -> BLACK MARKET ->
# UNDERGROUND OFFERS -> LST HUNT -> tap every "Tap to Claim" badge). Each step
# below maps 1:1 to a screen transition in the video.
#
# IMPORTANT: the PNG files referenced here must be captured from YOUR AVD
# running CODM, not from the phone recording. The phone is 2316x1080, the AVD
# in landscape ends up ~2400x1080 — UI scales differ enough that templates
# don't transfer cleanly. Use `python capture.py interactive --name <NN_name>`
# from each screen on the AVD to grab them.
DEFAULT_STEPS: list[Step] = [
    Step(
        name="dismiss_popups",
        template="10_popup_close_x.png",
        tap_all=True,
        on_missing="skip",
        timeout_override=5.0,
        settle_after=1.0,
        notes="CODM throws promo popups ('DON'T MISS OUT', battle pass, event banners) "
              "after launch. tap_all + loop-until-dry will tap every X, re-scan, until "
              "no more are showing. settle_after=1.0 shaves time off each round (default "
              "2.5s was conservative); the next popup renders in well under a second.",
    ),
    Step(
        name="tap_home_icon",
        template="08_home_icon.png",
        on_missing="skip",
        timeout_override=1.5,
        notes="The small HOUSE icon visible top-left on CODM mode screens (MULTIPLAYER, "
              "BR, etc.). Tapping returns to the main lobby. Skip-fast (1.5s) for the "
              "common case where we just dismissed popups and are already on main lobby.",
    ),
    Step(
        name="enter_dmz_mode",
        template="00_dmz_recon_mode.png",
        on_missing="skip",
        timeout_override=5.0,
        notes="From the CODM main lobby, tap DMZ: RECON to switch into DMZ mode. "
              "Skip-on-missing because in scenario 2 (warm relaunch landed directly in "
              "DMZ lobby) this template won't match. Short timeout for the same reason — "
              "if DMZ:RECON isn't on screen within 5s of arriving here, we're not on the "
              "main lobby and the next step (dmz_lobby_check) will validate.",
    ),
    Step(
        name="dmz_lobby_check",
        template="01_dmz_lobby_marker.png",
        tap=False,                         # verify-only — the NEXT step is the tap target
        on_missing="abort",
        notes="Sanity check that we landed in the DMZ lobby (BLACK MARKET button visible). "
              "If this fails, we're on some unexpected screen and recovery via BACK keys "
              "alone isn't reliable — better to abort and let the next cycle cold-boot.",
    ),
    Step(
        name="tap_black_market",
        template="02_black_market_button.png",
        notes="The bright cyan hex BLACK MARKET button bottom-left of the DMZ lobby.",
    ),
    Step(
        name="tap_black_market_inner",
        template="02b_black_market_inner.png",
        notes="On the BLACK MARKET / CRAFTING choice screen (hooded operator centered), "
              "tap the BLACK MARKET button on the LEFT — not CRAFTING on the right.",
    ),
    Step(
        name="tap_underground_offers",
        template="03_underground_offers_card.png",
        notes="The middle card on the Black Market screen (skull image, label 'UNDERGROUND OFFERS'). "
              "Crop the label, not the artwork — the skull animates and won't match reliably.",
    ),
    Step(
        name="tap_lst_hunt_tab",
        template="04_lst_hunt_tab.png",
        on_missing="abort",
        notes="The 'LST HUNT' tab in the left sidebar of UNDERGROUND OFFERS. The screen opens "
              "on DAILY SPECIAL OFFERS by default, so the LST HUNT tab arrives GREY (unselected). "
              "Capture it in its grey state — once tapped, the tab turns blue and the rewards load.",
    ),
    Step(
        name="tap_all_claims",
        template="05_tap_to_claim.png",
        tap_all=True,
        on_missing="skip",
        threshold_override=0.72,
        # The 'you got X' popup that appears after each claim closes on any
        # tap — so we don't template-match an OK button, we just tap the
        # screen center which lands on the popup's overlay and dismisses it.
        # On a 3120x1440 landscape AVD the center is (1560, 720).
        dismiss_after_tap=(1560, 720),
        dismiss_pause_seconds=1.5,
        notes="The purple 'Tap to Claim' badge. tap_all=True so we hit every ready reward; "
              "loose threshold (0.72 vs global 0.82) because the text is distinctive enough "
              "that false positives are unlikely, and we'd rather catch a borderline-rendered "
              "badge than miss a claim. After each tap, dismiss_after_tap clears the 'you got X' "
              "popup so the next claim's tap lands on the next card.",
    ),
    Step(
        name="back_to_lobby",
        template="07_back_arrow.png",
        on_missing="skip",
        notes="The < arrow top-left to back out of UNDERGROUND OFFERS. Daemon also sends KEYCODE_BACK "
              "afterwards as a fallback, so this is optional.",
    ),
]


@dataclass
class FlowResult:
    claims_attempted: int = 0
    steps_run: list[str] = field(default_factory=list)
    steps_skipped: list[str] = field(default_factory=list)
    aborted_at: str | None = None
    abort_reason: str | None = None

    def ok(self) -> bool:
        return self.aborted_at is None


def run_step(
    device: AdbDevice,
    step: Step,
    templates_dir: Path,
    threshold: float,
    step_timeout: float,
    settle_default: float,
    tap_settle: float,
    dry_run: bool,
) -> tuple[bool, int]:
    """Returns (ok, taps_made). ok=False means we couldn't satisfy this step."""
    tpl_path = templates_dir / step.template
    if not tpl_path.exists():
        log.warning("step %s: template missing at %s — treating as skip-eligible", step.name, tpl_path)
        return (step.on_missing == "skip" or step.optional), 0

    if step.pre_swipe:
        x1, y1, x2, y2, dur = step.pre_swipe
        log.info("step %s: pre-swipe (%d,%d)->(%d,%d)", step.name, x1, y1, x2, y2)
        device.swipe(x1, y1, x2, y2, dur)
        time.sleep(0.8)

    # Per-step timeout override — useful for fast skip-eligible checks like
    # tap_home_icon which doesn't need the full 20s budget.
    effective_timeout = step.timeout_override if step.timeout_override is not None else step_timeout
    # Per-step threshold override — for distinctive text-only templates
    # (e.g. tap_to_claim) we can afford a looser match.
    effective_threshold = step.threshold_override if step.threshold_override is not None else threshold
    deadline = time.time() + effective_timeout
    while time.time() < deadline:
        screen = device.screencap()
        if step.tap_all:
            # Loop scan-and-tap until N consecutive empty rounds. Handles:
            #   a) Threshold-bordered badges that match on a re-scan.
            #   b) Cooldowns expiring DURING the cycle (timer at 00:00:01
            #      flips to "Tap to Claim" while we're tapping others).
            # PATIENCE_ROUNDS controls how persistent we are about case (b):
            # after we tap something, we keep checking with delays even if
            # the immediate next round is empty — a timer might expire a
            # few seconds later. Without this, today's bug recurs: daemon
            # taps 1 of 3, round 2 empty 4s later -> done, but card B's
            # timer expired 10s later and user has to claim manually.
            MAX_ROUNDS = 8
            PATIENCE_ROUNDS = 3       # how many consecutive empty rounds before we give up
            PATIENCE_DELAY = 6.0      # seconds between empty rounds (long enough for a timer to tick over)
            total_tapped = 0
            empty_streak = 0
            for round_idx in range(1, MAX_ROUNDS + 1):
                if round_idx > 1:
                    screen = device.screencap()
                hits = find_all(screen, tpl_path, threshold=effective_threshold)
                if not hits:
                    if round_idx == 1 and total_tapped == 0:
                        # Nothing to claim from the start — drop out fast
                        break
                    empty_streak += 1
                    log.info("step %s: round %d found no matches (empty_streak=%d/%d)",
                             step.name, round_idx, empty_streak, PATIENCE_ROUNDS)
                    if empty_streak >= PATIENCE_ROUNDS:
                        log.info("step %s: %d consecutive empty rounds — done", step.name, PATIENCE_ROUNDS)
                        break
                    time.sleep(PATIENCE_DELAY)
                    continue
                empty_streak = 0  # reset patience counter when we find something
                log.info("step %s: round %d found %d match(es) (scores=%s)",
                         step.name, round_idx, len(hits), [round(h.score, 3) for h in hits])
                for h in hits:
                    if dry_run:
                        log.info("  [dry-run] would tap (%d,%d)", h.x, h.y)
                    else:
                        device.tap(h.x, h.y)
                    total_tapped += 1
                    if step.dismiss_after_tap and not dry_run:
                        time.sleep(step.dismiss_pause_seconds)
                        dx, dy = step.dismiss_after_tap
                        log.info("  dismiss-tap (%d,%d)", dx, dy)
                        device.tap(dx, dy)
                    time.sleep(tap_settle)
                time.sleep(step.settle_after or settle_default)
            if total_tapped > 0:
                return True, total_tapped
        else:
            hit = find_template(screen, tpl_path, threshold=effective_threshold)
            if hit:
                log.info("step %s: matched at (%d,%d) score=%.3f", step.name, hit.x, hit.y, hit.score)
                if not step.tap:
                    log.info("  (verify-only step, no tap)")
                elif dry_run:
                    log.info("  [dry-run] would tap (%d,%d)", hit.x, hit.y)
                else:
                    device.tap(hit.x, hit.y)
                time.sleep(step.settle_after or settle_default)
                return True, 1
        time.sleep(0.6)

    log.warning("step %s: template never appeared within %.1fs (last best score < threshold)",
                step.name, effective_timeout)
    return (step.on_missing in ("skip",) or step.optional), 0


def run_flow(
    device: AdbDevice,
    steps: list[Step],
    templates_dir: Path,
    *,
    threshold: float,
    step_timeout: float,
    screen_settle: float,
    tap_settle: float,
    dry_run: bool,
    pre_step_hook: Callable[[Step], None] | None = None,
) -> FlowResult:
    result = FlowResult()
    for step in steps:
        if pre_step_hook:
            pre_step_hook(step)
        ok, taps = run_step(
            device, step, templates_dir,
            threshold=threshold, step_timeout=step_timeout,
            settle_default=screen_settle, tap_settle=tap_settle,
            dry_run=dry_run,
        )
        if step.name == "tap_all_claims":
            result.claims_attempted = taps
        if ok:
            result.steps_run.append(step.name)
            continue
        if step.on_missing == "skip" or step.optional:
            result.steps_skipped.append(step.name)
            continue
        if step.on_missing == "retry_back":
            log.info("step %s failed; pressing BACK and retrying once", step.name)
            device.back()
            time.sleep(screen_settle)
            ok2, taps2 = run_step(
                device, step, templates_dir,
                threshold=threshold, step_timeout=step_timeout,
                settle_default=screen_settle, tap_settle=tap_settle,
                dry_run=dry_run,
            )
            if ok2:
                if step.name == "claim_buttons":
                    result.claims_attempted = taps2
                result.steps_run.append(step.name)
                continue
        result.aborted_at = step.name
        result.abort_reason = f"template '{step.template}' not found within {step_timeout}s"
        return result
    return result
