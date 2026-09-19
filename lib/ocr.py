"""OCR every 'Remaining HH:MM:SS' timer visible on the LST Hunt page.
Returns the soonest expiration so the daemon can sleep until just after
the next reward becomes claimable.

Color-independent on purpose: rarity tiers introduce new icon colors
(purple, blue, orange, red, ...) and we don't want to chase that every
time Activision adds one. We run tesseract over the entire rewards
region and regex-match 'Remaining HH:MM:SS' from the raw text — same
philosophy as the text-only Tap to Claim template.
"""
from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract

log = logging.getLogger(__name__)

# HH:MM:SS — anywhere in the OCR'd text. We tried anchoring on the
# "Remaining" prefix but tesseract sometimes eats the first letters of
# "Remaining" when the diamond icon overlaps them (the top-right badge
# is the worst offender). The middle separator is `\D?` because tesseract
# also frequently eats the second colon ("03:42:55" -> "03:4255").
# Sanity bounds in the parser (mm/ss < 60, hh <= 48) prevent false
# matches on unrelated text like currency or version numbers.
TIME_RE = re.compile(r"(\d{1,2}):(\d{2})\D?(\d{2})")


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


@dataclass
class Cooldown:
    raw_text: str
    seconds: int


def _crop_roi(screen_bgr: np.ndarray) -> np.ndarray:
    """Crop the screen to just the wolf/badge area.
    Whole-screen OCR sometimes drops the top badge because tesseract's page
    segmentation treats the very top as header noise. Cropping to the
    interior region (excluding the header bar at top, the tab sidebar at
    left, and the bottom UI) refocuses tesseract on just the badges.

    y1 starts at 1.5% of height: the top-right wolf's badge sits very close
    to the top of the screen (~y=50-80 on a 1440-tall frame), and clipping
    even 30px off would partially eat the badge text — exactly why the
    long-cooldown timer kept getting missed earlier.

    Coordinates are fractions so this survives screen-resolution changes."""
    H, W = screen_bgr.shape[:2]
    y1, y2 = int(H * 0.015), int(H * 0.58)
    x1, x2 = int(W * 0.19), W
    return screen_bgr[y1:y2, x1:x2]


def _preprocess_variant(screen_bgr: np.ndarray, upscale: float, method: str) -> np.ndarray:
    """Multiple preprocessing variants — tesseract can miss timers on one
    binarization but catch them on another, especially when the badge sits
    over a brightly-lit wolf body where global Otsu fails locally."""
    gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    if method == "clahe_otsu":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 8
        )
    else:  # raw_otsu
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if binary.mean() < 127:
        binary = cv2.bitwise_not(binary)
    return binary


def _ocr_text(image: np.ndarray, psm: int) -> str:
    return pytesseract.image_to_string(
        image,
        config=f"--psm {psm} -c tessedit_char_whitelist=0123456789:Remaining ",
    )


def _ocr_timers_located(image: np.ndarray, psm: int) -> list[tuple[int, int, int]]:
    """Every 'Remaining HH:MM:SS' on the image as (seconds, centre_x, centre_y).

    Positions matter because two different cards can legitimately be seconds
    apart -- a real screen had badges at 00:36:26 and 00:36:06, 20s apart --
    and no value-based rule can tell that from the same badge being re-read
    slightly differently across passes. Where the text sits on screen can:
    one badge is one place.
    """
    data = pytesseract.image_to_data(
        image,
        config=f"--psm {psm} -c tessedit_char_whitelist=0123456789:Remaining ",
        output_type=pytesseract.Output.DICT,
    )
    # Match each time as a SINGLE WORD and use that word's own box. Grouping
    # into lines does not work here: PSM 6 and 3 treat a whole row of the page
    # as one line, merging separate badges (and the currency in the header)
    # together, so the box spans the width and the x centre is meaningless --
    # the same badge came back at x=4984 on one pass and x=6316 on another.
    # A single word's box is tight and lands in the same place every pass.
    out: list[tuple[int, int, int]] = []
    for i, word in enumerate(data["text"]):
        w = word.strip()
        if not w:
            continue
        m = TIME_RE.fullmatch(w) or TIME_RE.search(w)
        if not m:
            continue
        hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if mm >= 60 or ss >= 60 or hh > 48:
            continue
        cx = data["left"][i] + data["width"][i] // 2
        cy = data["top"][i] + data["height"][i] // 2
        out.append((hh * 3600 + mm * 60 + ss, cx, cy))
    return out


# LST Hunt always has 3 cards. If OCR finds fewer than this, we log a
# warning — the daemon still works (min() drives the schedule) but it's
# a signal the preprocessing or PSM config might need tuning.
EXPECTED_TIMER_COUNT = 3


def read_cooldowns(screen_bgr: np.ndarray) -> list[Cooldown]:
    """OCR a single screen for `Remaining HH:MM:SS` timers.
    For the retry-when-missing-timers behavior, see read_cooldowns_with_retry().
    """
    if not tesseract_available():
        log.warning("tesseract binary not found on PATH — skipping cooldown OCR")
        return []
    # Proportional dedup: 25s absolute floor OR 5% of the value, whichever is
    # larger. Short timers (18min vs 20min, 132s gap) stay distinct because
    # 5% of 1080s = 54s. Long-timer OCR variants (03:42:55 misread as both
    # 13375s and 13675s, 300s gap) collapse because 5% of 13000s = 650s > 300s.
    def _is_dup(a: int, b: int) -> bool:
        return abs(a - b) <= max(25, int(0.05 * min(a, b)))
    out: list[Cooldown] = []
    seen: list[tuple[int, int]] = []   # centre of each badge already recorded
    # Crop to the badge ROI first — tesseract is far more reliable when
    # not distracted by the header / sidebar / footer regions.
    roi = _crop_roi(screen_bgr)
    # 3 preprocessing variants × 3 PSM modes = up to 9 passes on the ROI.
    # Tesseract is fast enough at this size and we run OCR once per cycle.
    for prep_method in ("clahe_otsu", "adaptive", "raw_otsu"):
        binary = _preprocess_variant(roi, upscale=4.0, method=prep_method)
        for psm in (11, 6, 3):
            # Only image_to_data is needed. The old image_to_string call still
            # ran here even though the regex path that used it was replaced by
            # position-based matching, so every pass paid for tesseract twice.
            try:
                located = _ocr_timers_located(binary, psm)
            except Exception as e:
                log.debug("image_to_data %s/PSM %d failed: %s", prep_method, psm, e)
                located = []
            # Same badge, seen again on another pass -> same place on screen.
            # Tolerance is generous (badges are far apart) but far tighter than
            # any value rule could be: it was the value rule, collapsing
            # anything within 5%, that silently merged two real cards 20s apart
            # and left the daemon on a low-confidence 2/3 read.
            h_roi, w_roi = binary.shape[:2]
            near_x, near_y = int(w_roi * 0.06), int(h_roi * 0.06)
            for secs, cx, cy in located:
                if any(abs(cx - sx) <= near_x and abs(cy - sy) <= near_y
                       for sx, sy in seen):
                    continue
                seen.append((cx, cy))
                out.append(Cooldown(raw_text=f"{secs//3600:02d}:{secs%3600//60:02d}:{secs%60:02d}",
                                    seconds=secs))
                log.info("read cooldown (%s/PSM %d) at (%d,%d): %ds (~%.1fh)",
                         prep_method, psm, cx, cy, secs, secs / 3600)
            # Nine passes exist to catch a badge one binarisation misses; once
            # every card is accounted for there is nothing left to catch, and
            # the remaining passes are pure latency on the critical path.
            if len(out) >= EXPECTED_TIMER_COUNT:
                break
        if len(out) >= EXPECTED_TIMER_COUNT:
            break
    if len(out) < EXPECTED_TIMER_COUNT:
        log.warning(
            "OCR found only %d cooldown timer(s), expected %d — daemon will still use "
            "min() for scheduling but one timer was missed",
            len(out), EXPECTED_TIMER_COUNT,
        )
    else:
        log.info("OCR found %d unique cooldown timer(s)", len(out))
    return out


def read_cooldowns_with_retry(
    device,
    max_attempts: int = 3,
    inter_attempt_seconds: float = 4.0,
    expected: int = EXPECTED_TIMER_COUNT,
) -> list[Cooldown]:
    """Take a fresh screenshot, OCR, and if we found fewer than `expected`
    timers, wait a few seconds and retry — the post-claim animation
    sometimes hides a badge for a moment. After max_attempts we accept
    whatever we have.

    Returns the largest set of unique cooldowns seen across all attempts
    (not just the last attempt), so a transient miss on attempt 2 doesn't
    discard a successful read from attempt 1.
    """
    import time
    best: list[Cooldown] = []
    last_screen = None
    for attempt in range(1, max_attempts + 1):
        screen = device.screencap()
        last_screen = screen
        cds = read_cooldowns(screen)
        # Merge with best-so-far (across attempts), preserving ±5s dedup
        # Same proportional dedup as read_cooldowns
        def _is_dup(a: int, b: int) -> bool:
            return abs(a - b) <= max(25, int(0.05 * min(a, b)))
        for cd in cds:
            if not any(_is_dup(b.seconds, cd.seconds) for b in best):
                best.append(cd)
        log.info(
            "OCR attempt %d/%d: %d timer(s) this pass, %d total unique",
            attempt, max_attempts, len(cds), len(best),
        )
        if len(best) >= expected:
            break  # got everything we expected
        if attempt < max_attempts:
            time.sleep(inter_attempt_seconds)
    if len(best) < expected:
        log.warning(
            "OCR across %d attempt(s) found only %d/%d timers — proceeding anyway "
            "(daemon's max-sleep ceiling will catch the missed one)",
            max_attempts, len(best), expected,
        )
        _dump_failed_ocr(last_screen)
    return best


def _dump_failed_ocr(screen_bgr) -> None:
    """Save the screen we failed to read timers off, so the miss is
    diagnosable afterwards. Without this the LST Hunt page is never
    captured anywhere and a bad read can only be reproduced by running a
    whole live cycle. Best-effort; never raises into the claim flow."""
    if screen_bgr is None:
        return
    try:
        import time as _t
        from pathlib import Path as _P
        d = _P(__file__).resolve().parent.parent / "logs"
        d.mkdir(exist_ok=True)
        f = d / f"{_t.strftime('%Y-%m-%dT%H-%M-%SZ', _t.gmtime())}_cooldown_ocr_miss.png"
        cv2.imwrite(str(f), screen_bgr)
        log.info("Saved the unreadable cooldown screen to %s", f)
    except Exception as e:
        log.debug("could not dump failed-OCR screen: %s", e)


def min_cooldown_seconds(screen_bgr: np.ndarray) -> int | None:
    """Convenience: the soonest cooldown on screen, or None if none readable.
    A return of None means "use whatever the daemon's default period is."""
    cds = read_cooldowns(screen_bgr)
    if not cds:
        return None
    return min(c.seconds for c in cds)
