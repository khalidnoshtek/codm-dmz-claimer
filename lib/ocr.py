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

# 'Remaining' label + HH:MM:SS or H:MM:SS (anchored on the label so we don't
# mis-parse other HH:MM:SS-looking text elsewhere on screen, e.g. server
# clocks or banner countdowns).
TIME_RE = re.compile(r"Remain[a-z]*\s*(\d{1,2}):(\d{2}):(\d{2})", re.IGNORECASE)


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
    Coordinates are fractions so this survives screen-resolution changes."""
    H, W = screen_bgr.shape[:2]
    y1, y2 = int(H * 0.04), int(H * 0.56)
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


# LST Hunt always has 3 cards. If OCR finds fewer than this, we log a
# warning — the daemon still works (min() drives the schedule) but it's
# a signal the preprocessing or PSM config might need tuning.
EXPECTED_TIMER_COUNT = 3


def read_cooldowns(screen_bgr: np.ndarray) -> list[Cooldown]:
    """OCR the screen across multiple preprocessing variants × PSM modes,
    regex-extract every `Remaining HH:MM:SS`, dedupe with a ±5s tolerance.

    Why so many passes: a single binarization + PSM rarely catches all 3
    LST Hunt timers — the badges sit over wolves whose bodies vary in
    brightness, so a global Otsu threshold that works for the top-right
    badge might wash out the bottom-right badge. Running multiple
    preprocessings (Otsu, adaptive, CLAHE+Otsu) × multiple page-segmentation
    modes maximizes the chance of catching every badge.
    """
    if not tesseract_available():
        log.warning("tesseract binary not found on PATH — skipping cooldown OCR")
        return []
    DEDUP_WINDOW = 5
    out: list[Cooldown] = []
    # Crop to the badge ROI first — tesseract is far more reliable when
    # not distracted by the header / sidebar / footer regions.
    roi = _crop_roi(screen_bgr)
    # 3 preprocessing variants × 3 PSM modes = up to 9 passes on the ROI.
    # Tesseract is fast enough at this size and we run OCR once per cycle.
    for prep_method in ("clahe_otsu", "adaptive", "raw_otsu"):
        binary = _preprocess_variant(roi, upscale=4.0, method=prep_method)
        for psm in (11, 6, 3):
            try:
                raw = _ocr_text(binary, psm)
            except Exception as e:
                log.debug("tesseract %s/PSM %d failed: %s", prep_method, psm, e)
                continue
            for m in TIME_RE.finditer(raw):
                hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if mm >= 60 or ss >= 60 or hh > 48:
                    continue
                secs = hh * 3600 + mm * 60 + ss
                if any(abs(c.seconds - secs) <= DEDUP_WINDOW for c in out):
                    continue
                cd = Cooldown(raw_text=m.group(0), seconds=secs)
                out.append(cd)
                log.info("read cooldown (%s/PSM %d): %r -> %ds (~%.1fh)",
                         prep_method, psm, m.group(0), secs, secs / 3600)
    if len(out) < EXPECTED_TIMER_COUNT:
        log.warning(
            "OCR found only %d cooldown timer(s), expected %d — daemon will still use "
            "min() for scheduling but one timer was missed",
            len(out), EXPECTED_TIMER_COUNT,
        )
    else:
        log.info("OCR found %d unique cooldown timer(s)", len(out))
    return out


def min_cooldown_seconds(screen_bgr: np.ndarray) -> int | None:
    """Convenience: the soonest cooldown on screen, or None if none readable.
    A return of None means "use whatever the daemon's default period is."""
    cds = read_cooldowns(screen_bgr)
    if not cds:
        return None
    return min(c.seconds for c in cds)
