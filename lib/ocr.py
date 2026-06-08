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


def _preprocess_for_ocr(screen_bgr: np.ndarray) -> np.ndarray:
    """Convert the screen to a black-on-white binary image suitable for
    tesseract. The badges are light text on a dark background; we invert
    so tesseract sees its preferred dark-text-on-light."""
    gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)
    # Upscale moderately — tesseract is more accurate on larger glyphs
    # but slows down linearly; 1.5x is a reasonable sweet spot.
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    # Otsu thresholding; then invert if the result has more dark than light
    # pixels (i.e. light-text-on-dark → flip to dark-text-on-light).
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if binary.mean() < 127:
        binary = cv2.bitwise_not(binary)
    return binary


def read_cooldowns(screen_bgr: np.ndarray) -> list[Cooldown]:
    """OCR the whole screen, regex-extract every `Remaining HH:MM:SS`
    occurrence, return them as Cooldown objects (in seconds). Order matches
    tesseract's reading order (top-to-bottom, left-to-right), which is fine
    for our use — we only consume `min(cooldowns_seconds)`."""
    if not tesseract_available():
        log.warning("tesseract binary not found on PATH — skipping cooldown OCR")
        return []
    binary = _preprocess_for_ocr(screen_bgr)
    # PSM 11 = "sparse text" — finds text wherever it is on the image without
    # assuming a single block layout. Fits the LST Hunt screen where badges
    # are scattered around the wolf models.
    raw = pytesseract.image_to_string(
        binary,
        config="--psm 11 -c tessedit_char_whitelist=0123456789:RemainigRemainingabcdefghijklmnopqrstuvwxyz ",
    )
    out: list[Cooldown] = []
    for m in TIME_RE.finditer(raw):
        hh, mm, ss = m.group(1), m.group(2), m.group(3)
        secs = int(hh) * 3600 + int(mm) * 60 + int(ss)
        out.append(Cooldown(raw_text=m.group(0), seconds=secs))
        log.info("read cooldown: %r -> %ds (~%.1fh)", m.group(0), secs, secs / 3600)
    log.info("OCR found %d cooldown timer(s) in the screen text", len(out))
    return out


def min_cooldown_seconds(screen_bgr: np.ndarray) -> int | None:
    """Convenience: the soonest cooldown on screen, or None if none readable.
    A return of None means "use whatever the daemon's default period is."""
    cds = read_cooldowns(screen_bgr)
    if not cds:
        return None
    return min(c.seconds for c in cds)
