"""OCR the 'Remaining HH:MM:SS' timer that appears on each LST Hunt card that
is still on cooldown. Returns the soonest expiration so the daemon can sleep
until just after the next reward becomes claimable.

Tesseract is used in a tight, character-restricted PSM 7 (single text line)
mode for speed and accuracy on UI text. We preprocess the screen by binarizing
on the purple/magenta hue of the "Remaining …" badge — when the badge isn't
present, OCR runs on a near-empty image and returns nothing, which is correct.
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

# Matches HH:MM:SS or H:MM:SS or MM:SS (rare but worth catching). All three
# captures are stringified ints we then convert to seconds.
TIME_RE = re.compile(r"(?:(\d{1,2}):)?(\d{1,2}):(\d{2})")


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


@dataclass
class Cooldown:
    raw_text: str
    seconds: int


def _purple_icon_mask(screen_bgr: np.ndarray) -> np.ndarray:
    """Mask for the small purple diamond icon that sits to the LEFT of the
    timer text in each 'Remaining HH:MM:SS' badge. The actual badge text is
    white-on-dark and not directly maskable by color — anchoring off the icon
    is far more robust."""
    hsv = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2HSV)
    # Hue ~115-150 (purple/violet), high saturation. Tuned from the recorded
    # frames; AVD render may need widening — see lib/ocr.py:_OCR_DEBUG.
    lower = np.array([115, 70, 60], dtype=np.uint8)
    upper = np.array([155, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask


def _icon_blobs(mask: np.ndarray, screen_shape: tuple[int, int, int]) -> list[tuple[int, int, int, int]]:
    """Connected components on the icon mask. Keep blobs that look like the
    purple diamond — roughly square, plausibly badge-sized for the screen."""
    H, W = screen_shape[:2]
    # Expected icon size ~3-5% of screen height
    min_dim = max(12, H // 60)
    max_dim = max(80, H // 12)
    num, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if w < min_dim or h < min_dim or w > max_dim or h > max_dim:
            continue
        aspect = w / max(1, h)
        if aspect < 0.6 or aspect > 1.8:  # diamond is roughly square
            continue
        if area < (min_dim * min_dim) // 2:
            continue
        blobs.append((x, y, w, h))
    return blobs


def read_cooldowns(screen_bgr: np.ndarray) -> list[Cooldown]:
    """Find every 'Remaining HH:MM:SS' badge on the screen by locating the
    purple diamond icon and OCR'ing the region immediately to its right.
    Returns a list of Cooldown(seconds=…) — empty if none readable."""
    if not tesseract_available():
        log.warning("tesseract binary not found on PATH — skipping cooldown OCR")
        return []
    H, W = screen_bgr.shape[:2]
    mask = _purple_icon_mask(screen_bgr)
    blobs = _icon_blobs(mask, screen_bgr.shape)
    log.info("icon detection: %d candidate badge(s)", len(blobs))
    out: list[Cooldown] = []
    for (x, y, w, h) in blobs:
        # OCR region: extend rightward by ~8x icon width (timer text is the
        # widest part of the badge), pad vertical generously.
        ocr_x1 = x + w  # right edge of icon
        ocr_y1 = max(0, y - h // 2)
        ocr_x2 = min(W, ocr_x1 + w * 9)
        ocr_y2 = min(H, y + h + h // 2)
        crop = screen_bgr[ocr_y1:ocr_y2, ocr_x1:ocr_x2]
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        # Invert so dark-bg light-text becomes black-on-white (tesseract prefers it).
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if binary.mean() < 127:
            binary = cv2.bitwise_not(binary)
        text = pytesseract.image_to_string(
            binary,
            config="--psm 7 -c tessedit_char_whitelist=0123456789:Remainig ",
        ).strip()
        m = TIME_RE.search(text)
        if not m:
            log.debug("no timer near icon (%d,%d): raw=%r", x, y, text)
            continue
        hh, mm, ss = m.group(1), m.group(2), m.group(3)
        secs = (int(hh) if hh else 0) * 3600 + int(mm) * 60 + int(ss)
        out.append(Cooldown(raw_text=text, seconds=secs))
        log.info("read cooldown @ icon(%d,%d): %r -> %ds (~%.1fh)", x, y, text, secs, secs / 3600)
    return out


def min_cooldown_seconds(screen_bgr: np.ndarray) -> int | None:
    """Convenience: the soonest cooldown on screen, or None if none readable.
    A return of None means "use whatever the daemon's default period is."""
    cds = read_cooldowns(screen_bgr)
    if not cds:
        return None
    return min(c.seconds for c in cds)
