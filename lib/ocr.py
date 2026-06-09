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


def _preprocess_for_ocr(screen_bgr: np.ndarray, upscale: float = 3.0) -> np.ndarray:
    """Convert the screen to a black-on-white binary image suitable for
    tesseract. The badges are light text on a dark background; we invert
    so tesseract sees its preferred dark-text-on-light. Larger upscale
    helps tesseract distinguish 6 vs 8 vs 0 — the most common misreads
    on the LST Hunt badges where wolf body lighting reduces digit
    contrast.
    """
    gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    # Boost local contrast before thresholding — CLAHE is far better than
    # global Otsu when text sits over varying-brightness backgrounds (like
    # the wolves whose bodies have bright highlights through them).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if binary.mean() < 127:
        binary = cv2.bitwise_not(binary)
    return binary


def _ocr_text(image: np.ndarray, psm: int) -> str:
    return pytesseract.image_to_string(
        image,
        config=f"--psm {psm} -c tessedit_char_whitelist=0123456789:Remaining ",
    )


def read_cooldowns(screen_bgr: np.ndarray) -> list[Cooldown]:
    """OCR the whole screen at a high upscale, run tesseract under MULTIPLE
    PSM modes (sparse text + single block), regex-extract every
    `Remaining HH:MM:SS`, dedupe by HH:MM:SS value (single best read of
    each timer survives). Returns a list of Cooldown in seconds.

    Why multiple PSMs: PSM 11 (sparse text) catches scattered badges but
    occasionally drops one. PSM 6 (single uniform block) catches it but
    sometimes mangles the layout. Running both and merging gets the best
    of both — same timer appearing in both passes is fine (we dedupe).
    """
    if not tesseract_available():
        log.warning("tesseract binary not found on PATH — skipping cooldown OCR")
        return []
    binary = _preprocess_for_ocr(screen_bgr, upscale=3.0)
    # Dedup with a ±5s tolerance — running multiple PSM passes occasionally
    # re-reads the same card at a moment where the seconds digit ticked, so
    # exact-second dedup overcounts. The clock granularity in the game is 1s
    # and OCR overhead between passes is ~0.5-1s, so 5s is a safe window.
    DEDUP_WINDOW = 5
    out: list[Cooldown] = []
    for psm in (11, 6, 7):
        try:
            raw = _ocr_text(binary, psm)
        except Exception as e:
            log.warning("tesseract PSM %d failed: %s", psm, e)
            continue
        for m in TIME_RE.finditer(raw):
            hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
            # Sanity: minutes/seconds out of range -> probably an OCR error
            if mm >= 60 or ss >= 60 or hh > 48:
                log.debug("skipping nonsensical timer from PSM %d: %r", psm, m.group(0))
                continue
            secs = hh * 3600 + mm * 60 + ss
            if any(abs(c.seconds - secs) <= DEDUP_WINDOW for c in out):
                continue
            cd = Cooldown(raw_text=m.group(0), seconds=secs)
            out.append(cd)
            log.info("read cooldown (PSM %d): %r -> %ds (~%.1fh)", psm, m.group(0), secs, secs / 3600)
    log.info("OCR found %d unique cooldown timer(s)", len(out))
    return out


def min_cooldown_seconds(screen_bgr: np.ndarray) -> int | None:
    """Convenience: the soonest cooldown on screen, or None if none readable.
    A return of None means "use whatever the daemon's default period is."""
    cds = read_cooldowns(screen_bgr)
    if not cds:
        return None
    return min(c.seconds for c in cds)
