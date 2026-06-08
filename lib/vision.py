"""OpenCV template matching with multi-scale support.

We use `TM_CCOEFF_NORMED` because it's robust to brightness shifts (the game
darkens overlays during transitions). For each lookup we also try a few
nearby scales — the AVD resolution can drift slightly from the template
capture if the emulator was resized.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Match:
    score: float
    x: int       # center x
    y: int       # center y
    w: int
    h: int

    def __bool__(self) -> bool:
        return True


def _load_template(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Template not found or unreadable: {path}")
    return img


def find_template(
    screen: np.ndarray,
    template_path: Path,
    threshold: float = 0.82,
    scales: tuple[float, ...] = (1.0, 0.95, 1.05, 0.9, 1.1),
) -> Match | None:
    """Return the best match for `template_path` in `screen`, or None if no
    scale beats `threshold`."""
    tpl_full = _load_template(template_path)
    best: Match | None = None
    for s in scales:
        if s == 1.0:
            tpl = tpl_full
        else:
            new_w = max(8, int(tpl_full.shape[1] * s))
            new_h = max(8, int(tpl_full.shape[0] * s))
            tpl = cv2.resize(tpl_full, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if tpl.shape[0] > screen.shape[0] or tpl.shape[1] > screen.shape[1]:
            continue
        res = cv2.matchTemplate(screen, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val >= threshold and (best is None or max_val > best.score):
            cx = max_loc[0] + tpl.shape[1] // 2
            cy = max_loc[1] + tpl.shape[0] // 2
            best = Match(score=max_val, x=cx, y=cy, w=tpl.shape[1], h=tpl.shape[0])
    return best


def find_all(
    screen: np.ndarray,
    template_path: Path,
    threshold: float = 0.82,
    min_distance: int = 30,
) -> list[Match]:
    """Find every distinct occurrence of the template (e.g. multiple CLAIM
    buttons stacked vertically). Uses non-maximum suppression via min_distance."""
    tpl = _load_template(template_path)
    if tpl.shape[0] > screen.shape[0] or tpl.shape[1] > screen.shape[1]:
        return []
    res = cv2.matchTemplate(screen, tpl, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(res >= threshold)
    candidates = sorted(
        ({"score": float(res[y, x]), "x": int(x), "y": int(y)} for y, x in zip(ys, xs)),
        key=lambda c: -c["score"],
    )
    kept: list[Match] = []
    for c in candidates:
        cx = c["x"] + tpl.shape[1] // 2
        cy = c["y"] + tpl.shape[0] // 2
        too_close = any(abs(cx - m.x) < min_distance and abs(cy - m.y) < min_distance for m in kept)
        if too_close:
            continue
        kept.append(Match(score=c["score"], x=cx, y=cy, w=tpl.shape[1], h=tpl.shape[0]))
    return kept
