"""Idealised round-dot ("scoreboard") stimuli, as in Scialom et al.: constant-size patches placed
anywhere along a contour, no electrode layout, no Dynaphos.

Used by Module B (reference arm R0) and Module E (evaluator training data). It depends on neither
planner, so it can sit on both sides of the B / E separation.

Pixel convention is the same as module_c.atlas: (row, col) index the image array.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def place_dots(weight: np.ndarray, k: int, rng: Optional[np.random.Generator] = None,
               floor_frac: float = 0.1, max_shrink: int = 30) -> np.ndarray:
    """Choose up to k well-spread pixels on a contour weight map. Returns (n, 2) float (row, col).

    rng=None: deterministic greedy best-overlap order (strongest contour first), as in the paper.
    rng given: random order (used to build diverse evaluator training data).
    Pixels below floor_frac * max(weight) are never used. The exclusion radius starts at the
    arc-length spacing (n_candidates / k) and shrinks until k dots fit or it reaches 1 px.
    """
    w = np.asarray(weight, np.float32)
    if w.max() <= 0 or k <= 0:
        return np.zeros((0, 2), np.float32)
    cand = np.argwhere(w >= floor_frac * w.max())
    if rng is None:
        cand = cand[np.argsort(-w[cand[:, 0], cand[:, 1]], kind="stable")]
    else:
        cand = cand[rng.permutation(len(cand))]
    r = max(1.0, len(cand) / float(k))
    chosen = np.zeros((0, 2), np.float32)
    for _ in range(max_shrink):
        sel = []
        for p in cand:
            if sel and np.min(np.hypot(*(np.asarray(sel) - p).T)) < r:
                continue
            sel.append(p)
            if len(sel) == k:
                break
        chosen = np.asarray(sel, np.float32).reshape(-1, 2)
        if len(chosen) >= k or r <= 1.0:
            break
        r = max(1.0, r * 0.8)
    return chosen


def render_dots(coords: np.ndarray, shape: tuple, sigma_px: float, amplitude: float = 1.0) -> np.ndarray:
    """Sum of Gaussian blobs with peak `amplitude`, clipped to [0, 1]. Returns (H, W) float32."""
    H, W = shape
    out = np.zeros((H, W), np.float32)
    if len(coords) == 0:
        return out
    rr = np.arange(H, dtype=np.float32)[None, :, None]
    cc = np.arange(W, dtype=np.float32)[None, None, :]
    cr = coords[:, 0].astype(np.float32)[:, None, None]
    ccn = coords[:, 1].astype(np.float32)[:, None, None]
    g = np.exp(-((rr - cr) ** 2 + (cc - ccn) ** 2) / (2.0 * sigma_px ** 2))
    return np.clip(amplitude * g.sum(0), 0.0, 1.0)
