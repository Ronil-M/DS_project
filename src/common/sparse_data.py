"""Sparse-dot image generator: the 'third source' of plan 7.1, shared by T_c (Module B) and E (Module E).

Mixes (i) scoreboard dots on the TRUE contour (random K, jitter, dropout, random size) with
(ii) Dynaphos renders of random contour-adjacent electrode subsets. No planner output is ever used, so
a classifier trained on this has never seen B1 / B2 images. Use different seeds (and architectures) for
T_c and E.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from .dots import place_dots, render_dots
from .imops import dilate


def make_sparse_dataset(shapes: dict, rng: np.random.Generator, *, sim=None, atlas=None,
                        frac_dynaphos: float = 0.3, k_range: Tuple[int, int] = (3, 64),
                        jitter_px: float = 1.0, p_drop_max: float = 0.3,
                        dot_sigma_px: Tuple[float, float] = (0.8, 2.5), tol_px: float = 3.0, chunk: int = 32):
    """Returns (images (N,1,H,W) float32, labels (N,) int64) aligned with shapes['y']."""
    con, y = shapes["contour"], shapes["y"]
    N, _, H, W = con.shape
    imgs = np.zeros((N, 1, H, W), np.float32)
    use_dyn = np.zeros(N, bool)
    if sim is not None and atlas is not None:
        use_dyn = rng.random(N) < frac_dynaphos
    lo, hi = k_range
    for i in np.flatnonzero(~use_dyn):
        k = int(np.exp(rng.uniform(np.log(lo), np.log(hi + 1))))
        pts = place_dots(con[i, 0], min(k, hi), rng=rng)
        if len(pts):
            pts = pts + rng.normal(0, jitter_px, pts.shape)
            pts = pts[rng.random(len(pts)) >= rng.uniform(0, p_drop_max)]
        imgs[i, 0] = render_dots(pts, (H, W), rng.uniform(*dot_sigma_px), rng.uniform(0.7, 1.0))
    idx = np.flatnonzero(use_dyn)
    usable = atlas.usable() if len(idx) else None
    for s0 in range(0, len(idx), chunk):                       # chunked: never holds all Dynaphos renders / dilations at once
        ci = idx[s0:s0 + chunk]
        near = dilate(torch.from_numpy(con[ci]), tol_px).numpy()[:, 0]
        cur = np.zeros((len(ci), atlas.n), np.float32)
        for j, i in enumerate(ci):
            pool = np.flatnonzero(usable & (atlas.sample_nearest(near[j]) > 0.05))
            if len(pool) == 0:
                continue
            k = min(int(np.exp(rng.uniform(np.log(lo), np.log(hi + 1)))), len(pool))
            e = rng.choice(pool, k, replace=False)
            cur[j, e] = rng.uniform(np.minimum(atlas.thr_amp[e] * 1.05, sim.i_max), sim.i_max)
        with torch.no_grad():
            imgs[ci] = sim.render(cur).cpu().numpy()
        del near, cur
    return imgs, y.copy()
