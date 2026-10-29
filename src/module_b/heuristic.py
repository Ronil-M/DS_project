"""B1: contour-driven heuristic planner (no training) and R1: random-K control.

B1 follows plan 6.2 / Scialom et al.:
  1. Gabor quadrature contour map of the input                      (module_b.contours)
  2. score every electrode: Gaussian-weighted contour overlap around its phosphene centre
  3. greedy best-first selection with an exclusion radius (visual-field degrees, scaled with local
     phosphene size) and a floor at `floor_frac` of the maximum score
  4. optional homogeneous thinning: build a larger pool, then repeatedly delete the lower-scoring member
     of the closest pair until K remain. With a fixed pool size the sets are NESTED across K.
  5. currents: brightness-targeted (default 70 %), or scaled by contour score

If fewer than K electrodes pass the floor, fewer are returned (the paper stops there too). Check
`last_n_selected` / `max_achievable_k` rather than assuming K were driven.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .calibrate import currents_for_brightness
from .constraints import CurrentParam
from .contours import contour_map
from .footprint import GaussianSampler
from .planner_api import as_image_batch


class HeuristicPlanner:
    name = "B1_heuristic"

    def __init__(self, atlas, sim=None, *, method: str = "thin", excl_deg: float = 0.65,
                 excl_sigma_mult: float = 2.0, pool: int = 96, pool_excl_deg: float = 0.25,
                 floor_frac: float = 0.10, amp_mode: str = "brightness", target_brightness: float = 0.7,
                 levels: Optional[int] = None, contour_kw: Optional[dict] = None):
        assert method in ("greedy", "thin") and amp_mode in ("brightness", "score")
        self.atlas, self.method, self.amp_mode = atlas, method, amp_mode
        self.excl_deg, self.excl_mult, self.pool, self.pool_excl = excl_deg, excl_sigma_mult, pool, pool_excl_deg
        self.floor_frac = floor_frac
        self.contour_kw = contour_kw or {}
        self.param = CurrentParam(atlas, levels=levels)
        self.sampler = GaussianSampler(atlas, tuple(atlas.meta["image_shape"]))
        self.xy = np.stack([atlas.x_deg, atlas.y_deg], 1)
        self.sigma_deg = np.nan_to_num(atlas.sigma_deg_ref, nan=0.0)
        self.usable = atlas.usable()
        if sim is not None:
            self.base_currents = currents_for_brightness(sim, atlas, target_brightness)
        else:   # fallback without a simulator: linear interpolation (documented approximation)
            thr = self.param.thr.numpy()
            self.base_currents = np.where(self.usable, thr + target_brightness * (self.param.i_max - thr), 0).astype(np.float32)
        self.last_n_selected = None

    # ---------------------------------------------------------------- scores
    def electrode_scores(self, x) -> np.ndarray:
        xb = as_image_batch(x)
        with torch.no_grad():
            s = self.sampler.sample(contour_map(xb, **self.contour_kw)).numpy()
        s[:, ~self.usable] = 0.0
        return s

    # ---------------------------------------------------------------- selection
    def _dist(self, idx: np.ndarray) -> np.ndarray:
        d = self.xy[idx][:, None, :] - self.xy[idx][None, :, :]
        return np.hypot(d[..., 0], d[..., 1])

    def _greedy(self, score: np.ndarray, limit: int, excl_min_deg: float, excl_mult: float) -> list:
        floor = self.floor_frac * score.max()
        order = np.argsort(-score, kind="stable")
        sel: list = []
        for e in order:
            if score[e] < floor or score[e] <= 0 or not self.usable[e]:
                break
            if sel:
                s = np.asarray(sel)
                d = np.hypot(*(self.xy[s] - self.xy[e]).T)
                r = np.maximum(excl_min_deg, excl_mult * np.maximum(self.sigma_deg[s], self.sigma_deg[e]))
                if np.any(d < r):
                    continue
            sel.append(int(e))
            if len(sel) >= limit:
                break
        return sel

    def select(self, score: np.ndarray, k: int) -> np.ndarray:
        k = self.param.clip_k(k)
        if score.max() <= 0 or k <= 0:
            return np.zeros(0, int)
        if self.method == "greedy":
            return np.asarray(self._greedy(score, k, self.excl_deg, self.excl_mult), int)
        pool = np.asarray(self._greedy(score, max(self.pool, k), self.pool_excl, 1.0), int)
        if len(pool) <= k:
            return pool
        keep = np.ones(len(pool), bool)
        D = self._dist(pool)
        np.fill_diagonal(D, np.inf)
        sc = score[pool]
        while keep.sum() > k:
            Dm = np.where(keep[:, None] & keep[None, :], D, np.inf)
            i, j = np.unravel_index(np.argmin(Dm), Dm.shape)
            keep[i if sc[i] < sc[j] else j] = False
        return pool[keep]

    # ---------------------------------------------------------------- public
    def plan(self, x, k: int, label=None) -> np.ndarray:
        scores = self.electrode_scores(x)
        cur = np.zeros_like(scores, dtype=np.float32)
        n_sel = []
        for b in range(scores.shape[0]):
            idx = self.select(scores[b], k)
            n_sel.append(len(idx))
            if self.amp_mode == "brightness":
                cur[b, idx] = self.base_currents[idx]
            else:
                frac = scores[b, idx] / max(scores[b, idx].max(), 1e-9) if len(idx) else 0
                thr = self.param.thr.numpy()[idx]
                cur[b, idx] = thr + (self.param.i_max - thr) * frac
        self.last_n_selected = n_sel
        out = self.param.quantise(torch.from_numpy(cur)).numpy()
        return out.astype(np.float32)

    def max_achievable_k(self, x) -> np.ndarray:
        """Number of electrodes above the score floor (upper bound on K for each input)."""
        s = self.electrode_scores(x)
        return (s >= self.floor_frac * s.max(1, keepdims=True)).sum(1)


class RandomPlanner:
    """R1: K random usable electrodes (inside `radius_deg` if given) at random above-threshold currents."""

    name = "R1_random"

    def __init__(self, atlas, radius_deg: Optional[float] = None, seed: int = 0, levels: Optional[int] = None):
        self.atlas, self.radius, self.seed = atlas, radius_deg, seed
        self.param = CurrentParam(atlas, levels=levels)
        self._calls = 0

    def plan(self, x, k: int, label=None) -> np.ndarray:
        B = as_image_batch(x).shape[0]
        rng = np.random.default_rng([self.seed, self._calls]); self._calls += 1
        cur = self.atlas.random_currents(k, B, rng, radius_deg=self.radius)
        return self.param.quantise(torch.from_numpy(cur)).numpy().astype(np.float32)
