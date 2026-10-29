"""Phosphene atlas (Module C, step C3): per-electrode geometry and calibrated thresholds.

Why an atlas? Dynaphos thresholds are expressed in *tissue activation* units, not amperes, and
depend on the stimulus duration (n_frames) and on the electrode's random draw. Module B needs
to know, per electrode: where its phosphene is, how big it is, and the smallest current that
actually produces a visible phosphene. The atlas measures this by running the frozen simulator.

Pixel convention: (row, col) index the rendered image array. Row 0 corresponds to the most
negative y in the visual field (y increases with row index). Use these, not degrees, when
sampling a contour map in Module B.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .simulator import ProstheticSimulator

_ARRAYS = ("x_deg", "y_deg", "ecc_deg", "row", "col", "sigma_deg_ref", "sigma_px_ref",
           "magnification", "thr_act", "thr_amp", "bright_ref")


@dataclass
class Atlas:
    x_deg: np.ndarray            # phosphene centre, degrees
    y_deg: np.ndarray
    ecc_deg: np.ndarray          # eccentricity, degrees
    row: np.ndarray              # sub-pixel image coordinates of the centre
    col: np.ndarray
    sigma_deg_ref: np.ndarray    # Gaussian sigma at the reference current, degrees
    sigma_px_ref: np.ndarray     # same in pixels (<~0.5 px: phosphene barely visible at this resolution)
    magnification: np.ndarray    # cortical magnification, mm / deg
    thr_act: np.ndarray          # Dynaphos activation threshold (internal units)
    thr_amp: np.ndarray          # smallest amplitude [A] giving a visible phosphene; inf if unreachable
    bright_ref: np.ndarray       # brightness in [0, 1] at the reference current
    meta: Dict = field(default_factory=dict)

    # --------------------------------------------------------------- basic info
    @property
    def n(self) -> int:
        return len(self.x_deg)

    def usable(self, i_max: Optional[float] = None) -> np.ndarray:
        """Boolean mask of electrodes that can produce a visible phosphene within i_max."""
        i_max = self.meta["i_max"] if i_max is None else i_max
        return np.isfinite(self.thr_amp) & (self.thr_amp <= i_max)

    def usable_within(self, radius_deg: float, i_max: Optional[float] = None) -> np.ndarray:
        return self.usable(i_max) & (self.ecc_deg <= radius_deg)

    def summary(self, i_max: Optional[float] = None) -> Dict:
        """Numbers to record in the experiment log (acceptance criterion C3)."""
        u = self.usable(i_max)
        fin = self.thr_amp[np.isfinite(self.thr_amp)]
        return {
            "n_electrodes_in_view": int(self.n),
            "n_usable_at_i_max": int(u.sum()),
            "frac_usable": float(u.mean()),
            "n_in_central_5deg_radius": int((self.ecc_deg <= 5).sum()),   # objects span ~10 deg
            "n_usable_in_central_5deg_radius": int(self.usable_within(5, i_max).sum()),
            "thr_amp_uA_median": float(np.median(fin) * 1e6) if len(fin) else float("nan"),
            "thr_amp_uA_p10_p90": [float(np.percentile(fin, 10) * 1e6), float(np.percentile(fin, 90) * 1e6)] if len(fin) else None,
            "frac_sigma_below_0.5px": float((self.sigma_px_ref < 0.5).mean()),
            "max_achievable_K": int(u.sum()),
        }

    # --------------------------------------------------------------- sampling helpers for Module B
    def sample_nearest(self, img2d: np.ndarray) -> np.ndarray:
        """Value of a 2-D map (H, W) at each electrode's nearest pixel."""
        H, W = img2d.shape
        r = np.clip(np.rint(self.row).astype(int), 0, H - 1)
        c = np.clip(np.rint(self.col).astype(int), 0, W - 1)
        return img2d[r, c]

    def sample_smoothed(self, img2d: np.ndarray, sigma_px: float) -> np.ndarray:
        """Like sample_nearest but after Gaussian smoothing (tolerates 1-2 px misalignment)."""
        from scipy.ndimage import gaussian_filter
        return self.sample_nearest(gaussian_filter(img2d.astype(np.float32), sigma_px))

    def random_currents(self, k: int, batch: int, rng: np.random.Generator, i_max: Optional[float] = None,
                        radius_deg: Optional[float] = None, margin: float = 1.05) -> np.ndarray:
        """(batch, N) currents: k random USABLE electrodes, amplitude uniform in [margin*thr, i_max]."""
        i_max = self.meta["i_max"] if i_max is None else i_max
        pool = np.flatnonzero(self.usable(i_max) if radius_deg is None else self.usable_within(radius_deg, i_max))
        if len(pool) == 0:
            raise ValueError("No usable electrodes; raise i_max or change the layout.")
        k = min(k, len(pool))
        out = np.zeros((batch, self.n), np.float32)
        for b in range(batch):
            idx = rng.choice(pool, size=k, replace=False)
            lo = np.minimum(self.thr_amp[idx] * margin, i_max)
            out[b, idx] = rng.uniform(lo, i_max)
        return out

    # --------------------------------------------------------------- io
    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, **{k: getattr(self, k) for k in _ARRAYS}, meta=json.dumps(self.meta))

    @classmethod
    def load(cls, path: str | Path) -> "Atlas":
        z = np.load(path, allow_pickle=False)
        return cls(**{k: z[k] for k in _ARRAYS}, meta=json.loads(str(z["meta"])))


def _find_thresholds(sim: ProstheticSimulator, n_frames: int, amp_max: float, iters: int) -> np.ndarray:
    """Smallest amplitude per electrode that is visible after `n_frames` static frames.

    Uses the batch dimension as a (B+1)-ary search: each round evaluates B candidate amplitudes
    per electrode at once. Assumes visibility is monotone in amplitude (checked in the tests).
    """
    B, N = sim.batch_size, sim.n_electrodes
    dev, dt = sim.device, sim.dtype
    hi_probe = torch.full((1, N), amp_max, device=dev, dtype=dt)
    reachable = sim.response(hi_probe, n_frames=n_frames)["visible"][0]
    lo = torch.zeros(N, device=dev, dtype=dt)
    hi = torch.full((N,), amp_max, device=dev, dtype=dt)
    frac = (torch.arange(1, B + 1, device=dev, dtype=dt) / (B + 1))[:, None]          # (B, 1)
    for _ in range(iters):
        amps = lo[None] + (hi - lo)[None] * frac                                      # (B, N) ascending rows
        vis = sim.response(amps, n_frames=n_frames)["visible"]                        # (B, N)
        n_inv = (~vis).sum(0)                                                          # monotone => count of low rows
        idx_hi = n_inv.clamp(max=B - 1)
        new_hi = torch.where(n_inv < B, amps.gather(0, idx_hi[None])[0], hi)
        idx_lo = (n_inv - 1).clamp(min=0)
        new_lo = torch.where(n_inv > 0, amps.gather(0, idx_lo[None])[0], lo)
        lo, hi = new_lo, new_hi
    thr = hi.cpu().numpy().astype(np.float64)
    thr[~reachable.cpu().numpy()] = np.inf
    return thr


def build_atlas(sim: ProstheticSimulator) -> Atlas:
    cfg = sim.cfg
    n_frames = sim.n_frames_default
    ref = cfg["atlas"]["ref_current"] or sim.i_max
    x, y = (np.asarray(a, float) for a in sim.coords.cartesian)
    r, _ = sim.coords.polar

    # pixel centres from Dynaphos' own distance maps (argmin), refined to sub-pixel via degrees
    H, W = sim.image_shape
    hemi = cfg["run"]["view_angle"] / 2
    ox, oy = cfg["run"]["origin"]
    col = (x - (ox - hemi)) / (2 * hemi) * (W - 1)
    row = (y - (oy - hemi)) / (2 * hemi) * (H - 1)

    thr_amp = _find_thresholds(sim, n_frames, float(cfg["atlas"]["amp_search_max"]), int(cfg["atlas"]["search_iters"]))

    st = sim.response(torch.full((1, sim.n_electrodes), ref, device=sim.device, dtype=sim.dtype), n_frames=n_frames)
    sigma_deg = sim.to_numpy(st["sigma_deg"][0]).astype(np.float64)
    bright = sim.to_numpy(st["brightness"][0]).astype(np.float64)
    bright = np.where(sim.to_numpy(st["visible"][0]), bright, 0.0)
    mag = sim.to_numpy(sim._sim.magnification).reshape(-1).astype(np.float64)

    meta = {
        "n_frames": n_frames, "i_max": sim.i_max, "ref_current": ref, "levels": sim.levels,
        "image_shape": [H, W], "deg2pix": sim.deg2pix, "view_angle": cfg["run"]["view_angle"],
        "layout_mode": cfg["layout"]["mode"], "layout_seed": cfg["layout"]["seed"], "run_seed": cfg["run"]["seed"],
        "dynaphos_version": _dynaphos_version(),
    }
    return Atlas(
        x_deg=x, y_deg=y, ecc_deg=np.asarray(r, float), row=row, col=col,
        sigma_deg_ref=sigma_deg, sigma_px_ref=sigma_deg * sim.deg2pix, magnification=mag,
        thr_act=sim.to_numpy(sim.base_thr).astype(np.float64), thr_amp=thr_amp, bright_ref=bright, meta=meta,
    )


def _dynaphos_version() -> str:
    try:
        from importlib.metadata import version
        return version("dynaphos")
    except Exception:
        return "unknown"
