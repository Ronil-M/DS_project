"""Per-electrode current that yields a target brightness (used by B1, plan 6.2 "Assign currents").

Brightness is a monotone sigmoid of the effective stimulus, so the current for a target brightness is
found with the same (B+1)-ary search that module_c.atlas uses for thresholds. It is measured on the
frozen simulator with each electrode driven ALONE (single-phosphene brightness; merging is ignored).
"""
from __future__ import annotations

import numpy as np
import torch


def currents_for_brightness(sim, atlas, target: float = 0.7, iters: int = 6, margin: float = 1.05) -> np.ndarray:
    """(N,) amperes in [margin*thr, i_max]; 0 for unusable electrodes. Clipped to i_max if the target is unreachable."""
    N, B = sim.n_electrodes, sim.batch_size
    dev, dt = sim.device, sim.dtype
    i_max = float(atlas.meta["i_max"])
    usable = atlas.usable(i_max)
    lo = torch.as_tensor(np.where(usable, np.minimum(atlas.thr_amp * margin, i_max), i_max), device=dev, dtype=dt)
    hi = torch.full((N,), i_max, device=dev, dtype=dt)
    frac = (torch.arange(1, B + 1, device=dev, dtype=dt) / (B + 1))[:, None]
    for _ in range(iters):
        amps = lo[None] + (hi - lo)[None] * frac
        br = sim.response(amps)["brightness"]                          # (B, N), ascending in amplitude
        below = (br < target).sum(0)                                   # rows that are too dim
        idx_hi = below.clamp(max=B - 1)
        new_hi = torch.where(below < B, amps.gather(0, idx_hi[None])[0], hi)
        idx_lo = (below - 1).clamp(min=0)
        new_lo = torch.where(below > 0, amps.gather(0, idx_lo[None])[0], lo)
        lo, hi = new_lo, new_hi
    out = hi.cpu().numpy().astype(np.float64)
    out[~usable] = 0.0
    return out.astype(np.float32)
