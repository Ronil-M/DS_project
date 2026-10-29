"""Constraint-by-construction parametrisation (plan section 6.3) on top of module_c.constraints.

    I_e = m_e * ( thr_e + (I_max - thr_e) * a_e ),   a_e = sigmoid(u_e),   m_e in {0, 1}, sum m = K

* active electrodes are always >= their calibrated threshold (atlas.thr_amp, times `margin`) and <= I_max
* inactive electrodes are exactly 0
* electrodes that are unusable at I_max (atlas.usable) can never be selected
* units are AMPERES, as everywhere in Module C

Quantisation reuses Module C's convention (levels equally spaced multiples of I_max / levels, zero
stays zero) but never rounds an active electrode below its threshold: the lowest allowed level is
ceil(thr / step).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from module_c.constraints import constraint_report, topk_mask


class CurrentParam:
    def __init__(self, atlas, i_max: Optional[float] = None, levels: Optional[int] = None,
                 margin: float = 1.05, device="cpu", dtype=torch.float32):
        self.i_max = float(atlas.meta["i_max"] if i_max is None else i_max)
        self.levels = levels if levels is not None else atlas.meta.get("levels")
        usable = atlas.usable(self.i_max)
        thr = np.where(usable, np.minimum(atlas.thr_amp * margin, self.i_max), self.i_max)
        self.usable = torch.as_tensor(usable, device=device)
        self.thr = torch.as_tensor(thr, device=device, dtype=dtype)               # (N,)
        self.n_usable = int(usable.sum())
        self.device, self.dtype = device, dtype

    # -------------------------------------------------------------- selection
    def clip_k(self, k: int) -> int:
        return int(min(k, self.n_usable))

    def hard_mask(self, scores: torch.Tensor, k: int) -> torch.Tensor:
        """0/1 top-k mask over USABLE electrodes only. scores (B, N)."""
        s = scores.detach().masked_fill(~self.usable.to(scores.device), float("-inf"))
        return topk_mask(s, self.clip_k(k)).to(scores.dtype)

    def ste_mask(self, logits: torch.Tensor, k: int, tau: float = 1.0) -> torch.Tensor:
        """Forward = hard top-k; backward = gradient of sigmoid(logits / tau) (straight-through)."""
        hard = self.hard_mask(logits, k)
        soft = torch.sigmoid(logits / tau)
        return hard + soft - soft.detach()

    # -------------------------------------------------------------- amplitudes
    def amplitude(self, u: torch.Tensor) -> torch.Tensor:
        """Nominal current of every electrode if it were active, in [thr, I_max]. u (B, N) or (N,)."""
        thr = self.thr.to(u.device)
        return thr + (self.i_max - thr) * torch.sigmoid(u)

    def quantise(self, current: torch.Tensor) -> torch.Tensor:
        """Round active currents to the level grid (STE), never below threshold; zero stays zero."""
        if self.levels is None:
            return current
        L = int(self.levels)
        step = self.i_max / L
        lo = torch.ceil(self.thr.to(current.device) / step - 1e-6)                    # lowest legal level index
        q = torch.maximum(torch.round(current / step), lo).clamp(max=L) * step
        q = torch.where(current > 0, q, torch.zeros_like(q))
        return current + (q - current).detach()

    def currents(self, mask: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.quantise(mask * self.amplitude(u))

    # -------------------------------------------------------------- checking
    def report(self, current, k: Optional[int] = None) -> dict:
        c = torch.as_tensor(current, dtype=self.dtype)
        rep = constraint_report(c, self.i_max, k=k, levels=self.levels)
        active = c > 0
        thr = self.thr.to(c.device).expand_as(c)
        rep["below_threshold"] = bool((active & (c < thr * (1 - 1e-5))).any())
        rep["on_unusable"] = bool((active & ~self.usable.to(c.device).expand_as(c)).any())
        rep["ok"] = bool(rep["ok"] and not rep["below_threshold"] and not rep["on_unusable"])
        return rep
