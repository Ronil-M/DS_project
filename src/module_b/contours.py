"""Gabor quadrature contour map (Scialom et al., eqs 1-2) and the blurred contour target T.

sigma / lambda are in PIXELS. The paper's 0.06 deg / 0.12 deg at Module C's default 16 px/deg are
~1 px / ~2 px; starting values of 2 / 4 px are more robust for filled shapes. Use
``params_from_sim(sim)`` to derive them from the simulator's deg2pix, and look at the map.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F

from common.imops import gaussian_blur


def gabor_bank(sigma: float = 2.0, lam: float = 4.0, n_orient: int = 8, device=None,
               dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """Even (cosine, zero-mean) and odd (sine) Gabor kernels, each (n_orient, 1, k, k)."""
    r = int(math.ceil(3 * sigma))
    t = torch.arange(-r, r + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(t, t, indexing="ij")
    env = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    ev, od = [], []
    for i in range(n_orient):
        th = math.pi * i / n_orient
        u = xx * math.cos(th) + yy * math.sin(th)
        e = env * torch.cos(2 * math.pi * u / lam)
        e = e - e.mean() * (env / env.mean())          # remove DC so flat regions give 0
        ev.append(e); od.append(env * torch.sin(2 * math.pi * u / lam))
    ev, od = torch.stack(ev)[:, None], torch.stack(od)[:, None]
    n = (ev.pow(2).sum((1, 2, 3), keepdim=True) + od.pow(2).sum((1, 2, 3), keepdim=True)).sqrt()
    return ev / n, od / n


def params_from_sim(sim, sigma_deg: float = 0.12, lam_deg: float = 0.24) -> Tuple[float, float]:
    """Gabor sigma, lambda in px for this simulator (defaults = 2x the paper's degrees, see module doc)."""
    return max(1.0, sigma_deg * sim.deg2pix), max(2.0, lam_deg * sim.deg2pix)


def contour_map(x: torch.Tensor, sigma: float = 2.0, lam: float = 4.0, n_orient: int = 8,
                pool: str = "max") -> torch.Tensor:
    """C = sqrt(even^2 + odd^2) pooled over orientation, normalised to max 1 per image. (B,1,H,W)."""
    ev, od = gabor_bank(sigma, lam, n_orient, x.device, x.dtype)
    pad = ev.shape[-1] // 2
    e = F.conv2d(x, ev, padding=pad)
    o = F.conv2d(x, od, padding=pad)
    c = torch.sqrt(e ** 2 + o ** 2 + 1e-12)
    c = c.amax(1, keepdim=True) if pool == "max" else c.sum(1, keepdim=True)
    return c / c.amax((1, 2, 3), keepdim=True).clamp_min(1e-6)


def contour_target(x: torch.Tensor, blur_sigma: float = 1.5, **kw) -> torch.Tensor:
    """Contour target T: Gabor contour map blurred by ~one phosphene radius, renormalised to max 1."""
    t = gaussian_blur(contour_map(x, **kw), blur_sigma)
    return t / t.amax((1, 2, 3), keepdim=True).clamp_min(1e-6)
