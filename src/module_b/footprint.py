"""Sampling image-space maps at electrode phosphene centres, weighted by each phosphene's size.

This is the electrode-space version of the paper's "overlap" (eq. 3): for electrode e we compute
``(G_sigma_e * map)(row_e, col_e)`` with G a unit-sum Gaussian whose sigma is the electrode's
phosphene sigma from the atlas. Snapping is done in image/visual-field coordinates (atlas row, col),
never by electrode index.

Used for (i) B1 electrode scores and (ii) the selection gradient of the learned planner.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from common.imops import gaussian_blur


class GaussianSampler:
    def __init__(self, atlas, image_shape, n_bins: int = 6, sigma_floor_px: float = 1.0,
                 device="cpu", dtype=torch.float32):
        H, W = image_shape
        self.H, self.W = H, W
        sig = np.maximum(np.nan_to_num(atlas.sigma_px_ref, nan=sigma_floor_px), sigma_floor_px)
        self.sigma_px = torch.as_tensor(sig, device=device, dtype=dtype)                 # (N,)
        lo, hi = float(sig.min()), float(sig.max())
        self.bin_sigmas = np.unique(np.geomspace(lo, hi, n_bins)) if hi > lo * 1.05 else np.array([lo])
        bins = np.abs(np.log(sig)[:, None] - np.log(self.bin_sigmas)[None]).argmin(1)
        self.bin_of = torch.as_tensor(bins, device=device)
        gx = 2.0 * atlas.col / (W - 1) - 1.0
        gy = 2.0 * atlas.row / (H - 1) - 1.0
        self.grid = torch.as_tensor(np.stack([gx, gy], -1), device=device, dtype=dtype)[None, None]  # (1,1,N,2)
        self.device, self.dtype = device, dtype

    def sample(self, maps: torch.Tensor) -> torch.Tensor:
        """maps (B,1,H,W) -> (B,N): Gaussian-weighted mean of the map around every phosphene centre."""
        B = maps.shape[0]
        out = torch.zeros(B, self.grid.shape[2], device=maps.device, dtype=maps.dtype)
        for b, s in enumerate(self.bin_sigmas):
            sel = (self.bin_of == b)
            if not bool(sel.any()):
                continue
            blurred = gaussian_blur(maps, float(s))
            g = F.grid_sample(blurred, self.grid.to(maps.device).expand(B, -1, -1, -1), mode="bilinear",
                              padding_mode="zeros", align_corners=True)[:, 0, 0]      # (B, N)
            out = torch.where(sel.to(maps.device)[None], g, out)
        return out

    def footprint_dot(self, maps: torch.Tensor) -> torch.Tensor:
        """Sum over pixels of maps * (unit-peak Gaussian footprint of electrode e): (B, N).

        = 2 pi sigma^2 * sample(maps). With maps = dLoss/dImage this is the first-order change of the
        loss when electrode e is switched on at unit peak brightness (surrogate selection gradient).
        """
        return self.sample(maps) * (2.0 * np.pi * self.sigma_px.to(maps.device) ** 2)[None]
