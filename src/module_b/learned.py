"""B2: learned planners.

B2b  AmortisedPlanner  g(image [, label]) -> per-electrode (selection logit s_e, amplitude logit u_e)
B2a  TemplatePlanner   one optimised current vector per class (sanity check, NOT a planner: it ignores
                       where and how large the object is)

Both emit currents through module_b.constraints.CurrentParam, so the constraints hold by construction.
Architecture of B2b (plan 6.3): small CNN encoder over the image -> feature map; for every electrode the
feature map is bilinearly sampled (grid_sample) at that electrode's phosphene centre (atlas row/col) and
passed, with [eccentricity, log phosphene size, threshold], through a shared MLP. This uses the layout
geometry directly and works for any N_elec.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constraints import CurrentParam
from .planner_api import as_image_batch


class AmortisedPlanner(nn.Module):
    name = "B2b_amortised"

    def __init__(self, atlas, levels: Optional[int] = None, n_labels: int = 0, width: int = 16,
                 feat: int = 32, hidden: int = 64, device="cpu"):
        super().__init__()
        self.atlas, self.n_labels = atlas, n_labels
        self.param = CurrentParam(atlas, levels=levels, device=device)
        H, W = atlas.meta["image_shape"]
        gx = 2.0 * atlas.col / (W - 1) - 1.0
        gy = 2.0 * atlas.row / (H - 1) - 1.0
        self.register_buffer("grid", torch.as_tensor(np.stack([gx, gy], -1), dtype=torch.float32)[None, None])
        half = atlas.meta["view_angle"] / 2
        sig = np.maximum(np.nan_to_num(atlas.sigma_px_ref, nan=1.0), 0.25)
        static = np.stack([atlas.ecc_deg / half, np.log(sig), self.param.thr.cpu().numpy() / self.param.i_max], 1)
        self.register_buffer("static", torch.as_tensor(static, dtype=torch.float32))      # (N, 3)

        def block(i, o, k=3, s=1, d=1):
            return nn.Sequential(nn.Conv2d(i, o, k, s, padding=d * (k // 2), dilation=d),
                                 nn.GroupNorm(4, o), nn.ReLU(inplace=True))
        self.encoder = nn.Sequential(block(1, width, 5, 2), block(width, 2 * width, 3, 2), block(2 * width, feat, 3, 2),
                                     block(feat, feat, 3, 1, 2), block(feat, feat, 3, 1, 4))
        self.head = nn.Sequential(nn.Linear(feat + 3 + n_labels, hidden), nn.ReLU(inplace=True),
                                  nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 2))
        nn.init.zeros_(self.head[-1].bias)
        self.to(device)

    @property
    def device(self):
        return self.static.device

    def forward(self, x: torch.Tensor, label: Optional[torch.Tensor] = None):
        """x (B,1,H,W) -> (s, u), each (B, N)."""
        B = x.shape[0]
        f = self.encoder(x)
        f = F.grid_sample(f, self.grid.expand(B, -1, -1, -1), mode="bilinear", padding_mode="border",
                          align_corners=True)[:, :, 0, :].transpose(1, 2)               # (B, N, F)
        parts = [f, self.static[None].expand(B, -1, -1)]
        if self.n_labels:
            if label is None:
                raise ValueError("this planner was built with label conditioning; pass label")
            oh = F.one_hot(torch.as_tensor(label, device=x.device).long(), self.n_labels).to(x.dtype)
            parts.append(oh[:, None, :].expand(-1, f.shape[1], -1))
        out = self.head(torch.cat(parts, -1))
        return out[..., 0], out[..., 1]

    @torch.no_grad()
    def plan(self, x, k: int, label=None) -> np.ndarray:
        was = self.training
        self.eval()
        xb = as_image_batch(x, self.device)
        s, u = self(xb, label)
        cur = self.param.currents(self.param.hard_mask(s, k), u)
        self.train(was)
        return cur.cpu().numpy().astype(np.float32)


class TemplatePlanner:
    """B2a: a fixed current vector per class. plan() needs the label (e.g. from Module A's predict)."""

    name = "B2a_templates"

    def __init__(self, currents: np.ndarray):
        self.currents = np.asarray(currents, np.float32)             # (n_classes, N)

    def plan(self, x, k: int, label=None) -> np.ndarray:
        if label is None:
            raise ValueError("TemplatePlanner needs the class label")
        lab = np.asarray(label.cpu() if torch.is_tensor(label) else label, int).reshape(-1)
        out = self.currents[lab].copy()
        for b in range(out.shape[0]):                                  # honour a smaller K than the template was fit for
            nz = np.flatnonzero(out[b])
            if len(nz) > k:
                out[b, nz[np.argsort(-out[b, nz])[k:]] ] = 0.0
        return out
