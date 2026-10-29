"""Common interface so Module E (and the experiment drivers) can treat every arm the same way.

    planner.plan(x, k, label=None) -> currents (B, N) float32 numpy, AMPERES  (module_c convention)
    arm.images(x, k, label=None, noise=None, generator=None) -> ArmOutput(image (B,1,H,W) tensor, currents|None)

`x` is the clean input image batch (B, 1, H, W) in [0, 1] at Module C's image_shape (numpy or torch).
Every planner returns currents that satisfy module_b.constraints (>= threshold when active, <= I_max,
at most K non-zero, exactly 0 elsewhere). The scoreboard arm R0 bypasses Dynaphos and has no currents.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
import torch

from common.dots import place_dots, render_dots
from .contours import contour_map


def as_image_batch(x, device="cpu", dtype=torch.float32) -> torch.Tensor:
    t = torch.as_tensor(x, dtype=dtype, device=device)
    if t.ndim == 2:
        t = t[None, None]
    elif t.ndim == 3:
        t = t[None] if t.shape[0] == 1 else t[:, None]
    if t.ndim != 4 or t.shape[1] != 1:
        raise ValueError(f"image batch must be (B,1,H,W); got {tuple(t.shape)}")
    return t


class Planner(Protocol):
    name: str

    def plan(self, x, k: int, label=None) -> np.ndarray: ...


@dataclass
class ArmOutput:
    image: torch.Tensor                       # (B, 1, H, W) in [0, 1]
    currents: Optional[np.ndarray] = None     # (B, N) amperes, None for the scoreboard arm


class PlannerArm:
    """Planner + frozen simulator. The only place where B's output meets C's render()."""

    def __init__(self, planner: Planner, sim):
        self.planner, self.sim, self.name = planner, sim, planner.name

    def with_sim(self, sim) -> "PlannerArm":
        """Same planner (still using the NOMINAL atlas) rendered through another simulator, e.g. a jittered layout."""
        return PlannerArm(self.planner, sim)

    def images(self, x, k, label=None, noise=None, generator=None) -> ArmOutput:
        cur = self.planner.plan(x, k, label)
        with torch.no_grad():
            img = self.sim.render(cur, noise=noise, generator=generator)
        return ArmOutput(img, cur)


class ScoreboardArm:
    """R0: K idealised Gaussian dots on the Gabor contour of the input, no electrode layout, no Dynaphos."""

    name = "R0_scoreboard"

    def __init__(self, image_shape, dot_sigma_px: float = 1.5, contour_kw: Optional[dict] = None, seed: int = 0):
        self.shape, self.sigma = tuple(image_shape), dot_sigma_px
        self.kw = contour_kw or {}
        self.seed = seed

    def images(self, x, k, label=None, noise=None, generator=None) -> ArmOutput:
        xb = as_image_batch(x)
        cm = contour_map(xb, **self.kw)[:, 0].numpy()
        rng = np.random.default_rng(self.seed)
        out = np.zeros((xb.shape[0], 1, *self.shape), np.float32)
        for b in range(xb.shape[0]):
            pts = place_dots(cm[b], k, rng=None)
            if noise is not None and noise.p_drop > 0 and len(pts):       # only dropout is meaningful for dots
                pts = pts[rng.random(len(pts)) >= noise.p_drop]
            out[b, 0] = render_dots(pts, self.shape, self.sigma)
        return ArmOutput(torch.from_numpy(out), None)
