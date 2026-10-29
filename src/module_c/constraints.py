"""Current constraints for Module B/C (Step C: "keep currents within the allowed range").

All currents are in AMPERES, shape (..., N_electrodes), 0 = electrode inactive.
"""
from __future__ import annotations

from typing import Optional

import torch


def quantize_ste(current: torch.Tensor, i_max: float, levels: Optional[int]) -> torch.Tensor:
    """Round active currents to `levels` equally spaced values in (0, i_max]; straight-through gradient.

    Zero stays exactly zero (inactive electrode). With levels=None this is the identity.
    """
    if levels is None:
        return current
    step = i_max / float(levels)
    q = torch.clamp(torch.round(current / step), 0, levels) * step
    return current + (q - current).detach()


def topk_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Hard 0/1 mask of the k largest scores along the last dim (no gradient)."""
    k = int(min(max(k, 0), scores.shape[-1]))
    mask = torch.zeros_like(scores)
    if k > 0:
        idx = scores.topk(k, dim=-1).indices
        mask.scatter_(-1, idx, 1.0)
    return mask


def project_currents(current: torch.Tensor, i_max: float, k: Optional[int] = None,
                     levels: Optional[int] = None) -> torch.Tensor:
    """Clamp to [0, i_max], optionally keep only the k largest, optionally quantise (STE)."""
    c = current.clamp(0.0, i_max)
    if k is not None:
        c = c * topk_mask(c.detach(), k)
    return quantize_ste(c, i_max, levels)


def constraint_report(current: torch.Tensor, i_max: float, k: Optional[int] = None,
                      levels: Optional[int] = None, atol: float = 1e-12) -> dict:
    """Check a current tensor against the constraints. Returns a dict; `ok` is the overall verdict."""
    c = current.detach()
    active = (c > 0).sum(-1)
    rep = {
        "max_current": float(c.max()) if c.numel() else 0.0,
        "min_nonzero": float(c[c > 0].min()) if (c > 0).any() else 0.0,
        "max_active": int(active.max()) if c.numel() else 0,
        "negative": bool((c < -atol).any()),
        "above_i_max": bool((c > i_max * (1 + 1e-6) + atol).any()),
    }
    rep["k_ok"] = True if k is None else rep["max_active"] <= k
    if levels is None:
        rep["levels_ok"] = True
    else:
        step = i_max / levels
        r = c / step
        rep["levels_ok"] = bool(((r - r.round()).abs() < 1e-3).all())
    rep["ok"] = not rep["negative"] and not rep["above_i_max"] and rep["k_ok"] and rep["levels_ok"]
    return rep
