"""Loss terms of plan section 6.3:  L = lam_c L_contour + lam_t L_task + lam_s L_budget.

L_contour = w_mse * MSE(blur(P), T)/mean mass  +  w_prec * precision-penalty  +  w_rec * recall-penalty
  * precision penalty: fraction of phosphene mass farther than `tol_px` from the contour
                       = sum P*(1 - dilate(T_bin)) / sum P         (stops the all-bright solution)
  * recall penalty:    fraction of contour left uncovered
                       = sum T*(1 - clip(blur(P),0,1)) / sum T     (stops the dark-image solution)
Both are normalised by mass instead of by pixel count (plan: plain means) so the weights do not depend
on image size or K. P, T are (B, 1, H, W); T is module_b.contours.contour_target.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from common.imops import dilate, gaussian_blur


def contour_loss(P: torch.Tensor, T: torch.Tensor, blur_sigma: float = 1.5, tol_px: float = 2.0,
                 w_mse: float = 1.0, w_prec: float = 1.0, w_rec: float = 1.0,
                 t_bin: float = 0.3, eps: float = 1e-6) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    bp = gaussian_blur(P, blur_sigma)
    tm = T.sum((1, 2, 3)) + eps
    mse = ((bp - T) ** 2).sum((1, 2, 3)) / tm
    near = dilate((T > t_bin).to(P.dtype), tol_px)
    prec = (P * (1.0 - near)).sum((1, 2, 3)) / (P.sum((1, 2, 3)) + eps)
    rec = (T * (1.0 - bp.clamp(0, 1))).sum((1, 2, 3)) / tm
    loss = (w_mse * mse + w_prec * prec + w_rec * rec).mean()
    return loss, {"mse": mse.mean().detach(), "precision_pen": prec.mean().detach(), "recall_pen": rec.mean().detach()}


def task_loss(classifier: torch.nn.Module, P: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of the frozen TRAINING classifier T_c on phosphene images (never the evaluator E)."""
    return F.cross_entropy(classifier(P), y)


def budget_loss(currents: torch.Tensor, i_max: float) -> torch.Tensor:
    """Mean total current per image, in units of I_max (0 = off)."""
    return currents.sum(-1).mean() / i_max
