"""Module E metrics (plan 7.2). Every function takes phosphene images P (B,1,H,W) in [0,1].

Contour fidelity is measured against the GROUND-TRUTH outline from datasets.shapes (thin mask), not
against B's Gabor target T, so a planner cannot win by fitting its own training target.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import ndimage

from common.imops import center_of_mass, dilate, shift_image


# --------------------------------------------------------------------------- accuracy and statistics
def top1(model, P, y, device="cpu") -> np.ndarray:
    """Per-image correctness (bool array) so CIs and McNemar can be computed."""
    from .classifiers import predict_logits
    return predict_logits(model, P, device=device).argmax(1).numpy() == np.asarray(y)


def bootstrap_ci(correct: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float, float]:
    """(mean, lo, hi) bootstrap CI over test images."""
    rng = np.random.default_rng(seed)
    c = np.asarray(correct, float)
    means = c[rng.integers(0, len(c), (n_boot, len(c)))].mean(1)
    return float(c.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> Dict[str, float]:
    """Exact two-sided McNemar test on paired per-image correctness (same images, two planners)."""
    from scipy.stats import binomtest
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    p = 1.0 if n01 + n10 == 0 else float(binomtest(min(n01, n10), n01 + n10, 0.5).pvalue)
    return {"a_wrong_b_right": n01, "a_right_b_wrong": n10, "p_value": p}


def k_at_accuracy(ks: Sequence[int], acc: Sequence[float], level: float) -> Optional[float]:
    """Smallest K (linearly interpolated) at which accuracy first reaches `level`; None if never."""
    ks, acc = np.asarray(ks, float), np.asarray(acc, float)
    for i in range(len(ks)):
        if acc[i] >= level:
            if i == 0:
                return float(ks[0])
            f = (level - acc[i - 1]) / max(acc[i] - acc[i - 1], 1e-9)
            return float(ks[i - 1] + f * (ks[i] - ks[i - 1]))
    return None


# --------------------------------------------------------------------------- contour fidelity
def contour_fidelity(P: torch.Tensor, contour: torch.Tensor, tol_px: float = 2.0, bright_thr: float = 0.2) -> Dict[str, torch.Tensor]:
    """Per-image precision / recall / F of phosphene mass vs the true outline within `tol_px`.

    precision = sum(P * dilate(contour)) / sum(P)        (phosphene mass that is on the outline)
    recall    = sum(contour * dilate(P_bin)) / sum(contour)   (outline within tol of a visible phosphene)
    """
    P = P.float(); C = (contour > 0.2).float()
    near_c = dilate(C, tol_px)
    pb = (P > bright_thr * P.amax((1, 2, 3), keepdim=True).clamp_min(1e-6)).float()
    pb = pb * (P.amax((1, 2, 3), keepdim=True) > 1e-4).float()
    prec = (P * near_c).sum((1, 2, 3)) / P.sum((1, 2, 3)).clamp_min(1e-6)
    rec = (C * dilate(pb, tol_px)).sum((1, 2, 3)) / C.sum((1, 2, 3)).clamp_min(1e-6)
    f = 2 * prec * rec / (prec + rec).clamp_min(1e-6)
    return {"precision": prec, "recall": rec, "f": f}


# --------------------------------------------------------------------------- visible phosphenes
def visible_blobs(P: torch.Tensor, rel_thr: float = 0.25, min_px: int = 1) -> np.ndarray:
    """Distinct bright blobs per image (connected components above rel_thr * max). Merged phosphenes count once."""
    out = []
    for p in P[:, 0].cpu().numpy():
        if p.max() < 1e-4:
            out.append(0); continue
        lab, n = ndimage.label(p > rel_thr * p.max())
        sizes = ndimage.sum(np.ones_like(p), lab, range(1, n + 1)) if n else []
        out.append(int((np.asarray(sizes) >= min_px).sum()) if n else 0)
    return np.asarray(out)


def visible_electrodes(sim, currents, **kw) -> np.ndarray:
    """Electrodes whose activation exceeds threshold (Module C's own definition). Compare with K."""
    return sim.n_visible(currents, **kw).cpu().numpy()


# --------------------------------------------------------------------------- equivariance probe
def equivariance_probe(arm, x, k: int, shifts_px: Sequence[Tuple[float, float]] = ((0, 6), (6, 0), (-6, 0), (0, -6)),
                       label=None) -> Dict[str, float]:
    """Shift the INPUT by (dy, dx) px, re-plan, and compare how the phosphene centre of mass moves.

    gain = <shift of phosphene COM, input shift> / |input shift|^2 averaged over images and shifts.
    A shape depiction follows the object (gain ~ 1); a class-code pattern does not (gain ~ 0).
    Also returns the mean residual |COM shift - input shift| in px.
    """
    x = torch.as_tensor(x, dtype=torch.float32)
    base = arm.images(x, k, label).image
    c0 = center_of_mass(base)
    gains, resid = [], []
    for dy, dx in shifts_px:
        xs = shift_image(x, dy, dx)
        c1 = center_of_mass(arm.images(xs, k, label).image)
        d = (c1 - c0).cpu()
        v = torch.tensor([dy, dx], dtype=torch.float32)
        gains.append(((d @ v) / (v @ v)).mean().item())
        resid.append((d - v).norm(dim=1).mean().item())
    return {"gain": float(np.mean(gains)), "residual_px": float(np.mean(resid))}


# --------------------------------------------------------------------------- current load
def current_load(currents: np.ndarray, cfg: dict) -> Dict[str, float]:
    """Mean / max current and charge. Charge per pulse = I * pulse width; per second = * frequency (cfg['default_stim'])."""
    c = np.asarray(currents, float)
    active = c > 0
    pw, f = cfg["default_stim"]["pw_default"], cfg["default_stim"]["freq_default"]
    per_img_sum = c.sum(1)
    return {
        "mean_active_uA": float(c[active].mean() * 1e6) if active.any() else 0.0,
        "max_uA": float(c.max() * 1e6) if c.size else 0.0,
        "mean_total_uA": float(per_img_sum.mean() * 1e6),
        "mean_charge_per_pulse_nC": float((per_img_sum * pw).mean() * 1e9),
        "mean_charge_per_s_uC": float((per_img_sum * pw * f).mean() * 1e6),
    }
