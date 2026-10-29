"""Small differentiable image operators shared by Module B (losses, contours) and Module E (metrics).

All tensors are (B, 1, H, W) float32 unless stated otherwise.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def gaussian_kernel1d(sigma: float, device=None, dtype=torch.float32) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    t = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    return k / k.sum()


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur with zero padding. sigma <= 0 returns x unchanged."""
    if sigma <= 0:
        return x
    k = gaussian_kernel1d(sigma, x.device, x.dtype)
    r = (len(k) - 1) // 2
    c = x.shape[1]
    kh = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    kw = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    x = F.conv2d(x, kh, padding=(r, 0), groups=c)
    return F.conv2d(x, kw, padding=(0, r), groups=c)


def dilate(x: torch.Tensor, radius_px: float) -> torch.Tensor:
    """Grey-scale dilation with a square window of half-width ceil(radius_px)."""
    r = int(math.ceil(radius_px))
    if r <= 0:
        return x
    return F.max_pool2d(x, kernel_size=2 * r + 1, stride=1, padding=r)


def shift_image(x: torch.Tensor, dy: float, dx: float) -> torch.Tensor:
    """Translate by (dy, dx) pixels (bilinear, zero fill). Positive dy moves content DOWN (row index up)."""
    B, _, H, W = x.shape
    theta = torch.tensor([[1.0, 0.0, -2.0 * dx / max(W - 1, 1)],
                          [0.0, 1.0, -2.0 * dy / max(H - 1, 1)]], dtype=x.dtype, device=x.device)
    grid = F.affine_grid(theta[None].expand(B, -1, -1), x.shape, align_corners=True)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def center_of_mass(img: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """(B, 2) intensity-weighted centre (row, col) in pixels."""
    B, _, H, W = img.shape
    rows = torch.arange(H, device=img.device, dtype=img.dtype).view(1, 1, H, 1)
    cols = torch.arange(W, device=img.device, dtype=img.dtype).view(1, 1, 1, W)
    m = img.sum((1, 2, 3)) + eps
    return torch.stack([(img * rows).sum((1, 2, 3)) / m, (img * cols).sum((1, 2, 3)) / m], -1)
