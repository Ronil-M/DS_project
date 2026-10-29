"""Procedural shape generator (shared by Module A, B training and Module E).

Every image comes with a thin ground-truth *contour mask*, so Module B's Gabor contour map can be
cross-checked and Module E can score contour fidelity against a contour that B never saw.

Geometry follows Module C: the object spans `scale * field_deg` degrees, converted to pixels with
`deg2pix` (``ProstheticSimulator.deg2pix``) and drawn on an image of the simulator's own
``image_shape``. Use ``ShapeSpec.from_sim(sim)`` to guarantee that the two agree.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

CLASSES = ["circle", "square", "triangle", "line", "cross", "star", "ellipse"]
FOUR = CLASSES[:4]


@dataclass
class ShapeSpec:
    image_shape: Tuple[int, int] = (256, 256)
    deg2pix: float = 16.0
    field_deg: float = 10.0                    # object spans up to this many degrees (Scialom et al.)
    scale_range: Tuple[float, float] = (0.5, 1.0)
    max_shift_deg: float = 1.0
    rotate: bool = True
    thickness_deg: Tuple[float, float] = (0.15, 0.5)   # stroke width of outlined shapes / lines

    @classmethod
    def from_sim(cls, sim, **kw) -> "ShapeSpec":
        return cls(image_shape=tuple(sim.image_shape), deg2pix=float(sim.deg2pix), **kw)


def _polygon(name: str, R: float, rot: float, aspect: float = 1.0) -> Tuple[np.ndarray, bool]:
    """Vertices (n, 2) as (x, y) around the origin, and whether the polyline is closed."""
    if name == "circle":
        t = np.linspace(0, 2 * np.pi, 96, endpoint=False)
        v = np.stack([R * np.cos(t), R * np.sin(t)], 1)
    elif name == "ellipse":
        t = np.linspace(0, 2 * np.pi, 96, endpoint=False)
        v = np.stack([R * np.cos(t), R * aspect * np.sin(t)], 1)
    elif name == "square":
        a = R / math.sqrt(2)
        v = np.array([[-a, -a], [a, -a], [a, a], [-a, a]])
    elif name == "triangle":
        t = np.pi / 2 + np.arange(3) * 2 * np.pi / 3
        v = np.stack([R * np.cos(t), R * np.sin(t)], 1)
    elif name == "line":
        v = np.array([[-R, 0.0], [R, 0.0]])
        return _rot(v, rot), False
    elif name == "cross":
        a, b = R * 0.95, R * 0.28
        v = np.array([[-b, -a], [b, -a], [b, -b], [a, -b], [a, b], [b, b],
                      [b, a], [-b, a], [-b, b], [-a, b], [-a, -b], [-b, -b]])
    elif name == "star":
        t = np.pi / 2 + np.arange(10) * np.pi / 5
        rad = np.where(np.arange(10) % 2 == 0, R, R * 0.42)
        v = np.stack([rad * np.cos(t), rad * np.sin(t)], 1)
    else:
        raise ValueError(name)
    return _rot(v, rot), True


def _rot(v: np.ndarray, a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return v @ np.array([[c, -s], [s, c]]).T


def render_shape(name: str, rng: np.random.Generator, spec: ShapeSpec,
                 filled: Optional[bool] = None, canonical: bool = False):
    """Return (image, contour, meta). image/contour are (H, W) float32 in [0, 1]."""
    H, W = spec.image_shape
    scale = 0.75 if canonical else rng.uniform(*spec.scale_range)
    R = 0.5 * scale * spec.field_deg * spec.deg2pix
    R = min(R, 0.5 * min(H, W) - 3)
    rot = 0.0 if (canonical or not spec.rotate) else rng.uniform(0, 2 * np.pi)
    aspect = 0.6 if canonical else rng.uniform(0.45, 0.75)
    j = 0.0 if canonical else min(spec.max_shift_deg * spec.deg2pix, max(0.0, 0.5 * min(H, W) - R - 3))
    cy = H / 2 + (rng.uniform(-j, j) if j else 0.0)
    cx = W / 2 + (rng.uniform(-j, j) if j else 0.0)
    if filled is None:
        filled = bool(rng.integers(0, 2)) if not canonical else True
    if name == "line":
        filled = False
    thick = max(1, int(round(rng.uniform(*spec.thickness_deg) * spec.deg2pix))) if not canonical else 3

    v, closed = _polygon(name, R, rot, aspect)
    pts = np.rint((v + np.array([cx, cy])) * 16).astype(np.int32)      # cv2 shift=4 sub-pixel
    img = np.zeros((H, W), np.uint8)
    con = np.zeros((H, W), np.uint8)
    if filled:
        cv2.fillPoly(img, [pts], 255, lineType=cv2.LINE_AA, shift=4)
    else:
        cv2.polylines(img, [pts], closed, 255, thickness=thick, lineType=cv2.LINE_AA, shift=4)
    cv2.polylines(con, [pts], closed, 255, thickness=1, lineType=cv2.LINE_AA, shift=4)
    meta = {"class": name, "centre_rc": [float(cy), float(cx)], "radius_px": float(R), "angle": float(rot),
            "filled": bool(filled), "thickness_px": int(thick), "scale": float(scale)}
    return img.astype(np.float32) / 255.0, con.astype(np.float32) / 255.0, meta


def make_dataset(n_per_class: int, seed: int, spec: ShapeSpec, classes: Sequence[str] = FOUR) -> Dict:
    """Seeded dataset. Returns dict(x (N,1,H,W), contour (N,1,H,W), y (N,), meta list, classes).

    Use different seeds for train / val / test (plan, step A1: never the same seed for two splits).
    """
    rng = np.random.default_rng(seed)
    H, W = spec.image_shape
    n = n_per_class * len(classes)
    x = np.zeros((n, 1, H, W), np.float32)
    c = np.zeros((n, 1, H, W), np.float32)
    y = np.zeros(n, np.int64)
    meta: List[dict] = []
    i = 0
    for k, name in enumerate(classes):
        for _ in range(n_per_class):
            xi, ci, m = render_shape(name, rng, spec)
            x[i, 0], c[i, 0], y[i] = xi, ci, k
            meta.append(m)
            i += 1
    perm = rng.permutation(n)
    return {"x": x[perm], "contour": c[perm], "y": y[perm], "meta": [meta[p] for p in perm],
            "classes": list(classes)}


def canonical_prototypes(spec: ShapeSpec, classes: Sequence[str] = FOUR):
    """One centred, un-rotated, filled exemplar per class (for B2a templates). Returns (x, contour) (C,1,H,W)."""
    rng = np.random.default_rng(0)
    xs, cs = [], []
    for name in classes:
        xi, ci, _ = render_shape(name, rng, spec, canonical=True)
        xs.append(xi[None]); cs.append(ci[None])
    return np.stack(xs), np.stack(cs)
