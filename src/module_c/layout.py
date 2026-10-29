"""Electrode / phosphene layout construction (Module C, step C3).

Dynaphos represents an electrode through the visual-field location of its phosphene
(a `dynaphos.utils.Map`). Three ways of producing that Map are supported:

* ``probabilistic``  exactly N phosphenes inside the field of view, eccentricities drawn
                     proportional to cortical magnification (dense fovea, sparse periphery).
* ``cortical_grid``  a regular electrode grid on the (simulated) V1 surface of both
                     hemispheres, mapped to the visual field with the cortex model.
                     Realistic but the number of electrodes that land inside a small field
                     of view can be small, so always inspect the count.
* ``array``          explicit x/y positions in visual-field degrees (e.g. from implant planning).

IMPORTANT: ``GaussianSimulator`` removes phosphenes outside the field of view *in place* on the
Map it is given. Electrode index i therefore always refers to the Map **after** the simulator
has been constructed. `ProstheticSimulator` takes care of this; do not build atlases from a
Map taken before simulator construction.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from dynaphos.cortex_models import (
    get_visual_field_coordinates_from_cortex_full,
    get_visual_field_coordinates_probabilistically,
)
from dynaphos.utils import Map


def load_config(path: str | Path) -> dict:
    """Load the module_c YAML (Dynaphos params + module_c sections)."""
    with open(path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    if cfg["run"]["batch_size"] < 1:
        raise ValueError("run.batch_size must be >= 1 (module_c uses a fixed internal batch).")
    if cfg["layout"]["mode"] not in ("probabilistic", "cortical_grid", "array"):
        raise ValueError(f"Unknown layout.mode: {cfg['layout']['mode']}")
    lv = cfg["constraints"]["levels"]
    if lv is not None and int(lv) < 2:
        raise ValueError("constraints.levels must be null or >= 2.")
    if cfg["render"]["n_frames"] < 1:
        raise ValueError("render.n_frames must be >= 1.")


def build_layout(
    cfg: dict,
    rng: Optional[np.random.Generator] = None,
    x_deg: Optional[np.ndarray] = None,
    y_deg: Optional[np.ndarray] = None,
) -> Map:
    """Return a Map of phosphene locations (visual-field degrees) for the configured layout."""
    lc = cfg["layout"]
    rng = np.random.default_rng(lc["seed"]) if rng is None else rng
    mode = lc["mode"]
    n = int(lc["n_electrodes"])

    if mode == "probabilistic":
        # needs run.view_angle, run.min_angle and cortex_model: pass the full config
        return get_visual_field_coordinates_probabilistically(copy.deepcopy(cfg), n, rng)

    if mode == "cortical_grid":
        coords = get_visual_field_coordinates_from_cortex_full(cfg["cortex_model"], None, rng)
        x, y = coords.cartesian
        # keep only phosphenes inside the field of view *before* subsampling, so that `n`
        # refers to usable electrodes rather than to electrodes in the far periphery
        hemi = cfg["run"]["view_angle"] / 2
        ox, oy = cfg["run"]["origin"]
        inside = (np.abs(x - ox) < hemi) & (np.abs(y - oy) < hemi)
        idx = np.flatnonzero(inside)
        if len(idx) > n:
            idx = np.sort(rng.choice(idx, size=n, replace=False))
        return Map(x=x[idx], y=y[idx])

    if mode == "array":
        if x_deg is None or y_deg is None:
            raise ValueError("layout.mode='array' needs x_deg and y_deg arrays.")
        return Map(x=np.asarray(x_deg, float), y=np.asarray(y_deg, float))

    raise ValueError(mode)
