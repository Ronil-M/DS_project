"""One place that builds the frozen world (Module C simulator + atlas) for B / E scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from module_c.atlas import Atlas, build_atlas
from module_c.layout import load_config
from module_c.simulator import ProstheticSimulator


def _merge(d: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _merge(d[k], v)
        else:
            d[k] = v
    return d


def build_world(config_path: str | Path = "configs/module_c.yaml", atlas_path: Optional[str | Path] = None,
                overrides: Optional[dict] = None) -> Tuple[dict, ProstheticSimulator, Atlas]:
    """Returns (cfg, sim, atlas). Loads atlas_path if it exists and matches the simulator, else rebuilds (and saves)."""
    cfg = _merge(load_config(config_path), overrides or {})
    sim = ProstheticSimulator(cfg)
    atlas = None
    if atlas_path is not None and Path(atlas_path).exists():
        a = Atlas.load(atlas_path)
        if a.n == sim.n_electrodes and a.meta.get("layout_seed") == cfg["layout"]["seed"] \
                and a.meta.get("image_shape") == list(sim.image_shape):
            atlas = a
    if atlas is None:
        atlas = build_atlas(sim)
        if atlas_path is not None:
            Path(atlas_path).parent.mkdir(parents=True, exist_ok=True)
            atlas.save(atlas_path)
    return cfg, sim, atlas
