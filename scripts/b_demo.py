#!/usr/bin/env python
"""Module B demo / validation (plan 6.2 "Validate"): R0, R1 and B1 renders at K = 8, 16, 32 for every class.

    python scripts/b_demo.py --out outputs/module_b
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.world import build_world
from datasets.shapes import ShapeSpec, make_dataset
from module_b.heuristic import HeuristicPlanner, RandomPlanner
from module_b.planner_api import PlannerArm, ScoreboardArm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/module_c.yaml")
    ap.add_argument("--atlas", default="outputs/module_c/atlas.npz")
    ap.add_argument("--out", default="outputs/module_b")
    ap.add_argument("--k", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--field-deg", type=float, default=10.0)
    a = ap.parse_args()
    cfg, sim, atlas = build_world(a.config, a.atlas)
    spec = ShapeSpec.from_sim(sim, field_deg=a.field_deg)
    ds = make_dataset(1, seed=7, spec=spec)
    arms = [ScoreboardArm(sim.image_shape), PlannerArm(RandomPlanner(atlas), sim), PlannerArm(HeuristicPlanner(atlas, sim), sim)]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    n_cls = len(ds["classes"])
    fig, ax = plt.subplots(n_cls * len(arms), len(a.k), figsize=(2.6 * len(a.k), 2.6 * n_cls * len(arms)))
    idx = [int(np.where(ds["y"] == c)[0][0]) for c in range(n_cls)]
    for ci, i in enumerate(idx):
        x = ds["x"][i:i + 1]; cname = ds["classes"][ci]
        for ai, arm in enumerate(arms):
            for ki, k in enumerate(a.k):
                out = arm.images(x, k, label=ds["y"][ci:ci + 1])
                n_vis = int((out.currents > 0).sum()) if out.currents is not None else k
                r = ci * len(arms) + ai
                ax[r, ki].imshow(out.image[0, 0].numpy(), cmap="gray", vmin=0, vmax=1); ax[r, ki].axis("off")
                ax[r, ki].set_title(f"{ds['classes'][ci]} | {arm.name} | K={k} ({n_vis} active)", fontsize=6)
    fig.tight_layout(); fig.savefig(Path(a.out) / "b_demo.png", dpi=110)
    print("saved", Path(a.out) / "b_demo.png")


if __name__ == "__main__":
    main()
