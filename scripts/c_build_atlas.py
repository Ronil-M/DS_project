#!/usr/bin/env python
"""Module C steps C3 + C4: build the electrode atlas, plot the layout, render a random-current test.

    python scripts/c_build_atlas.py --config configs/module_c.yaml --out outputs/module_c

Writes: atlas.npz, atlas_summary.json, layout.png, random_render.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from module_c.atlas import build_atlas  # noqa: E402
from module_c.layout import load_config  # noqa: E402
from module_c.simulator import ProstheticSimulator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/module_c.yaml")
    ap.add_argument("--out", default="outputs/module_c")
    ap.add_argument("--k", type=int, nargs="+", default=[5, 16, 48], help="K values for the random render test")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    sim = ProstheticSimulator(cfg)
    atlas = build_atlas(sim)
    atlas.save(out / "atlas.npz")
    summ = atlas.summary()
    (out / "atlas_summary.json").write_text(json.dumps({**summ, **atlas.meta}, indent=2))
    print(json.dumps(summ, indent=2))
    if summ["frac_sigma_below_0.5px"] > 0.5:
        print("WARNING: more than half of the phosphenes have sigma < 0.5 px at the reference current; "
              "they will alias in the rendered image. Increase run.resolution or reduce run.view_angle.")
    if summ["n_usable_in_central_5deg_radius"] < 64:
        print("WARNING: fewer than 64 usable electrodes within 5 deg of the centre; K sweeps up to 64 are not reachable.")

    # --- layout figure
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    thr = np.where(np.isfinite(atlas.thr_amp), atlas.thr_amp * 1e6, np.nan)
    sc = ax[0].scatter(atlas.x_deg, atlas.y_deg, c=thr, s=8, cmap="viridis")
    for rad in (5, atlas.meta["view_angle"] / 2):
        ax[0].add_patch(plt.Circle((0, 0), rad, fill=False, ls="--", color="k", lw=0.8))
    ax[0].set_aspect("equal"); ax[0].set_title("Phosphene centres (colour = threshold, uA)")
    ax[0].set_xlabel("x (deg)"); ax[0].set_ylabel("y (deg)"); plt.colorbar(sc, ax=ax[0])
    ax[1].hist(thr[np.isfinite(thr)], bins=40)
    ax[1].axvline(atlas.meta["i_max"] * 1e6, color="r", label="i_max")
    ax[1].set_xlabel("threshold current (uA)"); ax[1].set_title("Per-electrode thresholds"); ax[1].legend()
    fig.tight_layout(); fig.savefig(out / "layout.png", dpi=130); plt.close(fig)

    # --- C4: random-current render
    rng = np.random.default_rng(0)
    ks = args.k
    fig, axes = plt.subplots(1, len(ks), figsize=(4 * len(ks), 4))
    axes = np.atleast_1d(axes)
    for a, k in zip(axes, ks):
        cur = atlas.random_currents(k, 1, rng)
        img, st = sim.render(cur, return_state=True)
        a.imshow(sim.to_numpy(img[0, 0]), cmap="gray", vmin=0, vmax=1)
        a.set_title(f"K={k} active, {int(st['visible'].sum())} visible"); a.axis("off")
    fig.tight_layout(); fig.savefig(out / "random_render.png", dpi=130); plt.close(fig)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
