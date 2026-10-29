#!/usr/bin/env python
"""What validation contour loss is "good"?  Reference values on YOUR validation images, computed exactly like the
validation line in b_train.py (same loss, same images if --max-per-class and --seed match the training run):

    python scripts/b_baselines.py --data-dir data/split --k 50 --max-per-class 40 --n-val 80
    python scripts/b_baselines.py --data-dir data/split --k 50 --max-per-class 40 --n-val 80 --ckpt outputs/module_b/b2b_k50_s0_c1_t0_real.ckpt

Rows: dark (no phosphenes at all), random (K random usable electrodes), heuristic B1 (contour-following rule), and,
with --ckpt, your trained planner (best weights stored in a .ckpt or the final .pt). Lower is better. A trained planner
that is not clearly below `dark` and `random`, and far from `heuristic`, has not learned.
Columns: loss = mse + precision penalty + recall penalty (see module_b/losses.py).
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, torch, yaml

from common.world import build_world
from datasets.real_dataset import make_real_datasets
from datasets.shapes import ShapeSpec
from module_b.contours import contour_target
from module_b.heuristic import HeuristicPlanner, RandomPlanner
from module_b.learned import AmortisedPlanner
from module_b.losses import contour_loss


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--experiment", default="configs/experiment.yaml")
    ap.add_argument("--data-dir", required=True); ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max-per-class", type=int, default=100)
    ap.add_argument("--val-frac", type=float, default=0.2); ap.add_argument("--no-rotate", action="store_true")
    ap.add_argument("--n-val", type=int, default=80, help="evaluate on at most this many validation images")
    ap.add_argument("--ckpt", default=None, help="b_train .ckpt or .pt of the planner to compare")
    ap.add_argument("--n-labels", type=int, default=0)
    a = ap.parse_args()
    ex = yaml.safe_load(open(a.experiment)); H = ex["heuristic"]
    cfg, sim, atlas = build_world(ex["paths"]["module_c_config"], ex["paths"]["atlas"])
    spec = ShapeSpec.from_sim(sim)
    _, va = make_real_datasets(a.data_dir, spec, val_frac=a.val_frac, n_aug_train=1, seed=a.seed, rotate=not a.no_rotate,
                               max_per_class=a.max_per_class or None, keep_contour=False, log=lambda *x, **k: None)
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(va["y"]), min(a.n_val, len(va["y"])), replace=False))      # spread over all classes
    x = va["x"][idx]
    arms = {"dark (no phosphenes)": None,
            f"random K={a.k}": RandomPlanner(atlas, seed=0),
            f"heuristic B1 K={a.k}": HeuristicPlanner(atlas, sim, method=H["method"], pool=H["pool"], pool_excl_deg=H["pool_excl_deg"],
                                                      excl_deg=H["excl_deg"], floor_frac=H["floor_frac"], target_brightness=H["target_brightness"])}
    if a.ckpt:
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)                  # your own file
        state = ck.get("state") or ck.get("best_state")
        p = AmortisedPlanner(atlas, levels=sim.levels, n_labels=ck.get("n_labels", a.n_labels)); p.load_state_dict(state)
        arms[f"TRAINED planner K={a.k}"] = p.eval()
    print(f"{len(x)} validation images, K={a.k}\n{'arm':<28}{'loss':>8}{'mse':>8}{'precision pen':>15}{'recall pen':>12}   (lower is better)")
    for name, pl in arms.items():
        tot, parts, n = 0.0, {"mse": 0.0, "precision_pen": 0.0, "recall_pen": 0.0}, 0
        for i in range(0, len(x), 8):
            xb = torch.as_tensor(x[i:i + 8]); T = contour_target(xb, 1.5)
            with torch.no_grad():
                P = torch.zeros_like(xb) if pl is None else sim.render(torch.as_tensor(pl.plan(xb, a.k, None), device=sim.device))
                l, d = contour_loss(P, T.to(P.device), 1.5, 2.0)
            tot += float(l) * len(xb); n += len(xb)
            for key in parts: parts[key] += float(d[key]) * len(xb)
        print(f"{name:<28}{tot / n:>8.3f}{parts['mse'] / n:>8.3f}{parts['precision_pen'] / n:>15.3f}{parts['recall_pen'] / n:>12.3f}")


if __name__ == "__main__":
    main()
