#!/usr/bin/env python
"""How good are the trained classifiers, as a function of the number of dots?  (no planner, no Dynaphos)

    python scripts/e_check_evaluators.py --data-dir data/split --ks 4 8 16 32 64 --n-test 200

Puts K idealised dots (the scoreboard arm R0) on the true contour of your held-out test/ images and reports the
accuracy of E_main, E_alt and T_c at each K, next to chance (1 / number of classes). The headline "held-out dot
accuracy" printed by e_train_evaluators.py averages over K = 3 ... 64, so it understates what a classifier can do
when it is given enough dots. Read this table before trusting a planner result: if a classifier is near chance even
at K = 64 it cannot validate (or, for T_c, usefully train) anything.
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import yaml

from common.world import build_world
from datasets.real_dataset import make_real_test
from datasets.shapes import ShapeSpec
from evaluation.classifiers import PatchMixer, ResNetSmall, load_classifier
from evaluation.experiments import evaluate_arm, save_rows
from module_b.planner_api import ScoreboardArm
from module_b.task_classifier import PlainCNN


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--experiment", default="configs/experiment.yaml")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    ap.add_argument("--n-test", type=int, default=200, help="use at most this many test images")
    ap.add_argument("--max-per-class", type=int, default=0)
    a = ap.parse_args()
    ex = yaml.safe_load(open(a.experiment)); out = Path(ex["paths"]["out"])
    cfg, sim, atlas = build_world(ex["paths"]["module_c_config"], ex["paths"]["atlas"])
    spec = ShapeSpec.from_sim(sim)
    models, classes = {}, None
    for name, f, make in [("E_main", out / "module_e" / "e_main_real.pt", lambda n: ResNetSmall(n)),
                          ("E_alt", out / "module_e" / "e_alt_real.pt", lambda n: PatchMixer(n, image_size=128)),
                          ("T_c", out / "module_b" / "task_classifier_real.pt", PlainCNN)]:
        if f.exists():
            m, c = load_classifier(f, make)
            if classes is not None and c != classes:
                sys.exit(f"{name} was trained on {c} but the others on {classes}; retrain them on the same data.")
            models[name], classes = m, c
        else:
            print(f"[skip] {f} not found")
    if not models:
        sys.exit("no trained classifiers found; run scripts/e_train_evaluators.py --data-dir ... first")
    cap = a.max_per_class or -(-a.n_test // len(classes)) * 2        # load only what will be used (the test set was loaded in full before)
    test = make_real_test(a.data_dir, spec, classes, max_per_class=cap, log=lambda *x, **k: None)
    arm = ScoreboardArm(sim.image_shape, ex["scoreboard"]["dot_sigma_px"], seed=0)
    rows = []
    for k in a.ks:
        r = evaluate_arm(arm, test, k, models, cfg, n_max=a.n_test)["row"]; rows.append(r)
    n = min(a.n_test, len(test["y"]))
    print(f"\nAccuracy on {n} held-out test images, K idealised dots on the true contour  (chance = {1 / len(classes):.3f}, "
          f"{len(classes)} classes)")
    print(f"{'K':>4}  " + "  ".join(f"{m:>7}" for m in models) + f"  {'blobs seen':>10}")
    for r in rows:
        print(f"{r['k']:>4}  " + "  ".join(f"{r['acc_' + m]:>7.3f}" for m in models) + f"  {r['blobs_mean']:>10.1f}")
    save_rows(rows, out / "module_e" / "real" / "evaluator_check")
    print(f"saved {out / 'module_e' / 'real' / 'evaluator_check.csv'}")


if __name__ == "__main__":
    main()
