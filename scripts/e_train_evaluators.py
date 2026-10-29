#!/usr/bin/env python
"""Train the independent evaluators E_main (ResNetSmall) and E_alt (PatchMixer), plus the training classifier T_c.

    python scripts/e_train_evaluators.py                               # synthetic shapes (original behaviour)
    python scripts/e_train_evaluators.py --data-dir data/split         # YOUR images (train/ val/ [test/] folders)

All three train on common.sparse_data images (scoreboard dots on true contours + Dynaphos renders of random
contour-adjacent electrode subsets), with different seeds, and never see planner outputs. The held-out
accuracy on dot images is printed and saved: report it before any planner result (plan 7.1).

With --data-dir the three networks are trained on YOUR classes (class = sub-folder name) and written with a
"_real" suffix, together with the class list, so they can never be confused with the synthetic-shape networks:
    outputs/module_e/e_main_real.pt   outputs/module_e/e_alt_real.pt   outputs/module_b/task_classifier_real.pt
    outputs/module_e/evaluator_report_real.json
Only train/ and val/ are read here. test/ is reserved for scripts/e_run_experiments.py.
"""
import argparse, gc, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, torch, yaml

from common.sparse_data import make_sparse_dataset
from common.world import build_world
from datasets.shapes import ShapeSpec, make_dataset
from evaluation.classifiers import PatchMixer, ResNetSmall, accuracy, save_classifier, train_evaluator
from module_b.task_classifier import PlainCNN


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--experiment", default="configs/experiment.yaml")
    ap.add_argument("--data-dir", default=None, help="folder with train/ and val/ (from scripts/split_dataset.py), "
                                                     "or one sub-folder per class")
    ap.add_argument("--n-aug", type=int, default=5, help="augmented copies (shift/scale/rotation) per training image")
    ap.add_argument("--max-per-class", type=int, default=100, help="use at most this many source images per class (0 = all). Memory grows with it: ~2 MB per image at 512 px")
    ap.add_argument("--val-frac", type=float, default=0.2, help="only used if data-dir has no train/ and val/ folders")
    ap.add_argument("--no-rotate", action="store_true")
    ap.add_argument("--epochs", type=int, default=None, help="override evaluators.epochs")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--only", nargs="+", choices=["E_main", "E_alt", "T_c"], default=None,
                    help="retrain just these networks (the others and their report entries are kept)")
    ap.add_argument("--lr", type=float, default=2e-3, help="peak learning rate of the one-cycle schedule (default 2e-3; "
                                                           "try 5e-4 if the validation accuracy jumps around between epochs)")
    a = ap.parse_args()
    ex = yaml.safe_load(open(a.experiment)); E = ex["evaluators"]; sh = ex["shapes"]
    cfg, sim, atlas = build_world(ex["paths"]["module_c_config"], ex["paths"]["atlas"])
    spec = ShapeSpec.from_sim(sim)
    out = Path(ex["paths"]["out"]); (out / "module_e").mkdir(parents=True, exist_ok=True); (out / "module_b").mkdir(parents=True, exist_ok=True)
    epochs = E["epochs"] if a.epochs is None else a.epochs
    real = a.data_dir is not None
    sfx = "_real" if real else ""
    if real:
        from datasets.real_dataset import make_real_datasets
        tr_real, va_real = make_real_datasets(a.data_dir, spec, val_frac=a.val_frac, n_aug_train=a.n_aug,
                                              seed=a.split_seed, rotate=not a.no_rotate,
                                              max_per_class=a.max_per_class or None,
                                              keep_x=False)               # only contours + labels are used here: halves memory
        classes = tr_real["classes"]
    else:
        classes = sh["classes"]
    n = len(classes); report = {}
    rep_file = out / "module_e" / f"evaluator_report{sfx}.json"
    if a.only and rep_file.exists():
        report = json.loads(rep_file.read_text())                      # keep the entries of networks that are not retrained
        if any(v.get("classes") != list(classes) for v in report.values()):
            sys.exit("the saved report was made for different classes; retrain all three (omit --only).")
    print(f"classes ({n}): {classes}")
    for name, seed, make, dest in [("E_main", E["seed_main"], lambda: ResNetSmall(n), out / "module_e" / f"e_main{sfx}.pt"),
                                   ("E_alt", E["seed_alt"], lambda: PatchMixer(n, image_size=128), out / "module_e" / f"e_alt{sfx}.pt"),
                                   ("T_c", E["seed_task"], lambda: PlainCNN(n), out / "module_b" / f"task_classifier{sfx}.pt")]:
        if a.only and name not in a.only:
            continue
        if real:
            tr, va = tr_real, va_real                                    # same images; different dots / seeds per network
        else:
            tr = make_dataset(E["n_train_per_class"], 5000 + seed, spec, classes)   # shape seeds differ per network
            va = make_dataset(E["n_val_per_class"], 6000 + seed, spec, classes)
        rng = np.random.default_rng(seed)
        xt, yt = make_sparse_dataset(tr, rng, sim=sim, atlas=atlas, frac_dynaphos=E["frac_dynaphos"])
        xv, yv = make_sparse_dataset(va, rng, sim=sim, atlas=atlas, frac_dynaphos=E["frac_dynaphos"])
        model = train_evaluator(make(), xt, yt, xv, yv, epochs=epochs, seed=seed, lr=a.lr)
        if real:
            save_classifier(dest, model, classes)
        else:
            torch.save(model.state_dict(), dest)
        report[name] = {"heldout_dot_accuracy": accuracy(model, xv, yv), "seed": seed, "arch": type(model).__name__,
                        "n_train_images": int(len(xt)), "n_val_images": int(len(xv)), "classes": list(classes)}
        print(name, report[name])
        del xt, yt, xv, yv, model; gc.collect()                         # free this network's dot images before building the next set
    rep_file.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
