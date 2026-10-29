#!/usr/bin/env python
"""Train the learned planners (B2a templates, B2b amortised) for one K per run.

    python scripts/b_train.py --k 16 --seed 0 --lam-t 0                       # synthetic shapes (original behaviour)
    python scripts/b_train.py --k 16 --seed 0 --data-dir data/my_images       # YOUR images (folder per class)
    python scripts/b_train.py --k 16 --seed 0 --data-dir data/my_images --label-conditioned

With --data-dir the images are segmented, cleaned, resized and placed on the simulator's canvas by
src/datasets/real_dataset.py. --lam-t defaults to 0 (contour-only planner). For --lam-t > 0 run
scripts/e_train_evaluators.py --data-dir <same folder> FIRST: it writes outputs/module_b/task_classifier_real.pt for
YOUR classes, which is checked against this run's classes. B2a templates (--templates) remain synthetic-only.
Model selection uses the validation CONTOUR loss only. Checkpoints go to outputs/module_b/.
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch, yaml

from common.world import build_world
from datasets.shapes import ShapeSpec, canonical_prototypes, make_dataset
from module_b.learned import AmortisedPlanner
from evaluation.classifiers import load_classifier
from module_b.task_classifier import PlainCNN, freeze
from module_b.training import TrainCfg, fit_templates, train_amortised, warm_start
from module_c.noise import NoiseSpec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="configs/experiment.yaml")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam-c", type=float); ap.add_argument("--lam-t", type=float); ap.add_argument("--lam-s", type=float)
    ap.add_argument("--robust", action="store_true")
    ap.add_argument("--label-conditioned", action="store_true")
    ap.add_argument("--templates", action="store_true", help="also fit B2a per-class templates (synthetic shapes only)")
    ap.add_argument("--tag", default="")
    # --- real-image options
    ap.add_argument("--data-dir", default=None, help="folder with one sub-folder of images per class (or train/ and val/)")
    ap.add_argument("--val-frac", type=float, default=0.2, help="validation share when there are no train/ val/ folders")
    ap.add_argument("--n-aug", type=int, default=5, help="augmented copies (shift/scale/rotation) per training image")
    ap.add_argument("--max-per-class", type=int, default=100,
                    help="use at most this many source images per class (0 = all). Segmentation time and memory grow with it")
    ap.add_argument("--epochs", type=int, default=None, help="override learned.epochs from the experiment config")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run from its per-epoch .ckpt (same settings required)")
    ap.add_argument("--init-from", default=None, metavar="PT",
                    help="warm start: load planner weights from a finished run's .pt; optimizer/schedule start fresh")
    ap.add_argument("--no-ckpt", action="store_true", help="do not write the per-epoch .ckpt resume file")
    ap.add_argument("--no-rotate", action="store_true", help="do not rotate images when augmenting")
    a = ap.parse_args()
    ex = yaml.safe_load(open(a.experiment))
    cfg, sim, atlas = build_world(ex["paths"]["module_c_config"], ex["paths"]["atlas"])
    sh = ex["shapes"]; L = ex["learned"]
    spec = ShapeSpec.from_sim(sim)

    if a.data_dir:
        from datasets.real_dataset import make_real_datasets
        if a.templates:
            sys.exit("--templates needs the synthetic class prototypes; run it without --data-dir.")
        if a.lam_t is None:
            a.lam_t = 0.0                                    # contour-only by default
        tr, va = make_real_datasets(a.data_dir, spec, val_frac=a.val_frac, n_aug_train=a.n_aug,
                                    seed=a.seed, rotate=not a.no_rotate,
                                    max_per_class=a.max_per_class or None,
                                    keep_contour=False)                      # b_train never reads the contour arrays
        classes = tr["classes"]
    else:
        classes = sh["classes"]
        tr = make_dataset(sh["n_train_per_class"], sh["seed_train"], spec, classes)
        va = make_dataset(sh["n_val_per_class"], sh["seed_val"], spec, classes)

    tc = TrainCfg(k=a.k, epochs=L["epochs"] if a.epochs is None else a.epochs, batch=L["batch"], lr=L["lr"], lr_final=L["lr_final"],
                  lam_c=L["lam_c"] if a.lam_c is None else a.lam_c, lam_t=L["lam_t"] if a.lam_t is None else a.lam_t,
                  lam_s=L["lam_s"] if a.lam_s is None else a.lam_s, task_warmup_epochs=L["task_warmup_epochs"],
                  tau=L["tau"], seed=a.seed, noise=NoiseSpec(**L["robust_noise"]) if a.robust else None)
    clf = None
    if tc.lam_t > 0:
        if a.data_dir:
            f = Path(ex["paths"]["out"]) / "module_b" / "task_classifier_real.pt"
            if not f.exists():
                sys.exit(f"{f} not found. Run: python scripts/e_train_evaluators.py --data-dir {a.data_dir}")
            clf, saved = load_classifier(f, PlainCNN)
            if saved != list(classes):
                sys.exit(f"T_c was trained on classes {saved} but this run has {list(classes)}. Retrain T_c on this data.")
        else:
            clf = PlainCNN(len(classes))
            clf.load_state_dict(torch.load(Path(ex["paths"]["out"]) / "module_b" / "task_classifier.pt"))
        clf = freeze(clf)
    out = Path(ex["paths"]["out"]) / "module_b"; out.mkdir(parents=True, exist_ok=True)
    name = f"b2b_k{a.k}_s{a.seed}{'_robust' if a.robust else ''}_c{tc.lam_c:g}_t{tc.lam_t:g}{'_real' if a.data_dir else ''}{a.tag}"
    planner = AmortisedPlanner(atlas, levels=sim.levels, n_labels=len(classes) if a.label_conditioned else 0)
    ckpt = None if a.no_ckpt else out / f"{name}.ckpt"
    if ckpt is not None and ckpt.exists() and not a.resume:
        print(f"WARNING: {ckpt} exists and will be overwritten (pass --resume to continue it, or --tag to keep both).")
    if a.init_from:
        if a.resume and ckpt is not None and ckpt.exists():
            print("--resume takes precedence: ignoring --init-from because a checkpoint to resume from exists.")
        else:
            warm_start(planner, a.init_from)
    hist = train_amortised(planner, sim, atlas, tr["x"], tr["y"], va["x"], tc, task_clf=clf, y_val=va["y"],
                           ckpt_path=ckpt, resume=a.resume)
    torch.save({"state": planner.state_dict(), "n_labels": planner.n_labels, "k": a.k, "classes": list(classes)},
               out / f"{name}.pt")
    (out / f"{name}.json").write_text(json.dumps({"train_cfg": {**tc.__dict__, "noise": str(tc.noise)}, "history": hist,
                                                  "classes": list(classes), "data_dir": a.data_dir,
                                                  "n_train": int(len(tr["x"])), "n_val": int(len(va["x"])),
                                                  "dynaphos_version": atlas.meta["dynaphos_version"], "atlas_meta": atlas.meta},
                                                 indent=2, default=str))
    if a.templates:
        px, _ = canonical_prototypes(spec, classes)
        tp = fit_templates(sim, atlas, px, tc)
        import numpy as np; np.save(out / f"b2a_templates_k{a.k}.npy", tp.currents)
    print("saved", out / f"{name}.pt")
    if ckpt is not None:
        print("resume file (safe to delete once you are happy):", ckpt)


if __name__ == "__main__":
    main()
