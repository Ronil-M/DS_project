#!/usr/bin/env python
"""Split one big image dataset into train / val / test, BY SOURCE IMAGE, so no photo (or augmented copy of it)
is ever in two sets.

    python scripts/split_dataset.py --src data/raw --dst data/split --val-frac 0.15 --test-frac 0.15

--src may be  <class>/*.png  (sub-folder per class, nested folders are fine)  or a flat folder with names like
circle_001.png.  --dst gets train/<class>/, val/<class>/, test/<class>/ (symlinks by default: no extra disk space).
The same --seed always gives the same split; split.json lists every file in every set.
Use --dst with  b_train.py / e_train_evaluators.py / e_run_experiments.py --data-dir <dst>.
"""
import argparse, json, os, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np

from datasets.real_dataset import _collect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--val-frac", type=float, default=0.15); ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["symlink", "copy", "hardlink"], default="symlink")
    ap.add_argument("--max-per-class", type=int, default=0, help="keep at most this many images per class (0 = all)")
    a = ap.parse_args()
    dst = Path(a.dst)
    if dst.exists() and any(dst.iterdir()):
        sys.exit(f"{dst} is not empty; choose a new --dst (nothing was changed).")
    files = _collect(Path(a.src))
    rng = np.random.default_rng(a.seed)
    manifest = {"train": {}, "val": {}, "test": {}}
    for c, paths in sorted(files.items()):
        paths = sorted(paths); n = len(paths)
        if a.max_per_class and n > a.max_per_class:
            paths = [paths[i] for i in sorted(rng.choice(n, a.max_per_class, replace=False))]; n = len(paths)
        if n < 6:
            print(f"WARNING: class '{c}' has only {n} images; each set needs at least 2 to be meaningful.")
        order = rng.permutation(n)
        n_te = max(1, int(round(a.test_frac * n))) if n >= 3 else 0
        n_va = max(1, int(round(a.val_frac * n))) if n >= 3 else 0
        parts = {"test": order[:n_te], "val": order[n_te:n_te + n_va], "train": order[n_te + n_va:]}
        for split, idx in parts.items():
            out = dst / split / c; out.mkdir(parents=True, exist_ok=True)
            manifest[split][c] = []
            for i in idx:
                src = paths[i].resolve(); name = paths[i].name; t = out / name; k = 1
                while t.exists():                                  # same file name from different sub-folders
                    t = out / f"{paths[i].stem}__{k}{paths[i].suffix}"; k += 1
                if a.mode == "symlink": os.symlink(src, t)
                elif a.mode == "hardlink": os.link(src, t)
                else: shutil.copy2(src, t)
                manifest[split][c].append(str(src))
    (dst / "split.json").write_text(json.dumps(manifest, indent=1))
    print(f"{'class':<20}{'train':>8}{'val':>8}{'test':>8}")
    for c in sorted(files):
        print(f"{c:<20}{len(manifest['train'].get(c, [])):>8}{len(manifest['val'].get(c, [])):>8}{len(manifest['test'].get(c, [])):>8}")
    print(f"written to {dst} (mode={a.mode}). Next: --data-dir {dst}")


if __name__ == "__main__":
    main()
