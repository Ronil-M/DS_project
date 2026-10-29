"""Experiment drivers Exp 1-4 (plan 7.3). Exp 5 (human check) is not software and is not implemented.

Terminology
-----------
arm        object with .name and .images(x, k, label, noise, generator) -> ArmOutput   (module_b.planner_api)
test       dict from datasets.shapes.make_dataset: x, contour, y, classes
evaluators dict name -> trained classifier (E_main, E_alt). The training classifier T_c is passed
           separately (only Exp 4 reports it) so that no number can silently use the wrong network.
Labels given to label-conditioned planners are the ground truth by default (`labels=None`); pass Module A's
predictions to run the real pipeline.

Every result row is a flat dict so it can go straight to CSV / JSON (`save_rows`).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from module_c.noise import NoiseSpec

from . import metrics as M


def evaluate_arm(arm, test: dict, k: int, evaluators: Dict[str, torch.nn.Module], cfg: dict, *,
                 noise: Optional[NoiseSpec] = None, labels=None, batch: int = 4, seed: int = 0,
                 tol_px: float = 2.0, n_max: Optional[int] = None, extra_models: Optional[Dict[str, torch.nn.Module]] = None) -> dict:
    """Render the whole test set through one arm at one K and compute every metric. Returns a row plus raw arrays."""
    x, y, con = test["x"], test["y"], test["contour"]
    n = len(x) if n_max is None else min(n_max, len(x))
    lab = y if labels is None else labels
    gen = torch.Generator().manual_seed(seed)
    imgs, curs = [], []
    for i in range(0, n, batch):
        j = min(i + batch, n)
        out = arm.images(x[i:j], k, label=lab[i:j], noise=noise, generator=gen)
        imgs.append(out.image.cpu())
        if out.currents is not None:
            curs.append(out.currents)
    P = torch.cat(imgs)
    row = {"arm": arm.name, "k": k, "n_images": n, "noise": "none" if noise is None or noise.is_null() else str(noise)}
    correct = {}
    for name, model in {**evaluators, **(extra_models or {})}.items():
        correct[name] = M.top1(model, P, y[:n])
        row[f"acc_{name}"] = float(correct[name].mean())
        row[f"acc_{name}_lo"], row[f"acc_{name}_hi"] = M.bootstrap_ci(correct[name])[1:]
    parts = [M.contour_fidelity(P[i:min(i + 16, n)], torch.from_numpy(con[i:min(i + 16, n)]), tol_px) for i in range(0, n, 16)]   # chunked: dilate() makes several full-size copies
    fid = {key: torch.cat([q[key] for q in parts]) for key in parts[0]}
    for key, v in fid.items():
        row[f"fid_{key}"] = float(v.mean())
    row["blobs_mean"] = float(M.visible_blobs(P).mean())
    if curs:
        C = np.concatenate(curs)
        row["active_mean"] = float((C > 0).sum(1).mean())
        row.update({f"load_{k_}": v for k_, v in M.current_load(C, cfg).items()})
    else:
        row["active_mean"] = float("nan")
    return {"row": row, "correct": correct, "images": P}


def exp1_accuracy_vs_k(arm_factories: Dict[str, Callable[[int], object]], test: dict, ks: Sequence[int],
                       evaluators: dict, cfg: dict, seeds: Sequence[int] = (0, 1, 2), **kw) -> List[dict]:
    """Accuracy vs K for every arm. arm_factories[name](seed) -> arm (seed varies planner/layout randomness)."""
    rows = []
    for name, make in arm_factories.items():
        for seed in seeds:
            arm = make(seed)
            for k in ks:
                r = evaluate_arm(arm, test, k, evaluators, cfg, seed=seed, **kw)["row"]
                r["seed"] = seed
                rows.append(r)
    return rows


def k_table(rows: List[dict], evaluator: str = "E_main", levels=(0.5, 0.9)) -> List[dict]:
    """Table 1: K needed for 50 % / 90 % accuracy per arm (mean curve over seeds)."""
    out = []
    for arm in sorted({r["arm"] for r in rows}):
        ks = sorted({r["k"] for r in rows if r["arm"] == arm})
        acc = [np.mean([r[f"acc_{evaluator}"] for r in rows if r["arm"] == arm and r["k"] == k]) for k in ks]
        out.append({"arm": arm, **{f"K@{int(l * 100)}%": M.k_at_accuracy(ks, acc, l) for l in levels}})
    return out


def exp2_benchmark_plot(rows: List[dict], path: str | Path, evaluator: str = "E_main", x_key: str = "k") -> None:
    """Figure 1 / Exp 2: accuracy vs K (or vs visible blobs, x_key='blobs_mean') with the 29-65 band and the ~5 limit.

    Caveat (plan section 2): human free naming of complex objects vs a classifier on simple shapes is context,
    not a like-for-like comparison. The band is drawn for orientation only.
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for arm in sorted({r["arm"] for r in rows}):
        xs = sorted({r[x_key] if x_key == "k" else round(r[x_key], 1) for r in rows if r["arm"] == arm})
        ks = sorted({r["k"] for r in rows if r["arm"] == arm})
        ys = [np.mean([r[f"acc_{evaluator}"] for r in rows if r["arm"] == arm and r["k"] == k]) for k in ks]
        xv = ks if x_key == "k" else [np.mean([r[x_key] for r in rows if r["arm"] == arm and r["k"] == k]) for k in ks]
        ax.plot(xv, ys, marker="o", label=arm)
    ax.axvspan(29, 65, color="grey", alpha=0.15, label="29-65 patches (Scialom et al.)")
    ax.axvline(5, color="k", ls=":", lw=1, label="~5 simultaneous phosphenes (human)")
    ax.set_xlabel("active electrodes K" if x_key == "k" else "visible phosphenes (blobs)")
    ax.set_ylabel(f"{evaluator} top-1 accuracy"); ax.set_ylim(0, 1.02); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


def exp3_noise(arms: Dict[str, object], test: dict, ks: Sequence[int], grid: List[dict], evaluators: dict, cfg: dict,
               jitter_sigmas_deg: Sequence[float] = (), sim=None, **kw) -> List[dict]:
    """Robustness: accuracy/fidelity under each NoiseSpec in `grid`, plus phosphene position jitter if `sim` is given.

    Compare clean-trained against robust-trained planners by passing both as separate arms.
    """
    from .noise import jittered_simulator
    rows = []
    for name, arm in arms.items():
        for k in ks:
            for g in grid:
                r = evaluate_arm(arm, test, k, evaluators, cfg, noise=g["spec"], **kw)["row"]
                r["condition"], r["arm"] = g["label"], name
                rows.append(r)
            if sim is not None and hasattr(arm, "with_sim"):
                for s in jitter_sigmas_deg:
                    r = evaluate_arm(arm.with_sim(jittered_simulator(sim, s)), test, k, evaluators, cfg, **kw)["row"]
                    r["condition"], r["arm"] = f"pos_jitter_deg={s:g}", name
                    rows.append(r)
    return rows


def exp4_trap(arms: Dict[str, object], test: dict, k: int, evaluators: dict, t_c: torch.nn.Module, cfg: dict, **kw) -> List[dict]:
    """Trap ablation: for each arm (e.g. 'task_only', 'contour_only', 'both') report T_c accuracy, E accuracy and
    fidelity side by side. H3 predicts: task_only has high acc_T_c, low acc_E and low fidelity."""
    rows = []
    for name, arm in arms.items():
        r = evaluate_arm(arm, test, k, evaluators, cfg, extra_models={"T_c": t_c}, **kw)["row"]
        r["arm"] = name
        rows.append(r)
    return rows


def save_rows(rows: List[dict], path_stem: str | Path) -> None:
    stem = Path(path_stem); stem.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(stem.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    stem.with_suffix(".json").write_text(json.dumps(rows, indent=2, default=float))
