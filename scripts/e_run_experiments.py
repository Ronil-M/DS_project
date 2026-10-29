#!/usr/bin/env python
"""Experiments 1-4 (plan 7.3) from trained checkpoints. Exp 5 (human check) is out of scope for software.

    python scripts/e_run_experiments.py --exp 1 2          # accuracy vs K + benchmark figure
    python scripts/e_run_experiments.py --exp 3 4
    python scripts/e_run_experiments.py --data-dir data/split --exp 1 2 3 4     # YOUR images: held-out data-dir/test/

Expects: outputs/module_e/e_main.pt, e_alt.pt, outputs/module_b/task_classifier.pt and, for the learned arm,
one checkpoint per K named b2b_k{K}_s{seed}_robust?_c{lam_c}_t{lam_t}.pt (scripts/b_train.py).
Missing learned checkpoints are skipped with a warning, so R0 / R1 / B1 curves always run.
Analysis plan: write it down BEFORE running Exp 1 and commit it (plan 7.4).
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch, yaml

from common.world import build_world
from datasets.shapes import ShapeSpec, make_dataset
from evaluation.classifiers import PatchMixer, ResNetSmall, load_classifier
from evaluation.experiments import exp1_accuracy_vs_k, exp2_benchmark_plot, exp3_noise, exp4_trap, k_table, save_rows
from evaluation.noise import noise_grid
from module_b.heuristic import HeuristicPlanner, RandomPlanner
from module_b.learned import AmortisedPlanner
from module_b.planner_api import PlannerArm, ScoreboardArm
from module_b.task_classifier import PlainCNN, freeze


class KSwitchPlanner:
    """Wraps one trained B2b per K so a single arm can be swept over K (the plan trains one model per K)."""
    def __init__(self, planners, name): self.p, self.name = planners, name
    def plan(self, x, k, label=None):
        ks = sorted(self.p); kk = min(ks, key=lambda q: abs(q - k))
        return self.p[kk].plan(x, k, label)


def load_b2b(out, atlas, sim, seed, ks, robust, lam_c, lam_t, name, suffix=""):
    ps = {}
    for k in ks:
        f = out / "module_b" / f"b2b_k{k}_s{seed}{'_robust' if robust else ''}_c{lam_c:g}_t{lam_t:g}{suffix}.pt"
        if f.exists():
            ck = torch.load(f); p = AmortisedPlanner(atlas, levels=sim.levels, n_labels=ck["n_labels"]); p.load_state_dict(ck["state"]); ps[k] = p.eval()
    return KSwitchPlanner(ps, name) if ps else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--experiment", default="configs/experiment.yaml")
    ap.add_argument("--exp", type=int, nargs="+", default=[1, 2]); ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--data-dir", default=None, help="YOUR images (train/ val/ test/ from scripts/split_dataset.py)")
    ap.add_argument("--n-aug-test", type=int, default=1); ap.add_argument("--batch", type=int, default=4, help="images rendered at once; Dynaphos memory ~ batch x electrodes x 512^2 floats (lower it if memory is tight)"); ap.add_argument("--max-per-class", type=int, default=0)
    ap.add_argument("--allow-val-as-test", action="store_true",
                    help="if data-dir has no test/ folder, evaluate on val/ (NOT held out: val picked the checkpoints)")
    a = ap.parse_args()
    ex = yaml.safe_load(open(a.experiment)); sh = ex["shapes"]; out = Path(ex["paths"]["out"])
    cfg, sim, atlas = build_world(ex["paths"]["module_c_config"], ex["paths"]["atlas"])
    spec = ShapeSpec.from_sim(sim)
    real = a.data_dir is not None; sfx = "_real" if real else ""
    if real:
        from datasets.real_dataset import make_real_datasets, make_real_test
        e_main, classes = load_classifier(out / "module_e" / "e_main_real.pt", lambda k: ResNetSmall(k))
        n = len(classes)
        try:
            cap = a.max_per_class or (-(-a.n_test // n) * 2 if a.n_test else None)   # with --n-test, do not load more than needed
            test = make_real_test(a.data_dir, spec, classes, n_aug=a.n_aug_test, max_per_class=cap)
        except FileNotFoundError:
            if not a.allow_val_as_test:
                sys.exit(f"{a.data_dir}/test not found. Run scripts/split_dataset.py, or pass --allow-val-as-test "
                         "(results are then optimistic: val was used to choose checkpoints).")
            print("WARNING: evaluating on the VALIDATION images, which were used to pick checkpoints. Not a held-out test.")
            test = make_real_datasets(a.data_dir, spec, n_aug_train=1, n_aug_val=1)[1]
            if test["classes"] != classes:
                sys.exit(f"validation classes {test['classes']} differ from the trained classes {classes}")
    else:
        n = len(sh["classes"])
        test = make_dataset(sh["n_test_per_class"], sh["seed_test"], spec, sh["classes"])
        e_main = ResNetSmall(n); e_main.load_state_dict(torch.load(out / "module_e" / "e_main.pt"))
    evaluators = {"E_main": e_main.eval()}
    alt = out / "module_e" / f"e_alt{sfx}.pt"
    if alt.exists():
        if real:
            m, _ = load_classifier(alt, lambda k: PatchMixer(k, image_size=128))
        else:
            m = PatchMixer(n, image_size=128); m.load_state_dict(torch.load(alt))
        evaluators["E_alt"] = m.eval()
    ks, seeds = ex["sweep"]["k"], ex["seeds"]; H = ex["heuristic"]
    kw = {"n_max": a.n_test, "batch": a.batch}
    factories = {
        "R0_scoreboard": lambda s: ScoreboardArm(sim.image_shape, ex["scoreboard"]["dot_sigma_px"], seed=s),
        "R1_random": lambda s: PlannerArm(RandomPlanner(atlas, seed=s), sim),
        "B1_heuristic": lambda s: PlannerArm(HeuristicPlanner(atlas, sim, method=H["method"], pool=H["pool"], pool_excl_deg=H["pool_excl_deg"],
                                                              excl_deg=H["excl_deg"], floor_frac=H["floor_frac"], target_brightness=H["target_brightness"]), sim),
    }
    L = ex["learned"]
    # learned arm: one checkpoint per K and seed; skipped (with a warning) if none exist for that seed
    for s in seeds:
        if load_b2b(out, atlas, sim, s, ks, False, L["lam_c"], 0.0, "B2b_amortised", sfx) is None:
            print(f"[warn] no B2b checkpoints for seed {s}: that seed is skipped for B2b")
    b2b = {s: load_b2b(out, atlas, sim, s, ks, False, L["lam_c"], 0.0, "B2b_amortised", sfx) for s in seeds}
    b2b_seeds = [s for s in seeds if b2b[s] is not None]
    res = out / "module_e" / ("real" if real else "")
    if 1 in a.exp or 2 in a.exp:
        rows = exp1_accuracy_vs_k(factories, test, ks, evaluators, cfg, seeds=seeds, **kw)
        if b2b_seeds:
            rows += exp1_accuracy_vs_k({"B2b_amortised": lambda s: PlannerArm(b2b[s], sim)}, test, ks, evaluators, cfg,
                                       seeds=b2b_seeds, **kw)
        save_rows(rows, res / "exp1"); (res / "table1.json").write_text(json.dumps(k_table(rows), indent=2))
        exp2_benchmark_plot(rows, res / "fig1_accuracy_vs_k.png")
        exp2_benchmark_plot(rows, res / "fig1b_accuracy_vs_blobs.png", x_key="blobs_mean")
    if 3 in a.exp:
        arms = {"B1_heuristic": factories["B1_heuristic"](0), "R1_random": factories["R1_random"](0)}
        for tag, robust in (("clean", False), ("robust", True)):
            pl = load_b2b(out, atlas, sim, 0, [8, 16, 32], robust, L["lam_c"], 0.0, f"B2b_{tag}", sfx)
            if pl: arms[f"B2b_{tag}"] = PlannerArm(pl, sim)
        grid = noise_grid(ex["sweep"]["p_drop"], ex["sweep"]["sigma_thr"])
        save_rows(exp3_noise(arms, test, [8, 16, 32], grid, evaluators, cfg, jitter_sigmas_deg=(0.0, 0.1, 0.25), sim=sim, **kw), res / "exp3")
    if 4 in a.exp:
        if real:
            tc, tc_classes = load_classifier(out / "module_b" / "task_classifier_real.pt", PlainCNN)
            if tc_classes != classes:
                sys.exit(f"T_c classes {tc_classes} differ from the evaluator classes {classes}")
        else:
            tc = PlainCNN(n); tc.load_state_dict(torch.load(out / "module_b" / "task_classifier.pt"))
        tc = freeze(tc)
        arms = {}
        for tag, (lc, lt) in {"contour_only": (L["lam_c"], 0.0), "task_only": (0.0, 1.0), "both": (L["lam_c"], L["lam_t"])}.items():
            pl = load_b2b(out, atlas, sim, 0, [16], False, lc, lt, tag, sfx)
            if pl: arms[tag] = PlannerArm(pl, sim)
        save_rows(exp4_trap(arms, test, 16, evaluators, tc, cfg, **kw), res / "exp4")


if __name__ == "__main__":
    main()
