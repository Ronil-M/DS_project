"""Acceptance tests for Module E (independent evaluation) against the frozen Module C.

Run:  pytest tests/test_module_e.py -q
"""
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from common.dots import place_dots, render_dots
from common.sparse_data import make_sparse_dataset
from datasets.shapes import make_dataset
from evaluation import metrics as M
from evaluation.classifiers import PatchMixer, ResNetSmall, accuracy, train_evaluator
from evaluation.experiments import evaluate_arm, exp1_accuracy_vs_k, k_table, save_rows
from evaluation.noise import jittered_simulator, noise_grid
from module_b.heuristic import HeuristicPlanner, RandomPlanner
from module_b.planner_api import PlannerArm, ScoreboardArm
from module_c.noise import NoiseSpec

SRC = Path(__file__).resolve().parents[1] / "src"


# ------------------------------------------------------------------ independence (plan 7.1)
def test_evaluation_code_does_not_import_planners():
    for f in list((SRC / "evaluation").glob("*.py")) + list((SRC / "common").glob("*.py")):
        for line in f.read_text().splitlines():
            if re.match(r"\s*(from|import)\s", line):
                assert "module_b" not in line, f"{f.name} imports Module B: {line.strip()}"


def test_evaluator_architectures_differ_from_training_classifier():
    from module_b.task_classifier import PlainCNN
    assert type(PlainCNN(4)) not in (ResNetSmall, PatchMixer)
    assert ResNetSmall is not PatchMixer


# ------------------------------------------------------------------ statistics and metrics
def test_bootstrap_and_mcnemar():
    c = np.r_[np.ones(80), np.zeros(20)].astype(bool)
    m, lo, hi = M.bootstrap_ci(c)
    assert lo < m < hi and m == pytest.approx(0.8)
    r = M.mcnemar(c, c)
    assert r["p_value"] == 1.0
    assert M.mcnemar(np.ones(30, bool), np.zeros(30, bool))["p_value"] < 1e-6


def test_k_at_accuracy():
    assert M.k_at_accuracy([4, 8, 16], [0.2, 0.6, 1.0], 0.5) == pytest.approx(7.0)
    assert M.k_at_accuracy([4, 8], [0.2, 0.3], 0.9) is None


def test_contour_fidelity_good_vs_bad(shapes_test):
    con = torch.from_numpy(shapes_test["contour"][:4])
    H, W = con.shape[-2:]
    good = np.zeros((4, 1, H, W), np.float32)
    far = np.zeros_like(good)
    for b in range(4):
        pts = place_dots(shapes_test["contour"][b, 0], 16)
        good[b, 0] = render_dots(pts, (H, W), 1.2)
        far[b, 0] = render_dots(np.array([[3.0, 3.0], [4.0, H - 4.0]]), (H, W), 1.2)
    fg, ff = M.contour_fidelity(torch.from_numpy(good), con), M.contour_fidelity(torch.from_numpy(far), con)
    assert float(fg["precision"].mean()) > 0.9 and float(fg["recall"].mean()) > 0.5
    assert float(ff["precision"].mean()) < 0.2


def test_visible_blobs_counts_merging():
    img = np.zeros((1, 1, 64, 64), np.float32)
    img[0, 0] = render_dots(np.array([[10, 10], [10, 30], [40, 20]], np.float32), (64, 64), 1.5)
    assert M.visible_blobs(torch.from_numpy(img))[0] == 3
    img[0, 0] = render_dots(np.array([[10, 10], [10, 12]], np.float32), (64, 64), 2.0)     # merge
    assert M.visible_blobs(torch.from_numpy(img))[0] == 1


def test_equivariance_probe_separates_object_following_from_random(bsim, batlas, shapes_test):
    x = shapes_test["x"][:6]
    follows = M.equivariance_probe(ScoreboardArm(bsim.image_shape), x, 12, shifts_px=((0, 8), (8, 0)))
    random = M.equivariance_probe(PlannerArm(RandomPlanner(batlas, seed=3), bsim), x, 12, shifts_px=((0, 8), (8, 0)))
    assert follows["gain"] > 0.7
    assert abs(random["gain"]) < follows["gain"] - 0.3


def test_current_load(bcfg):
    c = np.zeros((2, 10)); c[0, :3] = 50e-6; c[1, 0] = 100e-6
    r = M.current_load(c, bcfg)
    assert r["max_uA"] == pytest.approx(100) and r["mean_active_uA"] == pytest.approx(62.5)
    assert r["mean_charge_per_pulse_nC"] == pytest.approx(((150e-6 + 100e-6) / 2) * 170e-6 * 1e9)


# ------------------------------------------------------------------ noise tooling
def test_noise_grid_and_jittered_simulator(bsim, batlas):
    g = noise_grid()
    assert any(x["spec"].p_drop == 0.3 for x in g) and any(x["spec"].sigma_thr == 0.5 for x in g)
    js = jittered_simulator(bsim, 0.2, seed=1)
    assert js.n_electrodes == bsim.n_electrodes
    assert torch.equal(js.base_thr, bsim.base_thr)
    cur = batlas.random_currents(8, 2, np.random.default_rng(0))
    assert not torch.allclose(js.render(cur), bsim.render(cur))
    assert torch.allclose(bsim.render(cur), bsim.render(cur))                # nominal simulator untouched


# ------------------------------------------------------------------ evaluator training data and models
def test_sparse_dataset_has_no_planner_input(bsim, batlas, shapes_spec):
    sh = make_dataset(2, seed=5, spec=shapes_spec)
    imgs, y = make_sparse_dataset(sh, np.random.default_rng(0), sim=bsim, atlas=batlas, frac_dynaphos=0.5)
    assert imgs.shape == sh["x"].shape and np.array_equal(y, sh["y"]) and imgs.max() <= 1.0 and imgs.sum() > 0


@pytest.mark.parametrize("cls", [ResNetSmall, PatchMixer])
def test_evaluators_forward_and_learn(cls, shapes_spec, bsim):
    tr = make_dataset(60, seed=11, spec=shapes_spec)
    va = make_dataset(15, seed=12, spec=shapes_spec)
    xt, yt = make_sparse_dataset(tr, np.random.default_rng(1), k_range=(24, 64))
    xv, yv = make_sparse_dataset(va, np.random.default_rng(2), k_range=(24, 64))
    model = cls(4) if cls is ResNetSmall else cls(4, image_size=64)
    assert model(torch.from_numpy(xt[:2])).shape == (2, 4)
    train_evaluator(model, xt, yt, xv, yv, epochs=12, batch=16, log=lambda *_: None)
    assert accuracy(model, xv, yv) > 0.4                                     # chance = 0.25


# ------------------------------------------------------------------ end-to-end drivers
@pytest.fixture(scope="module")
def tiny_eval(shapes_spec):
    ev = ResNetSmall(4, width=8)                                             # untrained: only plumbing is tested
    return {"E_main": ev.eval()}


def test_evaluate_arm_end_to_end(bsim, batlas, bcfg, shapes_test, tiny_eval):
    arm = PlannerArm(HeuristicPlanner(batlas, bsim), bsim)
    res = evaluate_arm(arm, shapes_test, 8, tiny_eval, bcfg, n_max=8, batch=4)
    row = res["row"]
    for key in ("acc_E_main", "fid_precision", "fid_recall", "fid_f", "blobs_mean", "active_mean", "load_max_uA"):
        assert key in row and np.isfinite(row[key])
    assert row["active_mean"] <= 8 and res["images"].shape[0] == 8
    noisy = evaluate_arm(arm, shapes_test, 8, tiny_eval, bcfg, n_max=8, batch=4, noise=NoiseSpec(p_drop=1.0))["row"]
    assert noisy["fid_recall"] == 0.0 or noisy["blobs_mean"] == 0.0           # everything dropped -> dark image


def test_exp1_ktable_and_save(bsim, batlas, bcfg, shapes_test, tiny_eval, tmp_path):
    arms = {"B1": lambda s: PlannerArm(HeuristicPlanner(batlas, bsim), bsim),
            "R1": lambda s: PlannerArm(RandomPlanner(batlas, seed=s), bsim),
            "R0": lambda s: ScoreboardArm(bsim.image_shape, seed=s)}
    rows = exp1_accuracy_vs_k(arms, shapes_test, [4, 8], tiny_eval, bcfg, seeds=(0,), n_max=8, batch=4)
    assert len(rows) == 6 and {r["arm"] for r in rows} >= {"B1_heuristic", "R1_random", "R0_scoreboard"}
    assert len(k_table(rows)) == 3
    save_rows(rows, tmp_path / "exp1")
    assert (tmp_path / "exp1.csv").exists() and (tmp_path / "exp1.json").exists()
