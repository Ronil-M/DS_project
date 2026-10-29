"""Acceptance tests for Module B (planners) against the frozen Module C.

Run:  pytest tests/test_module_b.py -q
"""
import numpy as np
import pytest
import torch

from common.imops import dilate
from datasets.shapes import FOUR, canonical_prototypes
from module_b.calibrate import currents_for_brightness
from module_b.constraints import CurrentParam
from module_b.contours import contour_map, contour_target
from module_b.heuristic import HeuristicPlanner, RandomPlanner
from module_b.learned import AmortisedPlanner
from module_b.losses import contour_loss
from module_b.planner_api import PlannerArm, ScoreboardArm
from module_b.training import TrainCfg, _loss_and_surrogate, fit_templates, train_amortised
from module_b.footprint import GaussianSampler


# ------------------------------------------------------------------ contour map
def test_gabor_contour_peaks_on_true_outline(shapes_test):
    x = torch.from_numpy(shapes_test["x"][:8])
    con = torch.from_numpy(shapes_test["contour"][:8])
    c = contour_map(x, sigma=2.0, lam=4.0)
    assert c.shape == x.shape and float(c.max()) == pytest.approx(1.0, abs=1e-5)
    near = dilate((con > 0.2).float(), 4)
    top = (c > 0.5).float()
    assert float((top * near).sum() / top.sum()) > 0.8          # strong responses sit on/near the real contour


# ------------------------------------------------------------------ constraints by construction
@pytest.mark.parametrize("levels", [None, 16])
def test_parametrisation_satisfies_constraints(batlas, levels):
    p = CurrentParam(batlas, levels=levels)
    B, N = 3, batlas.n
    torch.manual_seed(0)
    s, u = torch.randn(B, N), 3 * torch.randn(B, N)
    k = 12
    I = p.currents(p.ste_mask(s, k), u)
    rep = p.report(I, k=k)
    assert rep["ok"], rep
    assert int((I > 0).sum(1).max()) == min(k, p.n_usable)
    assert float(I.max()) <= p.i_max * (1 + 1e-6)


def test_unusable_electrodes_never_selected(batlas):
    p = CurrentParam(batlas)
    s = torch.zeros(1, batlas.n)
    s[0, ~torch.as_tensor(batlas.usable())] = 100.0             # try to force unusable ones to win
    m = p.hard_mask(s, 5)
    assert float((m * (~p.usable).float()).sum()) == 0.0 or p.n_usable == batlas.n


# ------------------------------------------------------------------ calibration
def test_brightness_calibration(bsim, batlas):
    cur = currents_for_brightness(bsim, batlas, 0.7)
    idx = np.flatnonzero(batlas.usable())[:4]
    c = np.zeros((4, bsim.n_electrodes), np.float32)
    for b, e in enumerate(idx):
        c[b, e] = cur[e]
    br = bsim.response(c)["brightness"].numpy()
    got = np.array([br[b, e] for b, e in enumerate(idx)])
    assert np.all(np.abs(got - 0.7) < 0.12) or np.all(cur[idx] >= 0.99 * bsim.i_max)


# ------------------------------------------------------------------ B1 heuristic
@pytest.fixture(scope="module")
def b1(bsim, batlas):
    return HeuristicPlanner(batlas, bsim)


def test_b1_constraints_and_render(b1, bsim, batlas, shapes_test):
    x = shapes_test["x"][:4]
    for k in (4, 8, 16):
        cur = b1.plan(x, k)
        assert cur.shape == (4, bsim.n_electrodes) and cur.dtype == np.float32
        rep = b1.param.report(torch.from_numpy(cur), k=k)
        assert rep["ok"], rep
        img = bsim.render(cur)
        assert img.shape == (4, 1, *bsim.image_shape) and float(img.max()) > 0.1


def test_b1_dots_trace_contour(b1, batlas, shapes_test):
    """Selected phosphene centres lie on or near the true outline (high precision)."""
    x, con = shapes_test["x"][:8], shapes_test["contour"][:8]
    cur = b1.plan(x, 12)
    near = dilate(torch.from_numpy(con) , 4).numpy()[:, 0]
    hits, tot = 0, 0
    for b in range(len(x)):
        e = np.flatnonzero(cur[b])
        hits += int((batlas.sample_nearest(near[b])[e] > 0.2).sum()); tot += len(e)
    assert tot > 0 and hits / tot > 0.7


def test_b1_nested_sets_with_thinning(b1, shapes_test):
    x = shapes_test["x"][:2]
    a, b = b1.plan(x, 8) > 0, b1.plan(x, 16) > 0
    assert np.all(a <= b)                                        # K=8 set is a subset of the K=16 set


def test_equivariance_of_b1(b1, bsim, shapes_test):
    from evaluation.metrics import equivariance_probe
    arm = PlannerArm(b1, bsim)
    r = equivariance_probe(arm, shapes_test["x"][:4], 12, shifts_px=((0, 8), (8, 0)))
    assert r["gain"] > 0.5, r


def test_b1_reports_fewer_than_k_when_contour_is_exhausted(b1, shapes_test):
    cur = b1.plan(shapes_test["x"][:1], 5000)
    assert (cur > 0).sum() <= b1.param.n_usable
    assert b1.last_n_selected[0] <= b1.param.n_usable


# ------------------------------------------------------------------ R0 / R1
def test_reference_arms(bsim, batlas, shapes_test):
    x = shapes_test["x"][:3]
    r1 = RandomPlanner(batlas)
    cur = r1.plan(x, 10)
    assert CurrentParam(batlas).report(torch.from_numpy(cur), k=10)["ok"]
    out = PlannerArm(r1, bsim).images(x, 10)
    assert out.image.shape == (3, 1, *bsim.image_shape)
    r0 = ScoreboardArm(bsim.image_shape).images(x, 10)
    assert r0.image.shape == (3, 1, *bsim.image_shape) and r0.currents is None and float(r0.image.max()) > 0.5


# ------------------------------------------------------------------ loss
def test_contour_loss_orders_good_bad_empty(bsim, b1, shapes_test):
    x = torch.from_numpy(shapes_test["x"][:4])
    T = contour_target(x)
    good = bsim.render(b1.plan(x, 16))
    rnd = bsim.render(RandomPlanner(b1.atlas).plan(x, 16))
    lg, _ = contour_loss(good, T); lr, _ = contour_loss(rnd, T); le, _ = contour_loss(torch.zeros_like(good), T)
    assert lg < lr and lg < le                                   # neither random dots nor a dark image beats B1


# ------------------------------------------------------------------ the selection-gradient surrogate
def test_selection_gradient_reaches_inactive_electrodes(bsim, batlas, shapes_test):
    """Plain straight-through top-k gives ZERO selection gradient to inactive electrodes (Dynaphos: dI=0 at I=0).
    The footprint surrogate must give non-zero, finite gradients to them, and real gradients to active amplitudes."""
    param = CurrentParam(batlas)
    sampler = GaussianSampler(batlas, tuple(batlas.meta["image_shape"]))
    x = torch.from_numpy(shapes_test["x"][:2])
    T = contour_target(x)
    torch.manual_seed(0)
    s = (0.1 * torch.randn(2, batlas.n)).requires_grad_(True)
    u = torch.zeros(2, batlas.n, requires_grad=True)
    k = 8
    _loss_and_surrogate(bsim, param, sampler, s, u, k, T, None, TrainCfg(k=k), None, 0.0)
    active = param.hard_mask(s, k).bool()
    assert torch.isfinite(s.grad).all() and torch.isfinite(u.grad).all()
    assert float(s.grad[~active].abs().sum()) > 0                # the point of the surrogate
    assert float(u.grad[active].abs().sum()) > 0                 # amplitudes trained through real Dynaphos
    assert float(u.grad[~active].abs().sum()) == 0               # inactive electrodes have no amplitude gradient


# ------------------------------------------------------------------ B2 smoke training
def test_amortised_training_smoke(bsim, batlas, shapes_test, tmp_path):
    torch.manual_seed(0)
    planner = AmortisedPlanner(batlas, n_labels=4)
    x, y = shapes_test["x"][:8], shapes_test["y"][:8]
    cfg = TrainCfg(k=8, epochs=2, batch=4, lam_t=0.0)
    hist = train_amortised(planner, bsim, batlas, x, y, shapes_test["x"][8:12], cfg, y_val=shapes_test["y"][8:12], log=lambda *_: None)
    assert len(hist) == 2 and all(np.isfinite(h["val_contour"]) for h in hist)
    cur = planner.plan(x[:4], 8, label=y[:4])
    assert CurrentParam(batlas).report(torch.from_numpy(cur), k=8)["ok"]
    assert bsim.render(cur).shape == (4, 1, *bsim.image_shape)


def test_template_planner_smoke(bsim, batlas, shapes_spec):
    px, _ = canonical_prototypes(shapes_spec, FOUR)
    tp = fit_templates(bsim, batlas, px, TrainCfg(k=8), steps=3, log=lambda *_: None)
    cur = tp.plan(px, 8, label=np.arange(4))
    assert cur.shape == (4, bsim.n_electrodes)
    assert CurrentParam(batlas).report(torch.from_numpy(cur), k=8)["ok"]
    with pytest.raises(ValueError):
        tp.plan(px, 8)                                           # templates need the class label


# ------------------------------------------------------------------ checkpoint / resume / warm start
def _quiet(*_):
    pass


def _fresh_planner(batlas, seed):
    torch.manual_seed(seed)
    return AmortisedPlanner(batlas)


def test_resume_after_crash_matches_uninterrupted_run(bsim, batlas, shapes_test, tmp_path):
    from module_b.training import train_amortised
    x, y, xv = shapes_test["x"][:8], shapes_test["y"][:8], shapes_test["x"][8:12]
    cfg = TrainCfg(k=8, epochs=3, batch=4, lam_t=0.0, seed=1)

    pa = _fresh_planner(batlas, 0)                                           # A: never interrupted
    ha = train_amortised(pa, bsim, batlas, x, y, xv, cfg, log=_quiet, ckpt_path=tmp_path / "a.ckpt")

    class Crash(Exception):
        pass

    def log_crash(msg):                                                       # B: dies right after epoch 1 finished
        if "epoch   1  train" in msg:
            raise Crash

    pb = _fresh_planner(batlas, 0)
    with pytest.raises(Crash):
        train_amortised(pb, bsim, batlas, x, y, xv, cfg, log=log_crash, ckpt_path=tmp_path / "b.ckpt")
    assert (tmp_path / "b.ckpt").exists()

    pc = _fresh_planner(batlas, 999)                                          # C: DIFFERENT random init, then resumed
    hc = train_amortised(pc, bsim, batlas, x, y, xv, cfg, log=_quiet, ckpt_path=tmp_path / "b.ckpt", resume=True)

    assert [h["epoch"] for h in hc] == [0, 1, 2]
    assert [h["val_contour"] for h in hc] == pytest.approx([h["val_contour"] for h in ha], rel=1e-3)
    for (n1, p1), (n2, p2) in zip(pa.state_dict().items(), pc.state_dict().items()):
        assert torch.allclose(p1.float(), p2.float(), atol=1e-4), n1


def test_resume_refuses_changed_settings(bsim, batlas, shapes_test, tmp_path):
    from module_b.training import train_amortised
    x, y, xv = shapes_test["x"][:8], shapes_test["y"][:8], shapes_test["x"][8:12]
    train_amortised(_fresh_planner(batlas, 0), bsim, batlas, x, y, xv, TrainCfg(k=8, epochs=1, batch=4, lam_t=0.0),
                    log=_quiet, ckpt_path=tmp_path / "c.ckpt")
    with pytest.raises(ValueError, match="Cannot resume"):
        train_amortised(_fresh_planner(batlas, 0), bsim, batlas, x, y, xv, TrainCfg(k=8, epochs=5, batch=4, lam_t=0.0),
                        log=_quiet, ckpt_path=tmp_path / "c.ckpt", resume=True)


def test_resume_without_checkpoint_starts_fresh(bsim, batlas, shapes_test, tmp_path):
    from module_b.training import train_amortised
    x, y, xv = shapes_test["x"][:8], shapes_test["y"][:8], shapes_test["x"][8:12]
    msgs = []
    h = train_amortised(_fresh_planner(batlas, 0), bsim, batlas, x, y, xv, TrainCfg(k=8, epochs=1, batch=4, lam_t=0.0),
                        log=msgs.append, ckpt_path=tmp_path / "none.ckpt", resume=True)
    assert len(h) == 1 and any("does not exist" in m for m in msgs)


def test_warm_start_loads_weights_and_checks_labels(batlas, tmp_path):
    from module_b.training import warm_start
    src = _fresh_planner(batlas, 3)
    torch.save({"state": src.state_dict(), "n_labels": 0, "k": 16}, tmp_path / "w.pt")
    dst = _fresh_planner(batlas, 4)
    warm_start(dst, tmp_path / "w.pt", log=_quiet)
    for p1, p2 in zip(src.state_dict().values(), dst.state_dict().values()):
        assert torch.equal(p1, p2)
    with pytest.raises(ValueError, match="n_labels"):
        warm_start(AmortisedPlanner(batlas, n_labels=4), tmp_path / "w.pt", log=_quiet)
