"""Acceptance tests for Module C (Dynaphos wrapper).

Run:  pytest tests/test_module_c.py -q

A reduced configuration (8 deg field, 128 px, 150 electrodes) keeps the suite fast on CPU
while giving phosphenes of 0.5-1.7 px sigma so image-based assertions are meaningful.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from module_c.atlas import Atlas, build_atlas
from module_c.constraints import constraint_report, project_currents, quantize_ste, topk_mask
from module_c.layout import build_layout, load_config
from module_c.noise import NoiseSpec
from module_c.simulator import ProstheticSimulator

CFG_PATH = Path(__file__).resolve().parents[1] / "configs" / "module_c.yaml"


@pytest.fixture(scope="module")
def cfg():
    c = load_config(CFG_PATH)
    c["run"]["resolution"] = [128, 128]
    c["run"]["view_angle"] = 8
    c["run"]["batch_size"] = 4
    c["run"]["gpu"] = None
    c["layout"]["n_electrodes"] = 150
    c["render"]["n_frames"] = 8
    return c


@pytest.fixture(scope="module")
def sim(cfg):
    return ProstheticSimulator(cfg)


@pytest.fixture(scope="module")
def atlas(sim):
    return build_atlas(sim)


def _single_electrode_currents(n, idxs, amps):
    c = np.zeros((len(idxs), n), np.float32)
    for b, (i, a) in enumerate(zip(idxs, amps)):
        c[b, i] = a
    return c


# ------------------------------------------------------------------ C1/C2: construction & layout
def test_construction_and_shapes(sim, atlas):
    assert sim.n_electrodes == atlas.n > 0
    out = sim.render(np.zeros((2, sim.n_electrodes), np.float32))
    assert out.shape == (2, 1, *sim.image_shape)
    assert sim.render(np.zeros(sim.n_electrodes, np.float32)).shape == (1, *sim.image_shape)


def test_layout_modes(cfg):
    c = dict(cfg)
    c["layout"] = dict(cfg["layout"], mode="probabilistic", n_electrodes=60)
    assert len(build_layout(c)) == 60
    c["layout"] = dict(cfg["layout"], mode="cortical_grid", n_electrodes=60)
    m = build_layout(c)
    assert 0 < len(m) <= 60
    x, y = m.cartesian
    assert np.all(np.abs(x) < cfg["run"]["view_angle"] / 2) and np.all(np.abs(y) < cfg["run"]["view_angle"] / 2)
    c["layout"] = dict(cfg["layout"], mode="array")
    m = build_layout(c, x_deg=np.array([0.0, 1.0]), y_deg=np.array([0.0, -1.0]))
    assert len(m) == 2


def test_thresholds_are_per_electrode_not_per_batch_row(sim):
    thr = sim._sim.threshold.get()[:, :, 0, 0]
    assert torch.allclose(thr[0], thr[-1])


# ------------------------------------------------------------------ C4: rendering behaviour
def test_zero_current_gives_black_image(sim):
    img = sim.render(np.zeros((3, sim.n_electrodes), np.float32))
    assert float(img.abs().max()) == 0.0


def test_threshold_gate(sim, atlas):
    """0.9*thr -> invisible, 1.1*thr -> visible (validates the atlas calibration and its monotonicity)."""
    cand = np.flatnonzero(np.isfinite(atlas.thr_amp) & (atlas.thr_amp * 1.1 < sim.cfg["atlas"]["amp_search_max"]))[:40]
    assert len(cand) >= 10
    for chunk in np.array_split(cand, int(np.ceil(len(cand) / sim.batch_size))):
        below = sim.response(_single_electrode_currents(sim.n_electrodes, chunk, 0.9 * atlas.thr_amp[chunk]))["visible"]
        above = sim.response(_single_electrode_currents(sim.n_electrodes, chunk, 1.1 * atlas.thr_amp[chunk]))["visible"]
        for b, e in enumerate(chunk):
            assert not bool(below[b, e]), f"electrode {e} visible below threshold"
            assert bool(above[b, e]), f"electrode {e} not visible above threshold"


def test_brightness_monotonic_in_current(sim, atlas):
    e = int(np.flatnonzero(atlas.usable())[0])
    amps = np.linspace(atlas.thr_amp[e] * 1.05, min(atlas.meta["i_max"], atlas.thr_amp[e] * 3), 4)
    st = sim.response(_single_electrode_currents(sim.n_electrodes, [e] * 4, amps))
    b = st["brightness"][:, e].cpu().numpy()
    assert np.all(np.diff(b) >= -1e-6)
    assert np.all(np.diff(st["sigma_deg"][:, e].cpu().numpy()) >= -1e-9)   # size grows with current


def test_atlas_position_matches_render(sim, atlas):
    usable = np.flatnonzero(atlas.usable())
    e = int(usable[np.argmax(atlas.sigma_px_ref[usable])])
    if atlas.sigma_px_ref[e] < 0.7:
        pytest.skip("no sufficiently large phosphene at this resolution")
    img = sim.render(_single_electrode_currents(sim.n_electrodes, [e], [atlas.meta["i_max"]]))[0, 0].cpu().numpy()
    r, c = np.unravel_index(np.argmax(img), img.shape)
    assert abs(r - atlas.row[e]) <= 1.5 and abs(c - atlas.col[e]) <= 1.5, (r, c, atlas.row[e], atlas.col[e])


def test_batch_matches_single(sim, atlas):
    rng = np.random.default_rng(1)
    cur = atlas.random_currents(8, 3, rng)
    batch = sim.render(cur)
    singles = torch.cat([sim.render(cur[i]) [None] for i in range(3)], 0)
    assert torch.allclose(batch, singles, atol=1e-6)


def test_determinism(sim, atlas):
    cur = atlas.random_currents(10, 2, np.random.default_rng(2))
    assert torch.equal(sim.render(cur), sim.render(cur))


# ------------------------------------------------------------------ C5: gradients
def test_gradients(sim, atlas):
    usable = np.flatnonzero(atlas.usable())
    big = usable[np.argsort(-atlas.sigma_px_ref[usable])][:6]
    sub = usable[np.argsort(-atlas.sigma_px_ref[usable])][6:8]         # will be driven below threshold
    cur = np.zeros((1, sim.n_electrodes), np.float32)
    cur[0, big] = np.minimum(atlas.thr_amp[big] * 1.3, atlas.meta["i_max"])
    cur[0, sub] = atlas.thr_amp[sub] * 0.5
    x = torch.tensor(cur, requires_grad=True)
    sim.render(x).sum().backward()
    g = x.grad[0].numpy()
    assert np.all(np.isfinite(g))
    if atlas.sigma_px_ref[big].max() >= 0.7:
        assert (g[big] > 0).any(), "no gradient reaches supra-threshold electrodes"
    assert np.all(g[sub] == 0), "sub-threshold electrodes must have exactly zero gradient"
    others = np.setdiff1d(np.arange(sim.n_electrodes), np.concatenate([big, sub]))
    assert np.all(g[others] == 0), "inactive electrodes must have zero gradient"


def test_gradient_sign_matches_finite_difference(sim, atlas):
    usable = np.flatnonzero(atlas.usable())
    e = int(usable[np.argmax(atlas.sigma_px_ref[usable])])
    if atlas.sigma_px_ref[e] < 0.7:
        pytest.skip("phosphenes too small at this resolution")
    a0 = min(atlas.thr_amp[e] * 1.2, atlas.meta["i_max"] * 0.95)
    base = np.zeros((1, sim.n_electrodes), np.float32)
    base[0, e] = a0
    x = torch.tensor(base, requires_grad=True)
    sim.render(x).sum().backward()
    d = 0.02 * a0
    up, dn = base.copy(), base.copy()
    up[0, e] += d
    dn[0, e] -= d
    fd = float((sim.render(up).sum() - sim.render(dn).sum()) / (2 * d))
    assert fd > 0 and float(x.grad[0, e]) > 0


# ------------------------------------------------------------------ noise
def test_dropout_all_gives_black(sim, atlas):
    cur = atlas.random_currents(20, 2, np.random.default_rng(3))
    img = sim.render(cur, noise=NoiseSpec(p_drop=1.0))
    assert float(img.abs().max()) == 0.0


def test_noise_reproducible_and_restores_thresholds(sim, atlas):
    cur = np.where(atlas.usable(), atlas.thr_amp * 1.0, 0.0).astype(np.float32)[None]   # borderline electrodes
    clean_before = sim.render(cur)
    g1, g2 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    n1 = sim.render(cur, noise=NoiseSpec(sigma_thr=0.5), generator=g1)
    n2 = sim.render(cur, noise=NoiseSpec(sigma_thr=0.5), generator=g2)
    clean_after = sim.render(cur)
    assert torch.equal(n1, n2)
    assert not torch.equal(n1, clean_before), "threshold jitter had no effect on borderline electrodes"
    assert torch.equal(clean_before, clean_after), "clean thresholds were not restored"


def test_n_visible_counts(sim, atlas):
    cur = atlas.random_currents(12, 2, np.random.default_rng(4), margin=1.2)
    nv = sim.n_visible(cur)
    assert torch.all(nv >= 1) and torch.all(nv <= 12)


# ------------------------------------------------------------------ constraints & atlas io
def test_constraints():
    i_max = 100e-6
    c = torch.tensor([[-5e-6, 0.0, 50e-6, 120e-6, 80e-6, 10e-6]])
    p = project_currents(c, i_max, k=3)
    assert float(p.min()) >= 0 and float(p.max()) <= i_max and int((p > 0).sum()) == 3
    q = quantize_ste(torch.tensor([0.0, 33e-6, 99e-6]), i_max, levels=4)
    assert torch.allclose(q, torch.tensor([0.0, 25e-6, 100e-6]), atol=1e-9)
    assert int(topk_mask(torch.tensor([0.1, 0.9, 0.5]), 2).sum()) == 2
    assert constraint_report(p, i_max, k=3)["ok"]
    assert not constraint_report(c, i_max, k=3)["ok"]
    x = torch.tensor([30e-6], requires_grad=True)
    quantize_ste(x, i_max, 4).sum().backward()
    assert float(x.grad) == 1.0                                          # straight-through


def test_atlas_roundtrip_and_summary(atlas, tmp_path):
    atlas.save(tmp_path / "a.npz")
    b = Atlas.load(tmp_path / "a.npz")
    assert np.array_equal(atlas.thr_amp, b.thr_amp) and b.meta == atlas.meta
    s = b.summary()
    assert s["n_electrodes_in_view"] == atlas.n and 0 <= s["frac_usable"] <= 1
    assert atlas.sample_nearest(np.ones(tuple(atlas.meta["image_shape"]))).shape == (atlas.n,)
