# Module C: Dynaphos phosphene simulator (frozen forward model)

Maps electrode currents (amperes) to a phosphene image. Verified against dynaphos 0.1.3.

    pip install -r requirements.txt
    pytest tests/test_module_c.py -q                 # 16 acceptance tests (CPU, ~15 s)
    python scripts/c_build_atlas.py                  # atlas.npz, layout.png, random_render.png

## Use from Module B / E

```python
import sys; sys.path.insert(0, "src")
import numpy as np, torch
from module_c.layout import load_config
from module_c.simulator import ProstheticSimulator
from module_c.atlas import build_atlas, Atlas
from module_c.constraints import project_currents
from module_c.noise import NoiseSpec

cfg   = load_config("configs/module_c.yaml")
sim   = ProstheticSimulator(cfg)            # frozen; builds layout, drops out-of-view electrodes
atlas = build_atlas(sim)                    # per-electrode position, size, calibrated current threshold
atlas.save("outputs/module_c/atlas.npz")

I   = atlas.random_currents(k=16, batch=4, rng=np.random.default_rng(0))   # (4, N) amperes
img = sim.render(I)                                                         # (4, 1, H, W) in [0, 1]
img = sim.render(I, noise=NoiseSpec(p_drop=0.1, sigma_thr=0.25))            # robustness tests
n_visible = sim.n_visible(I)                                                # phosphenes actually seen

# differentiable (for B2): gradient w.r.t. currents
x = torch.tensor(I, requires_grad=True)
sim.render(project_currents(x, sim.i_max, k=16)).mean().backward()
```

## Things that will bite you (all found while testing)

1. **Units are amperes.** Thresholds in Dynaphos are on tissue activation, not current; use `atlas.thr_amp`.
2. **Static stimuli need several frames.** One frame at 100 uA lights only ~58 % of electrodes; five or
   more frames give ~100 %. `render.n_frames` (default 12) sets the warm-up. The last frame is returned.
3. **Gradient = last-frame sensitivity** (Dynaphos detaches earlier state). Use Adam, not raw SGD.
4. **Zero current gave NaN gradients** in raw Dynaphos (sqrt size model). The wrapper floors amplitude at 1 nA.
5. **Sub-threshold electrodes have exactly zero gradient.** Parametrise active currents in
   `[atlas.thr_amp, i_max]` (see doc section 6.3).
6. **Dynaphos draws a different threshold per batch row** in batched mode; the wrapper makes thresholds a
   per-electrode property (shared across rows) unless you ask for noise.
7. **Resolution matters.** At 256 px / 16 deg, ~48 % of phosphenes have sigma < 0.5 px at 100 uA. At 128 px
   most are sub-pixel and vanish. Check `atlas.summary()["frac_sigma_below_0.5px"]`.
8. **Out-of-view electrodes are removed in place** from the layout. Electrode index = index after construction.
9. Dynaphos switches on `torch.use_deterministic_algorithms(True)`; the wrapper relaxes this (`render.deterministic`).
10. **i_max default (100 uA)** is Dynaphos' `stimulus_scale`, not a safety limit. Set and cite your own value.

---

# Modules B and E (planner and evaluation), built against Module C

    pytest tests/ -q                       # C: 16, B: 15, E: 14 tests (CPU, ~2 min total)
    python scripts/c_build_atlas.py        # atlas first (B and E read outputs/module_c/atlas.npz)
    python scripts/b_demo.py               # R0 / R1 / B1 renders at K = 8, 16, 32
    python scripts/e_train_evaluators.py   # E_main, E_alt and T_c (different archs and seeds)
    python scripts/b_train.py --k 16 --seed 0 [--lam-t 0.05 | --lam-c 0 --lam-t 1 | --robust]
    python scripts/e_run_experiments.py --exp 1 2 3 4

## How B and E use C
- Currents are **amperes** everywhere (the plan's microamps are converted). Every planner returns `(B, N)` float32
  currents that pass `module_b.constraints.CurrentParam.report`: active >= 1.05 x `atlas.thr_amp`, <= `i_max`,
  at most K non-zero, unusable electrodes never chosen, quantisation per `constraints.levels`.
- Rendering only ever goes through `ProstheticSimulator.render`; noise through `NoiseSpec`. Contour maps are sampled at the
  atlas `row/col`, never by electrode index. The shape generator takes `image_shape`/`deg2pix` from the simulator.
- Position jitter (not supported by C) is done in `evaluation/noise.py` with a second simulator that keeps electrode indices
  and thresholds; the planner keeps the nominal atlas.

## Deviations from the plan (and why)
1. **Selection gradient.** Straight-through top-K alone gives inactive electrodes zero gradient (Dynaphos gradient is 0 at
   0 A). `module_b/training.py` adds a first-order footprint surrogate; amplitudes still train through the real simulator.
   Ignores saturation and phosphene merging. Tested in `test_selection_gradient_reaches_inactive_electrodes`.
2. **Loss terms are mass-normalised** (precision / recall as fractions) instead of pixel means, so weights do not depend on
   image size or K.
3. Evaluator and T_c training data live in `common/sparse_data.py`, imported by neither planner; a test checks that
   `evaluation/` and `common/` never import `module_b`.
4. Module A is **not** included; `datasets/shapes.py` is a minimal generator. B2b label conditioning takes integer labels,
   so A's `predict` output plugs in directly.
5. Exp 5 (human check) is not implemented.

## Honest status of the evidence
- All tests pass on a reduced config (128 px, 8 deg, 150 electrodes). **The full 256 px / 500 electrode config was not run**
  (the sandbox has 3 GB RAM); `b_demo.py` and the experiment scripts are untested end to end, and `b_train.py` /
  `e_run_experiments.py` have only been syntax-checked.
- B2b smoke run (24 images, K=8, 6 epochs, reduced config): validation contour loss 2.39 -> 1.86, still falling, vs B1 1.68
  and random 2.45. **B2b has not been shown to match or beat B1.** lr 1e-3 and 3e-3 diverged; default is now 5e-4.
- At this reduced resolution the contour loss barely separates B1 (1.677) from a dark image (1.684), because phosphenes are
  tiny relative to the pixel grid. Check `atlas.summary()["frac_sigma_below_0.5px"]` and prefer the 256 px config for real runs.
