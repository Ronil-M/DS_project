"""Robustness perturbations for Experiment 3 (plan 3.4 / 7.3).

Dropout, threshold jitter and gain noise are applied by Module C at render time (module_c.noise.NoiseSpec);
this file builds the sweep grid and adds PHOSPHENE POSITION JITTER, which Module C does not support because
positions are baked into Dynaphos' phosphene maps: `jittered_simulator` builds a second simulator whose
phosphene centres are displaced, keeping electrode indices and thresholds identical. The planner still uses
the NOMINAL atlas, which is the point: the implant's real phosphenes are not where the model says.
"""
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

from module_c.noise import NoiseSpec

P_DROP = (0.0, 0.05, 0.10, 0.20, 0.30)
SIGMA_THR = (0.0, 0.10, 0.25, 0.50)


def noise_grid(p_drop: Iterable[float] = P_DROP, sigma_thr: Iterable[float] = SIGMA_THR,
               sigma_gain: float = 0.0, one_at_a_time: bool = True) -> List[Dict]:
    """List of {'label', 'spec'}. one_at_a_time sweeps each factor alone (plan Exp 3); else the full product."""
    out = []
    if one_at_a_time:
        out += [{"label": f"p_drop={p:g}", "spec": NoiseSpec(p_drop=p, sigma_gain=sigma_gain)} for p in p_drop]
        out += [{"label": f"sigma_thr={s:g}", "spec": NoiseSpec(sigma_thr=s, sigma_gain=sigma_gain)} for s in sigma_thr if s > 0]
    else:
        out += [{"label": f"p_drop={p:g},sigma_thr={s:g}", "spec": NoiseSpec(p, s, sigma_gain)} for p in p_drop for s in sigma_thr]
    return out


def jittered_simulator(sim, sigma_deg: float, seed: int = 0):
    """Simulator with every phosphene centre displaced by N(0, sigma_deg) degrees (same electrode indices/thresholds)."""
    import copy
    from dynaphos.utils import Map
    from module_c.simulator import ProstheticSimulator
    rng = np.random.default_rng(seed)
    x, y = (np.asarray(a, float) for a in sim.coords.cartesian)
    hemi = sim.cfg["run"]["view_angle"] / 2
    ox, oy = sim.cfg["run"]["origin"]
    lim = hemi - 0.02                       # stay inside the field of view, else Dynaphos would delete electrodes
    xj = np.clip(x + rng.normal(0, sigma_deg, x.shape), ox - lim, ox + lim)
    yj = np.clip(y + rng.normal(0, sigma_deg, y.shape), oy - lim, oy + lim)
    cfg = copy.deepcopy(sim.cfg)
    cfg["layout"]["mode"] = "array"
    s2 = ProstheticSimulator(cfg, coords=Map(x=xj, y=yj))
    assert s2.n_electrodes == sim.n_electrodes, "jitter removed electrodes; reduce sigma_deg"
    s2.base_thr = sim.base_thr.clone()                      # same per-electrode thresholds as the nominal implant
    s2._apply_thresholds(s2.base_thr.expand(s2.batch_size, -1))
    return s2
