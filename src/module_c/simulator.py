"""Frozen Dynaphos wrapper (Module C, steps C2, C5, C6).

Maps electrode currents (amperes, shape (B, N)) to a phosphene image (B, 1, H, W).

Design notes (all verified against dynaphos 0.1.3 source):

* Dynaphos is *stateful*: activation and a memory trace evolve frame by frame. A static
  stimulus is therefore shown for ``render.n_frames`` frames and the last frame is returned.
* Gradients: Dynaphos detaches the previous state in every update. The warm-up frames are run
  under ``torch.no_grad`` here, so the gradient d(image)/d(current) is the sensitivity of the
  LAST frame with the warm-up state treated as constant. Direction is correct; magnitude is
  smaller than the true steady-state sensitivity. Use Adam-style optimisers, not raw SGD scales.
* NaN guard: Dynaphos' size model is sqrt(current), whose derivative is infinite at 0 A, so an
  electrode at exactly zero current yields NaN gradients (inf * 0). The wrapper therefore
  floors the amplitude fed to Dynaphos at ``_EPS_AMP`` = 1 nA, far below the rheobase
  (23.9 uA), so forward behaviour is unchanged (no charge, no phosphene) and gradients are 0.
* Electrodes below their activation threshold give an exactly zero gradient (hard gate in
  Dynaphos), as do electrodes at zero current. Parametrise active currents above threshold
  (see Atlas.thr_amp) in Module B.
* Dynaphos samples activation thresholds with shape (batch, N): in batched mode each batch row
  would get *different* thresholds for the same electrode. This wrapper overrides that so
  thresholds are a property of the electrode (shared by all rows) unless noise is requested.
* Dynaphos calls ``torch.use_deterministic_algorithms(True)`` globally on construction, which
  makes e.g. ``grid_sample`` backward raise on CUDA. We relax it (``render.deterministic``).
* Memory: the largest tensor is (B, N, H, W) floats. 500 electrodes at 128x128 with B=8 is
  about 260 MB per tensor (several are alive during autograd).
"""
from __future__ import annotations

import copy
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from dynaphos.simulator import GaussianSimulator

from .layout import build_layout, validate_config
from .noise import NoiseSpec, sample_noise

ArrayLike = Union[np.ndarray, torch.Tensor]

_EPS_AMP = 1e-9   # amperes; see NaN guard in the module docstring


class ProstheticSimulator:
    """Frozen forward model: electrode currents -> phosphene image."""

    def __init__(self, cfg: dict, coords=None, seed: Optional[int] = None):
        validate_config(cfg)
        self.cfg = copy.deepcopy(cfg)
        rng = np.random.default_rng(self.cfg["run"]["seed"] if seed is None else seed)

        if coords is None:
            coords = build_layout(self.cfg)
        # NOTE: GaussianSimulator removes out-of-view phosphenes from `coords` in place.
        self._sim = GaussianSimulator(self.cfg, coords, rng=rng)
        torch.use_deterministic_algorithms(bool(self.cfg["render"]["deterministic"]), warn_only=True)

        self.coords = coords                       # now filtered: index i == electrode i
        self.n_electrodes: int = self._sim.num_phosphenes
        self.batch_size: int = int(self.cfg["run"]["batch_size"])
        self.device = self._sim.data_kwargs["device"]
        self.dtype = self._sim.data_kwargs["dtype"]
        self.image_shape: Tuple[int, int] = tuple(self._sim.phosphene_maps.shape[-2:])  # (H, W)
        self.deg2pix: float = float(self._sim.deg2pix_coeff)
        self.i_max: float = float(self.cfg["constraints"]["i_max"])
        self.levels = self.cfg["constraints"]["levels"]
        self.n_frames_default: int = int(self.cfg["render"]["n_frames"])

        # per-electrode base thresholds (activation units), identical across batch rows
        thr = self._sim.threshold.get()                              # (B, N, 1, 1)
        self.base_thr = thr[0, :, 0, 0].clone()                      # (N,)
        self._apply_thresholds(self.base_thr.expand(self.batch_size, -1))

        # frozen: nothing in the wrapper is a trainable parameter
        self.frozen = True

    # ------------------------------------------------------------------ helpers
    def _apply_thresholds(self, thr_bn: torch.Tensor) -> None:
        arr = thr_bn.detach().cpu().numpy().reshape(self._sim.shape)
        self._sim.threshold.reinitialize(arr)

    def _as_batch(self, currents: ArrayLike) -> Tuple[torch.Tensor, bool]:
        c = torch.as_tensor(currents, device=self.device, dtype=self.dtype).contiguous()
        squeeze = c.ndim == 1
        if squeeze:
            c = c[None]
        if c.ndim != 2 or c.shape[1] != self.n_electrodes:
            raise ValueError(f"currents must have shape (B, {self.n_electrodes}); got {tuple(c.shape)}")
        return c, squeeze

    def _run_chunk(self, chunk: torch.Tensor, noise: Optional[NoiseSpec], n_frames: int,
                   generator: Optional[torch.Generator], make_image: bool):
        B, N = chunk.shape
        thr = self.base_thr.expand(B, -1)
        if noise is not None and not noise.is_null():
            ns = sample_noise(noise, B, N, self.device, self.dtype, generator)
            chunk = chunk * ns.keep * ns.gain          # differentiable w.r.t. chunk
            thr = thr * ns.thr_scale
        self._apply_thresholds(thr)
        chunk = chunk.clamp(min=_EPS_AMP)               # NaN guard (gradient of sqrt at 0)
        self._sim.reset()
        with torch.no_grad():
            for _ in range(n_frames - 1):
                self._sim.update(chunk.detach())
        if make_image:
            img = self._sim(chunk)                      # (B, H, W), clamped to [0, 1]
        else:
            self._sim.update(chunk)
            img = None
        st = self._sim.get_state()
        act = st["activation"][:, :, 0, 0]
        thr_state = st["threshold"][:, :, 0, 0]
        state = {
            "visible": (act > thr_state),                        # (B, N) bool
            "brightness": st["brightness"][:, :, 0, 0],          # (B, N) in [0, 1]
            "sigma_deg": st["sigma"][:, :, 0, 0],                # (B, N) phosphene sigma, degrees
            "activation": act,
            "threshold": thr_state,
        }
        return img, state

    def _run(self, currents, noise, n_frames, generator, make_image):
        c, squeeze = self._as_batch(currents)
        T = int(n_frames or self.n_frames_default)
        B = self.batch_size
        imgs, states = [], []
        try:
            for s in range(0, c.shape[0], B):
                chunk = c[s:s + B]
                nb = chunk.shape[0]
                if nb < B:                                        # pad with zero-current rows
                    pad = torch.zeros(B - nb, self.n_electrodes, device=self.device, dtype=self.dtype)
                    chunk = torch.cat([chunk, pad], 0)
                img, st = self._run_chunk(chunk, noise, T, generator, make_image)
                if img is not None:
                    imgs.append(img[:nb])
                states.append({k: v[:nb] for k, v in st.items()})
        finally:
            self._apply_thresholds(self.base_thr.expand(B, -1))   # always restore clean thresholds
        state = {k: torch.cat([s[k] for s in states], 0) for k in states[0]}
        img = torch.cat(imgs, 0).unsqueeze(1) if make_image else None
        if squeeze:
            state = {k: v[0] for k, v in state.items()}
            img = img[0] if img is not None else None
        return img, state

    # ------------------------------------------------------------------ public API
    def render(self, currents: ArrayLike, noise: Optional[NoiseSpec] = None,
               n_frames: Optional[int] = None, generator: Optional[torch.Generator] = None,
               return_state: bool = False):
        """Currents (B, N) [A] -> phosphene images (B, 1, H, W) in [0, 1].

        A 1-D input (N,) returns (1, H, W). Differentiable w.r.t. `currents` (see module notes).
        """
        img, state = self._run(currents, noise, n_frames, generator, make_image=True)
        return (img, state) if return_state else img

    def response(self, currents: ArrayLike, noise: Optional[NoiseSpec] = None,
                 n_frames: Optional[int] = None, generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
        """Per-electrode state only (visible mask, brightness, sigma). No image: much cheaper."""
        with torch.no_grad():
            _, state = self._run(currents, noise, n_frames, generator, make_image=False)
        return state

    __call__ = render

    def n_visible(self, currents: ArrayLike, **kw) -> torch.Tensor:
        """Number of electrodes that actually produce a phosphene (>= threshold), per sample."""
        return self.response(currents, **kw)["visible"].sum(-1)

    def to_numpy(self, x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()
