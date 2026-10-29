"""Hardware / biology perturbations applied at render time (used by Module E, robust training of B).

Supported perturbations
-----------------------
* electrode dropout     p_drop      : electrode receives zero current (fails to produce a phosphene)
* threshold variability sigma_thr   : per-electrode, per-sample log-normal factor on the tissue
                                      activation threshold (median-preserving)
* current gain noise    sigma_gain  : per-electrode, per-sample log-normal factor on the amplitude

Not supported: phosphene *position* jitter. Positions are baked into Dynaphos' phosphene maps at
construction time; to test it, build a second simulator from a jittered layout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class NoiseSpec:
    p_drop: float = 0.0
    sigma_thr: float = 0.0
    sigma_gain: float = 0.0

    def is_null(self) -> bool:
        return self.p_drop == 0 and self.sigma_thr == 0 and self.sigma_gain == 0


@dataclass
class NoiseSample:
    """Concrete draws for a (B, N) batch."""
    keep: torch.Tensor        # (B, N) 1 = working electrode, 0 = dropped
    thr_scale: torch.Tensor   # (B, N) multiplicative factor on activation threshold
    gain: torch.Tensor        # (B, N) multiplicative factor on current


def sample_noise(spec: NoiseSpec, batch: int, n: int, device, dtype,
                 generator: Optional[torch.Generator] = None) -> NoiseSample:
    def rnd(*shape):
        return torch.rand(*shape, generator=generator, device="cpu").to(device=device, dtype=dtype)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, device="cpu").to(device=device, dtype=dtype)

    keep = (rnd(batch, n) >= spec.p_drop).to(dtype)
    thr_scale = torch.exp(spec.sigma_thr * randn(batch, n))
    gain = torch.exp(spec.sigma_gain * randn(batch, n))
    return NoiseSample(keep=keep, thr_scale=thr_scale, gain=gain)
