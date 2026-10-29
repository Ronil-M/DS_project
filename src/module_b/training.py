"""Training of the learned planners through frozen Dynaphos (plan 6.3, "Training recipe").

THE SELECTION-GRADIENT PROBLEM (read this before changing anything)
-------------------------------------------------------------------
The plan's parametrisation I_e = m_e * (...) with a straight-through top-k mask does NOT, by itself,
let the render loss tell an *inactive* electrode to switch on: dLoss/dm_e = dLoss/dI_e * (...) and
Dynaphos gives dLoss/dI_e = 0 at I_e = 0 (hard threshold, plus the 1 nA floor in module_c). Only
electrodes that are already selected would ever receive a selection gradient.

Fix used here (first-order surrogate): every step runs ONE backward pass through Dynaphos with the hard mask
detached (this trains the amplitudes u_e through the real simulator) and records g = dLoss/dP. The
selection logits then get   dLoss/dm_e  ~=  < g , phi_e >,   phi_e = unit-peak Gaussian footprint of electrode e
at its atlas centre/sigma (module_b.footprint). That is exactly the loss change from adding phosphene e under
linear superposition, available for ALL electrodes, active or not. It ignores brightness saturation and
merging; the amplitudes are still trained through the true simulator.

Model selection uses the validation CONTOUR loss only; the evaluator of Module E never enters here.
"""
from __future__ import annotations

import copy
import gc
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from module_c.noise import NoiseSpec

from .contours import contour_target
from .footprint import GaussianSampler
from .learned import AmortisedPlanner, TemplatePlanner
from .losses import budget_loss, contour_loss, task_loss


@dataclass
class TrainCfg:
    k: int = 16
    epochs: int = 30
    batch: int = 8
    lr: float = 1e-3
    lr_final: float = 1e-4
    lam_c: float = 1.0                    # 0 => the "trap" arm (task loss only)
    lam_t: float = 0.0                    # task weight; keep small and warm it up late
    lam_s: float = 0.0                    # budget (charge) weight
    task_warmup_epochs: int = 5           # lam_t is 0 until this epoch, then ramps over the same number of epochs
    tau: float = 1.0                      # temperature of the straight-through soft mask
    peak: float = 0.7                     # assumed peak brightness of a newly switched-on phosphene (surrogate)
    noise: Optional[NoiseSpec] = None     # robust arm: dropout / threshold jitter sampled in every forward pass
    blur_sigma: float = 1.5
    tol_px: float = 2.0
    contour_kw: dict = field(default_factory=dict)
    seed: int = 0
    grad_clip: float = 1.0


# ----------------------------------------------------------------------------------------- checkpointing
_CKPT_VERSION = 1


def _signature(cfg: TrainCfg, n_train: int, planner: AmortisedPlanner) -> dict:
    """Everything that must be identical for a resume to be a true continuation."""
    return {"k": cfg.k, "epochs": cfg.epochs, "batch": cfg.batch, "lr": cfg.lr, "lr_final": cfg.lr_final,
            "lam_c": cfg.lam_c, "lam_t": cfg.lam_t, "lam_s": cfg.lam_s, "task_warmup_epochs": cfg.task_warmup_epochs,
            "tau": cfg.tau, "peak": cfg.peak, "blur_sigma": cfg.blur_sigma, "tol_px": cfg.tol_px, "seed": cfg.seed,
            "noise": str(cfg.noise), "n_train": int(n_train), "n_labels": int(planner.n_labels)}


def _save_ckpt(path: Path, payload: dict) -> None:
    """Atomic write: a crash during saving can never corrupt the previous checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def warm_start(planner: AmortisedPlanner, pt_path, log: Callable[[str], None] = print) -> None:
    """Load only the planner WEIGHTS of a finished run (the .pt written by b_train.py). Optimizer, schedule and
    epoch counter start fresh. Use it to continue from an earlier run with a different epoch budget or K."""
    ck = torch.load(pt_path, map_location=planner.device)
    if ck.get("n_labels", 0) != planner.n_labels:
        raise ValueError(f"--init-from was trained with n_labels={ck.get('n_labels')} but this run uses "
                         f"n_labels={planner.n_labels} (check --label-conditioned and the class list).")
    try:
        planner.load_state_dict(ck["state"])
    except RuntimeError as e:
        raise ValueError("--init-from weights do not fit this planner/atlas (different electrode layout or "
                         f"architecture?): {str(e).splitlines()[0]}") from e
    log(f"Warm start: loaded planner weights from {pt_path} (trained for K={ck.get('k')}); "
        f"optimizer and learning-rate schedule start fresh.")


def _loss_and_surrogate(sim, planner_param, sampler, s, u, k, T, y, cfg: TrainCfg, task_clf, lam_t, generator=None):
    """One forward/backward. Returns (loss value, dict of parts). Gradients are accumulated into s/u's graph."""
    hard = planner_param.hard_mask(s, k)
    soft = torch.sigmoid(s / cfg.tau)
    m_ste = hard + soft - soft.detach()
    I = planner_param.quantise(hard * planner_param.amplitude(u))          # grads flow to u only
    P = sim.render(I, noise=cfg.noise, generator=generator)
    store: Dict[str, torch.Tensor] = {}
    P.register_hook(lambda g: store.__setitem__("g", g))
    lc, parts = contour_loss(P, T, cfg.blur_sigma, cfg.tol_px)
    loss = cfg.lam_c * lc
    if task_clf is not None and lam_t > 0 and y is not None:
        lt = task_loss(task_clf, P, y)
        loss = loss + lam_t * lt
        parts["task"] = lt.detach()
    if cfg.lam_s > 0:
        loss = loss + cfg.lam_s * budget_loss(I, planner_param.i_max)
    loss.backward(retain_graph=True)                                        # amplitudes (and the encoder via u)
    g = store["g"].detach()
    g_sel = sampler.footprint_dot(g) * cfg.peak                              # (B, N) first-order dLoss of switching e on
    (m_ste * g_sel).sum().backward()                                         # selection logits (and the encoder via s)
    parts["contour"] = lc.detach()
    return float(loss.detach()), parts


def _task_weight(cfg: TrainCfg, epoch: int) -> float:
    if cfg.lam_t <= 0:
        return 0.0
    if cfg.lam_c == 0:                                  # trap arm: task loss from the start
        return cfg.lam_t
    w = max(cfg.task_warmup_epochs, 1)
    return cfg.lam_t * float(np.clip((epoch - cfg.task_warmup_epochs + 1) / w, 0.0, 1.0))


@torch.no_grad()
def validation_contour_loss(planner: AmortisedPlanner, sim, x_val, cfg: TrainCfg, batch: int = 2, y_val=None) -> float:   # batch 2: Dynaphos memory ~ batch x electrodes x H x W (16 images at 512 px = ~4 GB per tensor)
    planner.eval()
    tot, n = 0.0, 0
    for i in range(0, len(x_val), batch):
        xb = torch.as_tensor(x_val[i:i + batch], device=planner.device)
        T = contour_target(xb, cfg.blur_sigma, **cfg.contour_kw)
        lab = y_val[i:i + batch] if (y_val is not None and planner.n_labels) else None
        I = torch.as_tensor(planner.plan(xb, cfg.k, lab), device=sim.device)
        P = sim.render(I)
        l, _ = contour_loss(P, T.to(P.device), cfg.blur_sigma, cfg.tol_px)
        tot += float(l) * len(xb); n += len(xb)
    return tot / max(n, 1)


def train_amortised(planner: AmortisedPlanner, sim, atlas, x_train, y_train, x_val, cfg: TrainCfg,
                    task_clf: Optional[torch.nn.Module] = None, y_val=None, log: Callable[[str], None] = print,
                    ckpt_path: Optional[str] = None, resume: bool = False) -> List[dict]:
    """Train B2b for one K. Keeps the checkpoint with the best VALIDATION CONTOUR loss. Returns the history.

    ckpt_path: if given, a full training state (weights, optimizer, LR schedule, RNG streams, best-so-far, history)
               is written atomically after EVERY epoch.
    resume:    continue from ckpt_path if it exists (exact continuation; settings must match the saved run).
               A crash mid-epoch repeats that epoch from its start."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    sampler = GaussianSampler(atlas, tuple(atlas.meta["image_shape"]), device=planner.device)
    opt = torch.optim.Adam(planner.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: (cfg.lr_final / cfg.lr) ** (e / max(cfg.epochs - 1, 1)))
    gen = torch.Generator().manual_seed(cfg.seed)
    best, best_state, hist = float("inf"), None, []
    n = len(x_train)
    start_ep = 0
    if resume and ckpt_path is not None and Path(ckpt_path).exists():
        ck = torch.load(ckpt_path, map_location=planner.device, weights_only=False)   # our own file: trusted
        want, have = _signature(cfg, n, planner), ck["signature"]
        diff = {k: (have.get(k), want[k]) for k in want if have.get(k) != want[k]}
        if diff:
            raise ValueError("Cannot resume: settings differ from the saved run (saved, now): " + str(diff) +
                             ". Use the same settings, or --init-from <finished .pt> for a warm start.")
        planner.load_state_dict(ck["planner"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        best, best_state, hist = ck["best"], ck["best_state"], ck["hist"]
        rng.bit_generator.state = ck["rng_np"]; gen.set_state(ck["gen"].cpu()); torch.set_rng_state(ck["rng_torch"].cpu())
        if ck.get("rng_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([t.cpu() for t in ck["rng_cuda"]])
        start_ep = ck["epoch"] + 1
        log(f"[B2b K={cfg.k}] RESUMED from {ckpt_path}: {start_ep}/{cfg.epochs} epochs already done, best val_contour {best:.4f}")
        if start_ep >= cfg.epochs:
            log("  nothing left to train; returning the best saved weights.")
    elif resume and ckpt_path is not None:
        log(f"[B2b K={cfg.k}] --resume given but {ckpt_path} does not exist; starting from scratch.")
    for ep in range(start_ep, cfg.epochs):
        planner.train()
        perm = rng.permutation(n)
        lam_t = _task_weight(cfg, ep)
        run = []
        n_steps, t_ep = max(n // cfg.batch, 1), time.time()
        if ep == 0:
            log(f"[B2b K={cfg.k}] {n} training images, {n_steps} steps/epoch, {cfg.epochs} epochs = {n_steps * cfg.epochs} steps")
        for i in range(0, n - cfg.batch + 1, cfg.batch):
            idx = perm[i:i + cfg.batch]
            xb = torch.as_tensor(x_train[idx], device=planner.device)
            yb = torch.as_tensor(y_train[idx], device=planner.device) if y_train is not None else None
            with torch.no_grad():
                T = contour_target(xb, cfg.blur_sigma, **cfg.contour_kw)
            opt.zero_grad(set_to_none=True)
            s, u = planner(xb, yb if planner.n_labels else None)
            l, parts = _loss_and_surrogate(sim, planner.param, sampler, s, u, cfg.k, T, yb, cfg, task_clf, lam_t, gen)
            torch.nn.utils.clip_grad_norm_(planner.parameters(), cfg.grad_clip)
            opt.step()
            run.append(l)
            st = len(run)
            if st in (1, 5) or st % 50 == 0:
                sps = (time.time() - t_ep) / st
                log(f"    epoch {ep} step {st}/{n_steps}  {sps:.2f}s/step  (~{sps * (n_steps - st) / 60:.1f} min left in this epoch)")
        sched.step()
        v = validation_contour_loss(planner, sim, x_val, cfg, y_val=y_val)
        gc.collect()
        hist.append({"epoch": ep, "train_loss": float(np.mean(run)), "val_contour": v, "lam_t": lam_t})
        if v < best:
            best, best_state = v, copy.deepcopy(planner.state_dict())
        if ckpt_path is not None:                                                  # BEFORE the log line (see tests)
            _save_ckpt(ckpt_path, {
                "version": _CKPT_VERSION, "epoch": ep, "signature": _signature(cfg, n, planner),
                "planner": planner.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "best": best, "best_state": best_state, "hist": hist,
                "rng_np": rng.bit_generator.state, "gen": gen.get_state(), "rng_torch": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None})
        log(f"[B2b K={cfg.k}] epoch {ep:3d}  train {hist[-1]['train_loss']:.4f}  val_contour {v:.4f}  lam_t {lam_t:.3g}"
            f"  ({(time.time() - t_ep) / 60:.1f} min)")
    planner.load_state_dict(best_state)
    planner.eval()
    return hist


def fit_templates(sim, atlas, proto_x: np.ndarray, cfg: TrainCfg, steps: int = 150, lr: float = 0.1,
                  log: Callable[[str], None] = print) -> TemplatePlanner:
    """B2a: optimise one (s, u) vector per class against that class's prototype contour. Returns a TemplatePlanner."""
    from .constraints import CurrentParam
    torch.manual_seed(cfg.seed)
    param = CurrentParam(atlas, device=sim.device)
    sampler = GaussianSampler(atlas, tuple(atlas.meta["image_shape"]), device=sim.device)
    C = len(proto_x)
    s = torch.zeros(C, sim.n_electrodes, device=sim.device)
    s += 0.01 * torch.randn_like(s)
    u = torch.zeros(C, sim.n_electrodes, device=sim.device)
    s.requires_grad_(True); u.requires_grad_(True)
    T = contour_target(torch.as_tensor(proto_x, device=sim.device), cfg.blur_sigma, **cfg.contour_kw)
    opt = torch.optim.Adam([s, u], lr=lr)
    cfg0 = copy.copy(cfg); cfg0.noise = None; cfg0.lam_t = 0.0
    for it in range(steps):
        opt.zero_grad(set_to_none=True)
        l, _ = _loss_and_surrogate(sim, param, sampler, s, u, cfg.k, T, None, cfg0, None, 0.0)
        opt.step()
        if it % 25 == 0 or it == steps - 1:
            log(f"[B2a K={cfg.k}] step {it:3d}  loss {l:.4f}")
    with torch.no_grad():
        cur = param.currents(param.hard_mask(s, cfg.k), u).cpu().numpy()
    return TemplatePlanner(cur)
