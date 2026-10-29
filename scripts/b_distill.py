#!/usr/bin/env python
"""Teach the learned planner (B2b) to imitate the heuristic planner (B1), with NO Dynaphos in the training loop.

    python scripts/b_distill.py --data-dir data/split --k 16 --max-per-class 40 --n-aug 2

Why: training through the simulator with the contour loss stalled (the planner found only about half of the
way from random to the heuristic). The heuristic already gets E_main accuracy ~0.45, so this script first makes
B2b reproduce its choices:  for every training image, the heuristic's electrodes at this K are the target; the
planner is trained with a class-balanced binary cross-entropy on its selection logits s_e and a small
regression on the amplitude logits u_e of the selected electrodes. No rendering is needed, so an epoch takes
seconds to a few minutes on CPU.

It writes the same file as b_train.py (outputs/module_b/b2b_k{K}_s{seed}_c1_t0_real{tag}.pt), so
b_baselines.py, e_run_experiments.py and `b_train.py --init-from` accept it directly. To refine it afterwards
with the contour loss through Dynaphos, use a LOW learning rate, e.g.
    python scripts/b_train.py --data-dir data/split --k 16 --init-from <that .pt> --epochs 10 --tag _ft
and keep the refined model only if it beats this one on b_baselines.py / E_main accuracy.

The planner is trained for ONE K per run (the heuristic's set depends on K). At another K the top-K of the same
scores is used, which is fine for nearby K; distil again for a very different K.
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, torch, torch.nn.functional as F, yaml

from common.world import build_world
from datasets.shapes import ShapeSpec
from module_b.heuristic import HeuristicPlanner
from module_b.learned import AmortisedPlanner


def heuristic_targets(heur, x, k, batch=16, log=print):
    cur = []
    t0 = time.time()
    for i in range(0, len(x), batch):
        cur.append(heur.plan(x[i:i + batch], k))
        if (i // batch) % 10 == 0:
            log(f"  heuristic targets {min(i + batch, len(x))}/{len(x)}  ({time.time() - t0:.0f}s)")
    return np.concatenate(cur)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--experiment", default="configs/experiment.yaml")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-per-class", type=int, default=40)
    ap.add_argument("--n-aug", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--no-rotate", action="store_true")
    ap.add_argument("--w-amp", type=float, default=0.2, help="weight of the amplitude regression")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    ex = yaml.safe_load(open(a.experiment)); H = ex["heuristic"]
    cfg, sim, atlas = build_world(ex["paths"]["module_c_config"], ex["paths"]["atlas"])
    spec = ShapeSpec.from_sim(sim)
    from datasets.real_dataset import make_real_datasets
    tr, va = make_real_datasets(a.data_dir, spec, val_frac=a.val_frac, n_aug_train=a.n_aug, seed=a.seed,
                                rotate=not a.no_rotate, max_per_class=a.max_per_class or None, keep_contour=False)
    classes = tr["classes"]
    heur = HeuristicPlanner(atlas, sim, method=H["method"], pool=H["pool"], pool_excl_deg=H["pool_excl_deg"],
                            excl_deg=H["excl_deg"], floor_frac=H["floor_frac"], target_brightness=H["target_brightness"])
    planner = AmortisedPlanner(atlas, levels=sim.levels, n_labels=0)
    param = planner.param
    thr, imax = param.thr.numpy(), param.i_max
    usable = param.usable

    def prep(x, name):
        print(f"computing heuristic targets for {name} images (K={a.k}) ...")
        cur = heuristic_targets(heur, x, a.k)
        m = (cur > 0).astype(np.float32)
        frac = np.clip((cur - thr) / np.maximum(imax - thr, 1e-12), 0.05, 0.95)
        u = np.log(frac / (1 - frac)).astype(np.float32)
        print(f"  heuristic switched on {m.sum(1).mean():.1f} electrodes per image on average (K={a.k})")
        return torch.from_numpy(m), torch.from_numpy(u)

    mt, ut = prep(tr["x"], "training"); mv, uv = prep(va["x"], "validation")
    pos = float(mt[:, usable].mean()); pw = float(np.clip((1 - pos) / max(pos, 1e-6), 1.0, 30.0))
    print(f"selection prevalence {pos:.3f} -> positive weight {pw:.1f}")
    xt = torch.as_tensor(tr["x"]); xv = torch.as_tensor(va["x"])
    if xt.ndim == 3: xt, xv = xt[:, None], xv[:, None]
    opt = torch.optim.AdamW(planner.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * ((len(xt) + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)
    g = torch.Generator().manual_seed(a.seed)
    um = usable[None]

    def evaluate():
        planner.eval(); inter = tot_sel = tot_tgt = 0.0; bce = 0.0
        with torch.no_grad():
            for i in range(0, len(xv), 16):
                s, u = planner(xv[i:i + 16])
                hard = param.hard_mask(s, a.k)
                m = mv[i:i + 16]
                inter += float((hard * m).sum()); tot_sel += float(hard.sum()); tot_tgt += float(m.sum())
                bce += float(F.binary_cross_entropy_with_logits(s[:, usable], m[:, usable], pos_weight=torch.tensor(pw), reduction="sum"))
        prec, rec = inter / max(tot_sel, 1), inter / max(tot_tgt, 1)
        return {"bce": bce / (len(xv) * int(usable.sum())), "overlap_precision": prec, "overlap_recall": rec,
                "f1": 2 * prec * rec / max(prec + rec, 1e-9)}

    best, best_state, hist = -1.0, None, []
    t0 = time.time()
    for ep in range(a.epochs):
        planner.train(); perm = torch.randperm(len(xt), generator=g); run = 0.0
        for i in range(0, len(xt), a.batch):
            idx = perm[i:i + a.batch]
            s, u = planner(xt[idx]); m, ut_b = mt[idx], ut[idx]
            l_sel = F.binary_cross_entropy_with_logits(s[:, usable], m[:, usable], pos_weight=torch.tensor(pw))
            sel = (m > 0)
            l_amp = F.mse_loss(u[sel], ut_b[sel]) if sel.any() else s.sum() * 0
            loss = l_sel + a.w_amp * l_amp
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(planner.parameters(), 1.0); opt.step(); sched.step()
            run += float(loss.detach()) * len(idx)
        ev = evaluate(); ev.update(epoch=ep + 1, train_loss=run / len(xt)); hist.append(ev)
        mark = ""
        if ev["f1"] > best:
            best, best_state, mark = ev["f1"], {k_: v.clone() for k_, v in planner.state_dict().items()}, "  *best"
        print(f"epoch {ep + 1:>3}/{a.epochs}  loss {ev['train_loss']:.4f}  val overlap with heuristic: "
              f"precision {ev['overlap_precision']:.3f} recall {ev['overlap_recall']:.3f} F1 {ev['f1']:.3f}  "
              f"({time.time() - t0:.0f}s){mark}")
    planner.load_state_dict(best_state)
    out = Path(ex["paths"]["out"]) / "module_b"; out.mkdir(parents=True, exist_ok=True)
    name = f"b2b_k{a.k}_s{a.seed}_c{ex['learned']['lam_c']:g}_t0_real{a.tag}"
    if (out / f"{name}.pt").exists():
        print(f"WARNING: overwriting {out / (name + '.pt')}")
    torch.save({"state": planner.state_dict(), "n_labels": 0, "k": a.k, "classes": list(classes), "method": "distilled"}, out / f"{name}.pt")
    (out / f"{name}_distill.json").write_text(json.dumps({"args": vars(a), "history": hist}, indent=2))
    print("saved", out / f"{name}.pt", f"(best val F1 vs heuristic {best:.3f})")
    print(f"next: python scripts/b_baselines.py --data-dir {a.data_dir} --k {a.k} --max-per-class {a.max_per_class} --n-val 80 --ckpt {out / (name + '.pt')}")


if __name__ == "__main__":
    main()
