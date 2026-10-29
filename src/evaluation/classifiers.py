"""Independent evaluators (plan 7.1). Different families from Module A (3-block CNN) and T_c (plain CNN).

E_main : ResNetSmall   (residual CNN, ResNet-18 layout at reduced width)
E_alt  : PatchMixer    (MLP-Mixer style, no convolutions after patch embedding)

Both are trained from scratch by `train_evaluator` on common.sparse_data images (scoreboard dots plus
random Dynaphos subsets) and never see planner outputs. Their accuracy on held-out random-dot images
must be reported first (`dot_accuracy`) so a low score on planner outputs can be interpreted.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Block(nn.Module):
    def __init__(self, i, o, stride):
        super().__init__()
        self.c1 = nn.Conv2d(i, o, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(o)
        self.c2 = nn.Conv2d(o, o, 3, 1, 1, bias=False);      self.b2 = nn.BatchNorm2d(o)
        self.sc = None if (i == o and stride == 1) else nn.Sequential(nn.Conv2d(i, o, 1, stride, bias=False), nn.BatchNorm2d(o))

    def forward(self, x):
        y = self.b2(self.c2(F.relu(self.b1(self.c1(x)))))
        return F.relu(y + (x if self.sc is None else self.sc(x)))


class ResNetSmall(nn.Module):
    def __init__(self, n_classes: int, width: int = 16, in_pool: int = 2):
        super().__init__()
        self.in_pool = in_pool
        w = width
        self.stem = nn.Sequential(nn.Conv2d(1, w, 5, 2, 2, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True))
        cfg = [(w, w, 1), (w, 2 * w, 2), (2 * w, 4 * w, 2), (4 * w, 8 * w, 2)]
        self.layers = nn.Sequential(*[nn.Sequential(_Block(i, o, s), _Block(o, o, 1)) for i, o, s in cfg])
        self.fc = nn.Linear(8 * w, n_classes)

    def forward(self, x):
        if self.in_pool > 1:
            x = F.max_pool2d(x, self.in_pool)
        return self.fc(self.layers(self.stem(x)).mean((2, 3)))


class PatchMixer(nn.Module):
    def __init__(self, n_classes: int, image_size: int = 128, patch: int = 8, dim: int = 64, depth: int = 4):
        super().__init__()
        self.image_size, self.patch = image_size, patch
        n = (image_size // patch) ** 2
        self.embed = nn.Conv2d(1, dim, patch, patch)
        self.blocks = nn.ModuleList([nn.ModuleDict({
            "n1": nn.LayerNorm(dim), "tok": nn.Sequential(nn.Linear(n, 2 * n), nn.GELU(), nn.Linear(2 * n, n)),
            "n2": nn.LayerNorm(dim), "ch": nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))})
            for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, n_classes)

    def forward(self, x):
        if x.shape[-1] != self.image_size:
            x = F.adaptive_max_pool2d(x, self.image_size)
        z = self.embed(x).flatten(2).transpose(1, 2)                    # (B, n, dim)
        for b in self.blocks:
            z = z + b["tok"](b["n1"](z).transpose(1, 2)).transpose(1, 2)
            z = z + b["ch"](b["n2"](z))
        return self.fc(self.norm(z).mean(1))


@torch.no_grad()
def predict_logits(model: nn.Module, x, batch: int = 64, device="cpu") -> torch.Tensor:
    model.eval().to(device)
    x = torch.as_tensor(x, dtype=torch.float32)
    return torch.cat([model(x[i:i + batch].to(device)).cpu() for i in range(0, len(x), batch)])


def accuracy(model, x, y, **kw) -> float:
    return float((predict_logits(model, x, **kw).argmax(1).numpy() == np.asarray(y)).mean())


def train_evaluator(model: nn.Module, x_train, y_train, x_val, y_val, *, epochs: int = 15, batch: int = 32,
                    lr: float = 2e-3, seed: int = 1, device="cpu", log: Callable[[str], None] = print) -> nn.Module:
    """Train from scratch on sparse-dot data; keep the checkpoint with the best accuracy on VAL DOT images."""
    import copy
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * max(1, len(x_train) // batch))
    xt, yt = torch.as_tensor(x_train), torch.as_tensor(y_train)
    best, state = -1.0, None
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(xt))
        for i in range(0, len(xt) - batch + 1, batch):
            idx = perm[i:i + batch]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(xt[idx].to(device)), yt[idx].to(device)).backward()
            opt.step(); sched.step()
        acc = accuracy(model, x_val, y_val, device=device)
        log(f"[E {type(model).__name__}] epoch {ep:2d}  val dot accuracy {acc:.3f}")
        if acc > best:
            best, state = acc, copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    return model.eval()


def save_classifier(path, model: nn.Module, classes) -> None:
    """Checkpoint that remembers WHICH classes the output units mean (used for models trained on your own images)."""
    torch.save({"state": model.state_dict(), "classes": list(classes), "arch": type(model).__name__}, path)


def load_classifier(path, make: Callable[[int], nn.Module]):
    """Returns (model in eval mode, classes). `make(n_classes)` builds the empty network."""
    ck = torch.load(path, map_location="cpu")
    model = make(len(ck["classes"]))
    model.load_state_dict(ck["state"])
    return model.eval(), list(ck["classes"])
