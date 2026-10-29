"""T_c: the frozen TRAINING classifier used by L_task (plan 6.3).

Deliberately a different architecture (plain VGG-style CNN) from Module A (3-block GAP CNN) and from the
evaluators of Module E (ResNet / MLP-Mixer), trained with its own seed on sparse-dot images
(common.sparse_data). Its accuracy is NOT evidence of recognisability: that is Module E's job.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlainCNN(nn.Module):
    def __init__(self, n_classes: int, width: int = 16, in_pool: int = 2):
        super().__init__()
        self.in_pool = in_pool
        w = width
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.features = nn.Sequential(blk(1, w), blk(w, 2 * w), blk(2 * w, 4 * w), blk(4 * w, 4 * w))
        self.fc = nn.Linear(4 * w, n_classes)

    def forward(self, x):
        if self.in_pool > 1:
            x = F.max_pool2d(x, self.in_pool)          # keep sparse-dot peaks while shrinking the image
        return self.fc(self.features(x).mean((2, 3)))


def freeze(model: nn.Module) -> nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
