"""Loss function helpers for next-tick models."""

from __future__ import annotations

import torch
from torch import nn


def make_focal(gamma: float = 2.0, alpha: float | None = None) -> nn.Module:
    """Return a focal loss module for multi-class logits."""

    class FocalLoss(nn.Module):
        def __init__(self, gamma: float, alpha: float | None) -> None:
            super().__init__()
            self.gamma = gamma
            self.alpha = alpha

        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            probs = torch.softmax(logits, dim=-1)
            targets_one_hot = nn.functional.one_hot(targets.long(), num_classes=probs.size(-1)).float()
            pt = (probs * targets_one_hot).sum(dim=-1)
            focal = (1 - pt) ** self.gamma
            if self.alpha is not None:
                focal = focal * self.alpha
            loss = -torch.log(pt.clamp_min(1e-8)) * focal
            return loss.mean()

    return FocalLoss(gamma, alpha)


def huber(delta: float = 1.0) -> nn.Module:
    return nn.SmoothL1Loss(beta=delta)


__all__ = ["make_focal", "huber"]
