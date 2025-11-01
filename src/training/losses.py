"""Loss function helpers for next-tick models."""

from __future__ import annotations

import torch
from torch import nn


def _parse_alpha(alpha: float | str | None) -> float | None:
    if alpha is None:
        return None
    if isinstance(alpha, str):
        if alpha.lower() == "balanced":
            return 1.0
        try:
            return float(alpha)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported alpha value '{alpha}'") from exc
    return float(alpha)


def make_focal(gamma: float = 2.0, alpha: float | str | None = None) -> nn.Module:
    """Return a focal loss module for multi-class logits."""

    alpha_value = _parse_alpha(alpha)

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

    return FocalLoss(gamma, alpha_value)


def _parse_delta(delta: float | str) -> float:
    if isinstance(delta, str):
        if delta.startswith("median*"):
            factor = float(delta.split("*")[1])
            return factor
        try:
            return float(delta)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported delta value '{delta}'") from exc
    return float(delta)


def huber(delta: float | str = 1.0) -> nn.Module:
    return nn.SmoothL1Loss(beta=_parse_delta(delta))


__all__ = ["make_focal", "huber"]
