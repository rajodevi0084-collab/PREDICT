"""Lightweight Temporal Convolutional Network backbone."""

from __future__ import annotations

import torch
from torch import nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple slice
        if self.chomp_size == 0:
            return x
        return x[..., : -self.chomp_size]


def _tcn_block(in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> nn.Sequential:
    padding = (kernel_size - 1) * dilation
    layers = [
        nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
        Chomp1d(padding),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
        Chomp1d(padding),
        nn.ReLU(),
        nn.Dropout(dropout),
    ]
    return nn.Sequential(*layers)


class TCN(nn.Module):
    """Temporal convolutional network with detachable heads."""

    def __init__(
        self,
        n_features: int,
        lookback: int,
        channels: int,
        blocks: int,
        kernel: int,
        dropout: float,
        *,
        n_classes: int = 3,
        weight_decay: float | None = None,
    ) -> None:
        super().__init__()
        self.lookback = lookback
        self.channels = channels
        self.blocks = blocks
        self.kernel = kernel
        self.dropout = dropout
        self.n_classes = n_classes
        self.weight_decay = weight_decay

        layers = []
        in_channels = n_features
        for i in range(blocks):
            dilation = 2 ** i
            layers.append(_tcn_block(in_channels, channels, kernel, dilation, dropout))
            in_channels = channels
        self.backbone = nn.Sequential(*layers)
        self.head_cls = nn.Conv1d(channels, n_classes, kernel_size=1)
        self.head_reg = nn.Conv1d(channels, 1, kernel_size=1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("Input tensor must be (batch, features, lookback)")
        return self.backbone(x)

    def forward_cls(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.forward_features(x)
        logits = self.head_cls(latent).mean(dim=-1)
        return logits

    def forward_reg(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.forward_features(x)
        reg = self.head_reg(latent).mean(dim=-1)
        return reg.squeeze(-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_cls(x), self.forward_reg(x)

    def parameter_groups(self) -> list[dict[str, object]]:
        if self.weight_decay is None:
            return [{"params": self.parameters()}]
        return [
            {"params": self.backbone.parameters(), "weight_decay": self.weight_decay},
            {"params": self.head_cls.parameters(), "weight_decay": 0.0},
            {"params": self.head_reg.parameters(), "weight_decay": 0.0},
        ]


__all__ = ["TCN"]
