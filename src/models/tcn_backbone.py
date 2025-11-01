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
    """Temporal convolutional network with two task-specific heads."""

    def __init__(
        self,
        n_features: int,
        lookback: int,
        channels: int,
        blocks: int,
        kernel: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers = []
        in_channels = n_features
        for i in range(blocks):
            dilation = 2 ** i
            layers.append(_tcn_block(in_channels, channels, kernel, dilation, dropout))
            in_channels = channels
        self.backbone = nn.Sequential(*layers)
        self.head_cls = nn.Sequential(nn.Conv1d(channels, 3, kernel_size=1))
        self.head_reg = nn.Sequential(nn.Conv1d(channels, 1, kernel_size=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, features, lookback)
        latent = self.backbone(x)
        cls = self.head_cls(latent).mean(dim=-1)
        reg = self.head_reg(latent).mean(dim=-1)
        return cls, reg.squeeze(1)


__all__ = ["TCN"]
