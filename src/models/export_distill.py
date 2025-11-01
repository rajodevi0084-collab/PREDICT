"""Teacher-to-student distillation utilities."""

"""Teacher-to-student distillation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from torch import nn


@dataclass
class DistillationConfig:
    temperature: float = 1.0
    epochs: int = 5
    lr: float = 1e-3


def distill_teacher(
    student: nn.Module,
    *,
    teacher_logits: torch.Tensor,
    teacher_targets: torch.Tensor,
    config: DistillationConfig | None = None,
    output_dir: str | Path | None = None,
) -> nn.Module:
    """Perform a simple distillation run and optionally persist the model."""

    config = config or DistillationConfig()
    student = student.train()

    optimizer = torch.optim.Adam(student.parameters(), lr=config.lr)
    loss_fn = nn.MSELoss()

    for _ in range(config.epochs):
        optimizer.zero_grad()
        preds = student(teacher_logits)[1]  # type: ignore[index]
        loss = loss_fn(preds, teacher_targets)
        loss.backward()
        optimizer.step()

    if output_dir is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(student.state_dict(), path / "student.pt")

    return student


__all__ = ["DistillationConfig", "distill_teacher"]
