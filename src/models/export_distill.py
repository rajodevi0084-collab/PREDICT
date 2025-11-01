"""Distillation utilities for exporting student models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import torch
from torch import nn


@dataclass
class DistillationConfig:
    loss_weights: Dict[str, float]
    temperature: float = 2.0
    epochs: int = 1
    lr: float = 1e-3
    quantize_int8: bool = False


def distill_teacher_student(
    teacher: nn.Module,
    student: nn.Module,
    dataloader: Iterable[Dict[str, torch.Tensor]],
    *,
    config: DistillationConfig,
) -> nn.Module:
    """Train ``student`` to match ``teacher`` using soft targets."""

    teacher.eval()
    device = next(student.parameters()).device
    student.train()

    optimizer = torch.optim.AdamW(student.parameters(), lr=config.lr)
    mse = nn.MSELoss()
    kl = nn.KLDivLoss(reduction="batchmean")

    for _ in range(config.epochs):
        for batch in dataloader:
            inputs = batch["inputs"].to(device)
            hard_reg = batch.get("y_reg")
            if hard_reg is not None:
                hard_reg = hard_reg.to(device)

            with torch.no_grad():
                teacher_cls, teacher_reg = teacher(inputs)
                teacher_probs = torch.softmax(teacher_cls / config.temperature, dim=-1)

            student_cls, student_reg = student(inputs)
            student_log_probs = torch.log_softmax(student_cls / config.temperature, dim=-1)

            loss_soft_cls = kl(student_log_probs, teacher_probs) * (config.temperature**2)
            loss_soft_reg = mse(student_reg, teacher_reg)
            loss_hard_reg = mse(student_reg, hard_reg) if hard_reg is not None else 0.0

            loss = (
                config.loss_weights.get("soft_cls", 0.0) * loss_soft_cls
                + config.loss_weights.get("soft_reg", 0.0) * loss_soft_reg
                + config.loss_weights.get("hard_reg", 0.0) * loss_hard_reg
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    student.eval()
    if config.quantize_int8:
        student = torch.quantization.quantize_dynamic(student, {nn.Linear}, dtype=torch.qint8)
    return student


def export_student_model(student: nn.Module, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "student.pt"
    torch.save(student.state_dict(), path)
    return path


__all__ = ["DistillationConfig", "distill_teacher_student", "export_student_model"]
