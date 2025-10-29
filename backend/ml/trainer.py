"""Training utilities for temporal transformer models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from backend.ml.feature_engineer import FeatureSpec, make_features
from backend.services.reporter import Reporter


@dataclass
class TrainerConfig:
    """Hyper-parameters governing the optimisation procedure."""

    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    max_epochs: int = 40
    patience: int = 5
    num_splits: int = 3
    embargo: int = 16
    regression_weight: float = 1.0
    smooth_l1_beta: float = 0.1
    coverage_targets: Sequence[float] = (0.5, 0.75, 0.9)


class _TemporalDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """PyTorch dataset providing tensorised features and supervision targets."""

    def __init__(
        self,
        X: pd.DataFrame,
        y_cls: pd.Series,
        y_reg: pd.Series,
        indices: Sequence[int],
    ) -> None:
        super().__init__()
        if not len(indices):
            raise ValueError("Dataset indices must not be empty")
        features = X.iloc[indices].to_numpy(dtype=np.float32, copy=True)
        cls_targets = y_cls.iloc[indices].to_numpy(dtype=np.int64, copy=True)
        reg_targets = y_reg.iloc[indices].to_numpy(dtype=np.float32, copy=True)

        self._features = torch.from_numpy(features).unsqueeze(1)  # (batch, seq=1, feat)
        self._cls_targets = torch.from_numpy(cls_targets)
        self._reg_targets = torch.from_numpy(reg_targets)

    def __len__(self) -> int:
        return self._features.shape[0]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        return (
            self._features[idx],
            self._cls_targets[idx],
            self._reg_targets[idx],
        )


class Trainer:
    """High-level training loop for temporal models with walk-forward splits."""

    def __init__(
        self,
        model: nn.Module,
        feature_spec: FeatureSpec,
        *,
        reporter: Reporter,
        run_id: str,
        config: Optional[TrainerConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.feature_spec = feature_spec
        self.reporter = reporter
        self.run_id = run_id
        self.config = config or TrainerConfig()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model.to(self.device)
        self._num_classes = getattr(getattr(model, "config", None), "num_classes", 3)
        self._reg_weight = self.config.regression_weight
        self._coverage_targets = tuple(self.config.coverage_targets)
        self._checkpoint_dir = Path("artifacts") / "models" / self.run_id
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def run(self, frame: pd.DataFrame) -> dict[str, float]:
        """Execute the walk-forward training loop on ``frame``."""

        X, y_cls, y_reg, meta = make_features(frame, self.feature_spec)
        if X.empty:
            raise ValueError("No rows available after feature engineering. Check input data.")

        order = meta.sort_values("timestamp").index.to_numpy()
        splits = self._build_splits(order)
        self.reporter.publish(
            self.run_id,
            event="training_started",
            payload={"splits": len(splits), "samples": int(len(order))},
        )

        best_score = math.inf
        best_state: Optional[dict[str, Tensor]] = None
        last_metrics: dict[str, float] = {}

        for split_idx, (train_idx, val_idx) in enumerate(splits, start=1):
            train_loader = self._build_dataloader(X, y_cls, y_reg, train_idx, shuffle=True)
            val_loader = self._build_dataloader(X, y_cls, y_reg, val_idx, shuffle=False)

            metrics = self._train_split(split_idx, train_loader, val_loader)
            last_metrics = metrics

            if metrics["val_loss"] < best_score:
                best_score = metrics["val_loss"]
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                torch.save({"model_state": best_state}, self._checkpoint_dir / "best.pt")
                self.reporter.publish(
                    self.run_id,
                    event="checkpoint_saved",
                    payload={"split": split_idx, "val_loss": best_score},
                )

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.reporter.publish(
            self.run_id,
            event="training_finished",
            payload=last_metrics,
        )
        return last_metrics

    # ---------------------------------------------------------------------
    # Helper methods
    # ---------------------------------------------------------------------
    def _build_dataloader(
        self,
        X: pd.DataFrame,
        y_cls: pd.Series,
        y_reg: pd.Series,
        indices: Sequence[int],
        *,
        shuffle: bool,
    ) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
        dataset = _TemporalDataset(X, y_cls, y_reg, indices)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    def _train_split(
        self,
        split_idx: int,
        train_loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
        val_loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    ) -> dict[str, float]:
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=max(self.config.max_epochs, 1))
        scaler = GradScaler(enabled=self.device.type == "cuda")
        bce_loss = nn.BCEWithLogitsLoss()
        smooth_l1 = nn.SmoothL1Loss(beta=self.config.smooth_l1_beta)

        best_val = math.inf
        patience = 0
        best_metrics: dict[str, float] = {}

        for epoch in range(1, self.config.max_epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            for features, cls_target, reg_target in train_loader:
                features = features.to(self.device)
                cls_target = cls_target.to(self.device)
                reg_target = reg_target.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                    outputs = self.model(features)
                    logits = outputs["logits_cls"]
                    reg_pred = outputs["y_reg"].squeeze(-1)
                    target_bce = F.one_hot((cls_target + 1).clamp(0, self._num_classes - 1), num_classes=self._num_classes)
                    target_bce = target_bce.to(dtype=logits.dtype)
                    bce = bce_loss(logits, target_bce)
                    reg = smooth_l1(reg_pred, reg_target)
                    loss = bce + self._reg_weight * reg

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.detach().item()
                num_batches += 1

            scheduler.step()
            train_loss = epoch_loss / max(num_batches, 1)

            val_metrics = self._evaluate(val_loader, bce_loss, smooth_l1)
            val_loss = val_metrics["val_loss"]
            val_metrics.update({"split": split_idx, "epoch": epoch, "train_loss": train_loss})
            self.reporter.publish(
                self.run_id,
                event="epoch_end",
                payload=val_metrics,
            )

            if val_loss < best_val:
                best_val = val_loss
                patience = 0
                best_metrics = val_metrics
                torch.save({"model_state": self.model.state_dict()}, self._checkpoint_dir / f"split{split_idx}_epoch{epoch}.pt")
            else:
                patience += 1
                if patience >= self.config.patience:
                    break

        return best_metrics

    def _evaluate(
        self,
        loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
        bce_loss: nn.Module,
        smooth_l1: nn.Module,
    ) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        logits_list: list[Tensor] = []
        reg_pred_list: list[Tensor] = []
        cls_targets: list[Tensor] = []
        reg_targets: list[Tensor] = []

        with torch.no_grad():
            for features, cls_target, reg_target in loader:
                features = features.to(self.device)
                cls_target = cls_target.to(self.device)
                reg_target = reg_target.to(self.device)

                outputs = self.model(features)
                logits = outputs["logits_cls"]
                reg_pred = outputs["y_reg"].squeeze(-1)
                target_bce = F.one_hot((cls_target + 1).clamp(0, self._num_classes - 1), num_classes=self._num_classes)
                target_bce = target_bce.to(dtype=logits.dtype)
                bce = bce_loss(logits, target_bce)
                reg = smooth_l1(reg_pred, reg_target)
                loss = bce + self._reg_weight * reg

                total_loss += loss.detach().item()
                num_batches += 1

                logits_list.append(logits.detach().cpu())
                reg_pred_list.append(reg_pred.detach().cpu())
                cls_targets.append(cls_target.detach().cpu())
                reg_targets.append(reg_target.detach().cpu())

        logits_tensor = torch.cat(logits_list, dim=0) if logits_list else torch.empty((0, self._num_classes))
        reg_pred_tensor = torch.cat(reg_pred_list, dim=0) if reg_pred_list else torch.empty((0,))
        cls_tensor = torch.cat(cls_targets, dim=0) if cls_targets else torch.empty((0,), dtype=torch.long)
        reg_tensor = torch.cat(reg_targets, dim=0) if reg_targets else torch.empty((0,))

        metrics = self._compute_metrics(logits_tensor, reg_pred_tensor, cls_tensor, reg_tensor)
        metrics["val_loss"] = total_loss / max(num_batches, 1)
        return metrics

    def _compute_metrics(
        self,
        logits: Tensor,
        reg_pred: Tensor,
        cls_targets: Tensor,
        reg_targets: Tensor,
    ) -> dict[str, float]:
        if logits.numel() == 0:
            return {"val_loss": float("nan")}

        cls_idx = (cls_targets + 1).clamp(0, self._num_classes - 1).long()
        temperature = self._calibrate_temperature(logits, cls_idx)
        calibrated = logits / temperature
        probs = torch.softmax(calibrated, dim=-1)

        pred_idx = torch.argmax(probs, dim=-1)
        accuracy = (pred_idx == cls_idx).float().mean().item()
        mae = torch.abs(reg_pred - reg_targets).mean().item() if reg_pred.numel() else float("nan")
        rmse = torch.sqrt(F.mse_loss(reg_pred, reg_targets)).item() if reg_pred.numel() else float("nan")

        metrics: dict[str, float] = {
            "accuracy": float(accuracy),
            "mae": float(mae),
            "rmse": float(rmse),
            "temperature": float(temperature),
        }

        for coverage in self._coverage_targets:
            acc_cov, actual_cov = self._accuracy_at_coverage(probs, cls_idx, coverage)
            metrics[f"acc@{int(coverage * 100)}"] = float(acc_cov)
            metrics[f"coverage@{int(coverage * 100)}"] = float(actual_cov)

        return metrics

    def _accuracy_at_coverage(self, probs: Tensor, targets: Tensor, coverage: float) -> tuple[float, float]:
        confidences = probs.max(dim=-1).values
        tau = self._select_tau_for_coverage(confidences, coverage)
        mask = confidences >= tau
        actual_cov = mask.float().mean().item() if mask.numel() else 0.0
        if not mask.any():
            return 0.0, actual_cov
        preds = probs.argmax(dim=-1)
        acc = (preds[mask] == targets[mask]).float().mean().item()
        return acc, actual_cov

    def _calibrate_temperature(self, logits: Tensor, targets: Tensor) -> float:
        if logits.numel() == 0:
            return 1.0
        logits = logits.to(torch.float64)
        targets = targets.to(torch.long)
        log_temp = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_temp], lr=0.25, max_iter=25, line_search_fn="strong_wolfe")

        def closure() -> Tensor:
            optimizer.zero_grad()
            temp = torch.exp(log_temp)
            scaled = logits / temp
            loss = F.nll_loss(F.log_softmax(scaled, dim=-1), targets)
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
        except RuntimeError:
            return 1.0
        temperature = float(torch.exp(log_temp).item())
        if not math.isfinite(temperature) or temperature <= 1e-3:
            return 1.0
        return temperature

    def _select_tau_for_coverage(self, confidences: Tensor, coverage: float) -> float:
        if confidences.numel() == 0:
            return 1.0
        coverage = float(np.clip(coverage, 0.0, 1.0))
        if coverage <= 0:
            return float(confidences.max().item())
        sorted_conf, _ = torch.sort(confidences, descending=True)
        k = max(int(math.ceil(len(sorted_conf) * coverage)) - 1, 0)
        return float(sorted_conf[k].item())

    def _build_splits(self, ordered_indices: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        total = len(ordered_indices)
        if total < 2:
            raise ValueError("At least two samples are required to build splits")

        splits: list[tuple[np.ndarray, np.ndarray]] = []
        window = max(total // (self.config.num_splits + 1), 1)
        embargo = max(int(self.config.embargo), 0)

        for split in range(self.config.num_splits):
            train_end = window * (split + 1)
            if train_end <= 1:
                continue
            train_end = min(train_end, total - 1)
            val_start = min(train_end + embargo, total - 1)
            val_end = min(val_start + window, total)
            if val_start >= val_end:
                break
            train_indices = ordered_indices[:train_end]
            val_indices = ordered_indices[val_start:val_end]
            if len(val_indices) == 0:
                break
            splits.append((train_indices, val_indices))

        if not splits:
            cutoff = max(total - window, 1)
            splits.append((ordered_indices[:cutoff], ordered_indices[cutoff:]))

        return splits


__all__ = ["Trainer", "TrainerConfig"]
