"""Regression calibration via simple affine transformation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class LinearCalibration:
    a: float = 1.0
    b: float = 0.0

    def fit(self, y_pred: np.ndarray, y_true: np.ndarray) -> "LinearCalibration":
        X = np.column_stack([y_pred, np.ones_like(y_pred)])
        params, *_ = np.linalg.lstsq(X, y_true, rcond=None)
        self.a, self.b = params
        return self

    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        return self.a * y_pred + self.b

    def save(self, path: str | Path) -> None:
        Path(path).write_text(f"{self.a:.6f},{self.b:.6f}\n", encoding="utf-8")


__all__ = ["LinearCalibration"]
