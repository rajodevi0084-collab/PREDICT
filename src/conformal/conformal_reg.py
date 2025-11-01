"""Conformal interval construction for regression outputs."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ConformalRegressor:
    coverage: float = 0.9
    residuals_: np.ndarray | None = None

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> "ConformalRegressor":
        residuals = np.abs(y_true - y_pred)
        self.residuals_ = np.sort(residuals)
        return self

    def interval(self, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.residuals_ is None:
            raise RuntimeError("ConformalRegressor must be fitted")
        alpha = 1 - self.coverage
        idx = int(np.ceil((1 - alpha) * len(self.residuals_))) - 1
        idx = max(idx, 0)
        q = self.residuals_[idx]
        lower = y_pred - q
        upper = y_pred + q
        return lower, upper


__all__ = ["ConformalRegressor"]
