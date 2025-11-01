"""Residual-based conformal prediction for regression."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ConformalInterval:
    alpha: float
    qhat: float

    def save(self, path: str | Path) -> None:
        Path(path).write_text(f"{self.alpha},{self.qhat}")

    @classmethod
    def load(cls, path: str | Path) -> "ConformalInterval":
        raw = Path(path).read_text().strip()
        alpha_str, qhat_str = raw.split(",")
        return cls(alpha=float(alpha_str), qhat=float(qhat_str))


def fit_conformal_interval(residuals: np.ndarray, coverage: float) -> ConformalInterval:
    residuals = np.asarray(residuals, dtype=float)
    alpha = 1 - coverage
    qhat = float(np.quantile(residuals, 1 - alpha, method="higher"))
    return ConformalInterval(alpha=alpha, qhat=qhat)


def apply_conformal(predictions: np.ndarray, calibration: ConformalInterval) -> np.ndarray:
    predictions = np.asarray(predictions, dtype=float)
    lower = predictions - calibration.qhat
    upper = predictions + calibration.qhat
    return np.vstack([lower, upper]).T


class ConformalRegressor:
    def __init__(self, coverage: float = 0.9) -> None:
        self.coverage = coverage
        self.alpha = 1 - coverage
        self.residuals_ = np.array([0.0])

    def fit(self, residuals: np.ndarray) -> "ConformalRegressor":
        self.residuals_ = np.asarray(residuals, dtype=float)
        return self

    def interval(self, preds: np.ndarray) -> np.ndarray:
        if self.residuals_.size == 0:
            raise RuntimeError("Conformal regressor is not fitted")
        qhat = float(np.quantile(np.abs(self.residuals_), 1 - self.alpha, method="higher"))
        preds = np.asarray(preds, dtype=float)
        lower = preds - qhat
        upper = preds + qhat
        return np.column_stack([lower, upper])


__all__ = ["ConformalInterval", "fit_conformal_interval", "apply_conformal", "ConformalRegressor"]
