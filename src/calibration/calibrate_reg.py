"""Linear calibration for regression outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class LinearCalibrationModel:
    slope: float
    intercept: float

    def save(self, path: str | Path) -> None:
        payload = {"a": float(self.slope), "b": float(self.intercept)}
        Path(path).write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: str | Path) -> "LinearCalibrationModel":
        raw = Path(path).read_text().strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            slope_str, intercept_str = raw.split(",")
            return cls(slope=float(slope_str), intercept=float(intercept_str))

        slope = float(payload.get("a", payload.get("slope", 1.0)))
        intercept = float(payload.get("b", payload.get("intercept", 0.0)))
        return cls(slope=slope, intercept=intercept)

    def as_mapping(self) -> dict[str, float]:
        return {"a": float(self.slope), "b": float(self.intercept)}


def fit_linear_calibration(preds: np.ndarray, targets: np.ndarray) -> LinearCalibrationModel:
    preds = np.asarray(preds, dtype=float)
    targets = np.asarray(targets, dtype=float)
    X = np.column_stack([preds, np.ones_like(preds)])
    beta, _, _, _ = np.linalg.lstsq(X, targets, rcond=None)
    slope, intercept = beta
    return LinearCalibrationModel(slope=float(slope), intercept=float(intercept))


def apply_linear_calibration(preds: np.ndarray, calibration: LinearCalibrationModel) -> np.ndarray:
    return preds * calibration.slope + calibration.intercept


class LinearCalibration:
    def __init__(self, slope: float = 1.0, intercept: float = 0.0) -> None:
        self.model = LinearCalibrationModel(slope=slope, intercept=intercept)

    def fit(self, preds: np.ndarray, targets: np.ndarray) -> "LinearCalibration":
        self.model = fit_linear_calibration(preds, targets)
        return self

    def transform(self, preds: np.ndarray) -> np.ndarray:
        return apply_linear_calibration(np.asarray(preds, dtype=float), self.model)


__all__ = ["LinearCalibration", "fit_linear_calibration", "apply_linear_calibration"]
