"""Temperature scaling for classification logits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TemperatureCalibration:
    temperature: float

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(str(self.temperature))

    @classmethod
    def load(cls, path: str | Path) -> "TemperatureCalibration":
        value = float(Path(path).read_text().strip())
        return cls(temperature=value)


def _nll_temperature(temperature: float, logits: np.ndarray, labels: np.ndarray) -> float:
    if temperature <= 0:
        return np.inf
    scaled = logits / temperature
    log_probs = scaled - scaled.max(axis=1, keepdims=True)
    log_probs = log_probs - np.log(np.exp(log_probs).sum(axis=1, keepdims=True))
    idx = labels.astype(int)
    return -float(np.mean(log_probs[np.arange(len(labels)), idx]))


def temperature_scale(logits: np.ndarray, labels: np.ndarray) -> TemperatureCalibration:
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=int)
    candidates = np.exp(np.linspace(np.log(0.1), np.log(10.0), 50))
    losses = [_nll_temperature(t, logits, labels) for t in candidates]
    best_idx = int(np.nanargmin(losses))
    return TemperatureCalibration(temperature=float(candidates[best_idx]))


def apply_temperature(logits: np.ndarray, calibration: TemperatureCalibration) -> np.ndarray:
    return logits / calibration.temperature


class TemperatureScaler:
    def __init__(self, temperature: float = 1.0) -> None:
        self.calibration = TemperatureCalibration(temperature=temperature)

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        self.calibration = temperature_scale(logits, labels)
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return apply_temperature(np.asarray(logits, dtype=float), self.calibration)


__all__ = ["TemperatureCalibration", "temperature_scale", "apply_temperature", "TemperatureScaler"]
