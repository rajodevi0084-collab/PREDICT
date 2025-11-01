"""Classification calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TemperatureScaler:
    temperature: float = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        logits = np.asarray(logits, dtype=float)
        labels = np.asarray(labels, dtype=int)
        self.temperature = _grid_search_temperature(logits, labels)
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return logits / self.temperature

    def save(self, path: str | Path) -> None:
        Path(path).write_text(f"{self.temperature:.6f}\n", encoding="utf-8")


def _grid_search_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    best_temp = 1.0
    best_loss = np.inf
    for temp in np.linspace(0.5, 5.0, num=50):
        scaled = logits / temp
        loss = _nll(scaled, labels)
        if loss < best_loss:
            best_loss = loss
            best_temp = temp
    return best_temp


def _nll(logits: np.ndarray, labels: np.ndarray) -> float:
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    idx = (np.arange(len(labels)), labels)
    return float(-np.log(probs[idx]).mean())


__all__ = ["TemperatureScaler"]
