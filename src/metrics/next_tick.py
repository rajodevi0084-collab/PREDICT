"""Metrics tailored for next-tick predictions."""

from __future__ import annotations

import numpy as np
from scipy import stats


def tick_mae(actual: np.ndarray, pred: np.ndarray, tick_size: float) -> float:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.abs(actual - pred).mean() / tick_size)


def hit_rate(sign_actual: np.ndarray, sign_pred: np.ndarray) -> float:
    sign_actual = np.asarray(sign_actual, dtype=int)
    sign_pred = np.asarray(sign_pred, dtype=int)
    mask = sign_actual != 0
    if mask.sum() == 0:
        return 0.0
    return float((sign_actual[mask] == sign_pred[mask]).mean())


def brier_3way(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    eye = np.eye(3)
    targets = eye[labels]
    return float(((probs - targets) ** 2).sum(axis=1).mean())


def dm_test(series_model: np.ndarray, series_benchmark: np.ndarray) -> float:
    diff = np.asarray(series_model) - np.asarray(series_benchmark)
    return float(stats.ttest_1samp(diff, popmean=0.0, alternative="less").pvalue)


__all__ = ["tick_mae", "hit_rate", "brier_3way", "dm_test"]
