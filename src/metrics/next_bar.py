"""Evaluation metrics tailored for next-bar forecasting."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def mae_price(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual - predicted)))


def brier_3way(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError("Probabilities must be of shape (n, 3)")
    one_hot = np.zeros_like(probs)
    label_to_index = {-1: 0, 0: 1, 1: 2}
    for i, label in enumerate(labels):
        if label not in label_to_index:
            raise ValueError("Labels must be in {-1,0,1}")
        one_hot[i, label_to_index[label]] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def hit_rate(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = actual != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.sign(actual[mask]) == np.sign(predicted[mask])))


def dm_test(model_errors: np.ndarray, benchmark_errors: np.ndarray) -> float:
    model_errors = np.asarray(model_errors, dtype=float)
    benchmark_errors = np.asarray(benchmark_errors, dtype=float)
    if model_errors.shape != benchmark_errors.shape:
        raise ValueError("Error vectors must share shape")
    diff = model_errors - benchmark_errors
    mean_diff = diff.mean()
    var_diff = np.var(diff, ddof=1)
    if var_diff == 0:
        return 0.0
    statistic = mean_diff / np.sqrt(var_diff / len(diff))
    # two-sided normal approximation
    p_value = 2 * (1 - 0.5 * (1 + erf(np.abs(statistic) / np.sqrt(2))))
    return float(p_value)


def crps_from_quantiles(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    interval = np.maximum(upper - lower, 1e-8)
    below = np.maximum(lower - actual, 0.0)
    above = np.maximum(actual - upper, 0.0)
    return float(np.mean((below**2 + above**2) / interval))


def erf(x: np.ndarray | float) -> np.ndarray | float:
    # Approximate error function via numerical approximation
    x = np.asarray(x)
    sign = np.sign(x)
    a1, a2, a3, a4, a5 = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429]
    p = 0.3275911
    t = 1.0 / (1.0 + p * np.abs(x))
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-(x**2))
    return sign * y


def lag_at_max_corr(
    r_pred: np.ndarray,
    r_true: np.ndarray,
    lags: Iterable[int] | range = range(-3, 4),
) -> int:
    r_pred = np.asarray(r_pred, dtype=float)
    r_true = np.asarray(r_true, dtype=float)
    best_lag = 0
    best_score = -np.inf
    for lag in lags:
        lag = int(lag)
        if lag < 0:
            offset = -lag
            pred_slice = r_pred[offset:]
            true_slice = r_true[: len(pred_slice)]
        elif lag > 0:
            pred_slice = r_pred[: len(r_pred) - lag]
            true_slice = r_true[lag : lag + len(pred_slice)]
        else:
            pred_slice = r_pred
            true_slice = r_true

        length = min(len(pred_slice), len(true_slice))
        if length < 2:
            continue
        pred_aligned = pred_slice[:length]
        true_aligned = true_slice[:length]
        if np.all(pred_aligned == pred_aligned[0]) or np.all(true_aligned == true_aligned[0]):
            continue
        corr = np.corrcoef(pred_aligned, true_aligned)[0, 1]
        if np.isnan(corr):
            continue
        score = abs(corr)
        if score > best_score or (np.isclose(score, best_score) and abs(lag) < abs(best_lag)):
            best_score = score
            best_lag = lag
    return int(best_lag)


def residual_acf1(residuals: np.ndarray) -> float:
    residuals = np.asarray(residuals, dtype=float)
    if residuals.size < 2:
        return 0.0
    residuals = residuals - np.nanmean(residuals)
    x = residuals[:-1]
    y = residuals[1:]
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 2:
        return 0.0
    x_masked = x[mask]
    y_masked = y[mask]
    denom = np.sqrt(np.sum(x_masked**2) * np.sum(y_masked**2))
    if denom == 0:
        return 0.0
    corr = float(np.dot(x_masked, y_masked) / denom)
    return corr


__all__ = [
    "mae_price",
    "brier_3way",
    "hit_rate",
    "dm_test",
    "crps_from_quantiles",
    "lag_at_max_corr",
    "residual_acf1",
]
