"""Classification metric helpers for label gating."""
from __future__ import annotations

import numpy as np

from pg.utils.logging_and_seed import get_logger


_LOG = get_logger(__name__)


def _validate_inputs(p: np.ndarray, y: np.ndarray) -> None:
    if p.shape != y.shape:
        raise ValueError("probability and label arrays must have the same shape")
    if p.ndim != 1:
        raise ValueError("inputs must be one-dimensional")
    if not np.isfinite(p).all():
        raise ValueError("probabilities must be finite")
    if not np.isfinite(y).all():
        raise ValueError("labels must be finite")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be in [0, 1]")
    unique = np.unique(y)
    if not np.array_equal(unique, unique.astype(int)):
        raise ValueError("labels must be integers")
    if not set(unique).issubset({0, 1}):
        raise ValueError("labels must be binary (0 or 1)")


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    """Compute the Brier score for binary outcomes."""

    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    _validate_inputs(p, y)
    score = float(np.mean((p - y) ** 2))
    _LOG.info("Brier score=%.4f", score)
    return score


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def auc_binary(p: np.ndarray, y: np.ndarray) -> float:
    """Compute ROC-AUC, returning 0.5 for degenerate cases."""

    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    _validate_inputs(p, y.astype(float))

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        _LOG.warning("AUC undefined with single-class labels; returning 0.5")
        return 0.5

    ranks = _average_ranks(p)
    rank_sum_pos = ranks[y == 1].sum()
    u_stat = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    auc = u_stat / (n_pos * n_neg)
    return float(auc)


def calibration_summary(p: np.ndarray, y: np.ndarray) -> None:
    """Print quick calibration diagnostics for debugging."""

    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    _validate_inputs(p, y)

    summary = {
        "mean_label": float(y.mean()),
        "mean_prob": float(p.mean()),
        "std_prob": float(p.std()),
        "p05": float(np.percentile(p, 5)),
        "p50": float(np.percentile(p, 50)),
        "p95": float(np.percentile(p, 95)),
    }
    _LOG.info("Calibration summary: %s", summary)
    print(summary)
