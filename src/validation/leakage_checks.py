"""Validation helpers aimed at avoiding target leakage in time-series tasks."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _coerce_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    if isinstance(index, pd.MultiIndex):
        for name in index.names[::-1]:
            if name is None:
                continue
            values = index.get_level_values(name)
            if pd.api.types.is_datetime64_any_dtype(values):
                return pd.DatetimeIndex(values, tz="UTC" if values.tz is None else values.tz)
        # fall back to the last level
        values = index.get_level_values(-1)
        if pd.api.types.is_datetime64_any_dtype(values):
            return pd.DatetimeIndex(values, tz="UTC" if values.tz is None else values.tz)
        raise TypeError("Unable to infer timestamp level from MultiIndex")
    if not pd.api.types.is_datetime64_any_dtype(index):
        index = pd.to_datetime(index, errors="raise", utc=True)
    return pd.DatetimeIndex(index)


def assert_no_future_features(X: pd.DataFrame, y_index: pd.Index) -> None:
    """Ensure that feature timestamps do not extend beyond label timestamps."""

    feature_index = _coerce_datetime_index(X.index)
    label_index = _coerce_datetime_index(y_index)

    if len(feature_index) < len(label_index):
        raise ValueError("Feature matrix must be at least as long as labels")

    # Align indices by position assuming features at t map to label at t
    aligned_features = feature_index[: len(label_index)]
    if (aligned_features > label_index).any():
        raise AssertionError("Detected future-looking features relative to labels")


def shuffle_target_sanity(
    X: pd.DataFrame | np.ndarray,
    y: Iterable[float],
    *,
    tolerance: float = 0.02,
    n_permutations: int = 10,
    random_state: int | None = 0,
) -> dict[str, float | np.ndarray]:
    """Ensure that shuffled targets yield near-zero skill.

    The routine fits a simple ridge regressor (or logistic scorer for binary
    targets) on the provided features and evaluates the signal strength for the
    true targets and for multiple shuffled permutations. If the shuffled metric
    consistently exceeds ``tolerance``, the function raises ``AssertionError``.
    """

    if isinstance(X, pd.DataFrame):
        X_arr = X.to_numpy(dtype=float, copy=False)
    else:
        X_arr = np.asarray(X, dtype=float)

    if X_arr.ndim != 2:
        raise ValueError("Feature matrix must be 2-D")

    y_arr = np.asarray(list(y), dtype=float)
    if y_arr.ndim != 1:
        raise ValueError("Target array must be 1-D")
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("Feature rows must match target length")

    binary_mask = np.isin(np.unique(y_arr), [0.0, 1.0]).all() or np.isin(
        np.unique(y_arr), [-1.0, 0.0, 1.0]
    ).all()

    X_aug = np.column_stack([np.ones(X_arr.shape[0]), X_arr])

    def _ridge_solution(target: np.ndarray) -> np.ndarray:
        ridge = 1e-6 * np.eye(X_aug.shape[1])
        ridge[0, 0] = 0.0  # do not regularise bias
        gram = X_aug.T @ X_aug + ridge
        coef = np.linalg.solve(gram, X_aug.T @ target)
        return X_aug @ coef

    def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        if ss_tot <= 0:
            return 0.0
        ss_res = np.sum((y_true - y_pred) ** 2)
        return float(1.0 - ss_res / ss_tot)

    def _auc(y_true: np.ndarray, scores: np.ndarray) -> float:
        # Map {-1,0,1} -> {0,0,1}
        y_bin = (y_true > 0).astype(float)
        order = np.argsort(scores)
        ranked_y = y_bin[order]
        cum_pos = np.cumsum(ranked_y)
        total_pos = cum_pos[-1]
        total_neg = len(y_bin) - total_pos
        if total_pos == 0 or total_neg == 0:
            return 0.5
        # Mann–Whitney U formulation of AUC.
        rank_sum = np.sum(np.nonzero(ranked_y)[0] + 1)
        auc = (rank_sum - total_pos * (total_pos + 1) / 2) / (total_pos * total_neg)
        return float(auc)

    def _score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if binary_mask:
            logits = y_pred
            probs = 1.0 / (1.0 + np.exp(-logits))
            return _auc(y_true, probs)
        return _r2(y_true, y_pred)

    baseline_pred = _ridge_solution(y_arr)
    baseline_score = _score(y_arr, baseline_pred)

    rng = np.random.default_rng(random_state)
    perm_scores = []
    for _ in range(n_permutations):
        permuted = rng.permutation(y_arr)
        perm_pred = _ridge_solution(permuted)
        perm_scores.append(_score(permuted, perm_pred))

    perm_scores_arr = np.asarray(perm_scores, dtype=float)
    if np.nanmean(np.abs(perm_scores_arr - (0.5 if binary_mask else 0.0))) > tolerance:
        raise AssertionError("Shuffle-target sanity check failed: residual signal detected")

    return {
        "baseline_score": baseline_score,
        "shuffle_scores": perm_scores_arr,
        "metric": "auc" if binary_mask else "r2",
    }


__all__ = ["assert_no_future_features", "shuffle_target_sanity"]
