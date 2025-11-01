"""Validation helpers aimed at avoiding target leakage in time-series tasks."""

from __future__ import annotations

from typing import Callable, Iterable

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


def shuffle_target_sanity_check(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    tolerance: float = 0.05,
    n_permutations: int = 10,
    random_state: int | None = 0,
) -> tuple[float, np.ndarray]:
    """Evaluate metric stability under target shuffling.

    Returns the baseline metric and the array of scores obtained after shuffling
    ``y_true``. If the shuffled scores exhibit systematic skill, an assertion is
    raised. This guard helps reveal inadvertent leakage or data misalignment.
    """

    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have identical shape")

    baseline = metric(y_true, y_pred)

    rng = np.random.default_rng(random_state)
    perm_scores = []
    for _ in range(n_permutations):
        permuted = rng.permutation(y_true)
        score = metric(permuted, y_pred)
        perm_scores.append(score)

    perm_scores_arr = np.asarray(perm_scores, dtype=float)
    if np.nanmean(np.abs(perm_scores_arr)) > tolerance:
        raise AssertionError(
            "Shuffle-target sanity check failed: shuffled labels retained signal",
        )

    return baseline, perm_scores_arr


__all__ = ["assert_no_future_features", "shuffle_target_sanity_check"]
