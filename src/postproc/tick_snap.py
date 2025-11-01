"""Tick size utilities for post-processing predictions."""

from __future__ import annotations

import numpy as np


def snap_to_tick(price: float | np.ndarray, tick_size: float) -> float | np.ndarray:
    price_arr = np.asarray(price, dtype=float)
    snapped = np.round(price_arr / tick_size) * tick_size
    if np.isscalar(price):
        return float(snapped)
    return snapped


def clip_log_return(y: np.ndarray, k_sigma: float, sigma_train: float) -> np.ndarray:
    limit = k_sigma * sigma_train
    return np.clip(y, -limit, limit)


__all__ = ["snap_to_tick", "clip_log_return"]
