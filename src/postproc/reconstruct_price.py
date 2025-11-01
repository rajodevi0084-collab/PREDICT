"""Utilities to reconstruct next-tick prices from model outputs."""

from __future__ import annotations

import numpy as np

from .tick_snap import snap_to_tick


def reconstruct(mid_price: float | np.ndarray, y_reg_hat: float | np.ndarray, tick_size: float) -> float | np.ndarray:
    mid_price_arr = np.asarray(mid_price, dtype=float)
    y_reg_arr = np.asarray(y_reg_hat, dtype=float)
    next_price = mid_price_arr * np.exp(y_reg_arr)
    snapped = snap_to_tick(next_price, tick_size)
    if np.isscalar(mid_price):
        return float(snapped)
    return snapped


__all__ = ["reconstruct"]
