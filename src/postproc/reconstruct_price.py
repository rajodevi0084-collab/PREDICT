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


def reconstruct_close(close_price: float | np.ndarray, y_reg_hat: float | np.ndarray) -> float | np.ndarray:
    """Return the next close estimate given a log-return prediction."""

    close_arr = np.asarray(close_price, dtype=float)
    y_reg_arr = np.asarray(y_reg_hat, dtype=float)
    return close_arr * np.exp(y_reg_arr)


__all__ = ["reconstruct", "reconstruct_close"]
