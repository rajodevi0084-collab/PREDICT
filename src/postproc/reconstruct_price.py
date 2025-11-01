"""Utilities to reconstruct next-tick prices from model outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .tick_snap import snap_to_tick


def reconstruct(mid_price: float | np.ndarray, y_reg_hat: float | np.ndarray, tick_size: float) -> float | np.ndarray:
    mid_price_arr = np.asarray(mid_price, dtype=float)
    y_reg_arr = np.asarray(y_reg_hat, dtype=float)
    next_price = mid_price_arr * np.exp(y_reg_arr)
    snapped = snap_to_tick(next_price, tick_size)
    if np.isscalar(mid_price):
        return float(snapped)
    return snapped


def reconstruct_close(
    close_price: float | np.ndarray | pd.Series,
    y_reg_hat: float | np.ndarray | pd.Series,
) -> float | np.ndarray | pd.Series:
    """Return the next close estimate given a log-return prediction."""

    if isinstance(close_price, pd.Series):
        aligned = y_reg_hat
        if isinstance(aligned, pd.Series):
            aligned = aligned.reindex(close_price.index)
        y_reg_arr = np.asarray(aligned, dtype=float)
        next_close = close_price.to_numpy(dtype=float) * np.exp(y_reg_arr)
        return pd.Series(next_close, index=close_price.index, name=close_price.name)

    close_arr = np.asarray(close_price, dtype=float)
    y_reg_arr = np.asarray(y_reg_hat, dtype=float)
    result = close_arr * np.exp(y_reg_arr)
    if np.isscalar(close_price):
        return float(result)
    return result


__all__ = ["reconstruct", "reconstruct_close"]
