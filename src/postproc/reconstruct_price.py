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
        base = close_price.astype(float)
        aligned = y_reg_hat
        if isinstance(aligned, pd.Series):
            aligned = aligned.reindex(base.index)
        y_reg_arr = np.asarray(aligned, dtype=float)
        if not np.isfinite(y_reg_arr).all():
            raise AssertionError("Regression predictions must be finite")
        next_close = base.to_numpy(dtype=float) * np.exp(y_reg_arr)
        if not np.isfinite(next_close).all():
            raise AssertionError("Reconstructed close contains non-finite values")
        return pd.Series(next_close, index=base.index, name=base.name)

    close_arr = np.asarray(close_price, dtype=float)
    y_reg_arr = np.asarray(y_reg_hat, dtype=float)
    if not np.isfinite(close_arr).all() or not np.isfinite(y_reg_arr).all():
        raise AssertionError("Inputs to reconstruct_close must be finite")
    result = close_arr * np.exp(y_reg_arr)
    if np.isscalar(close_price):
        if not np.isfinite(result):
            raise AssertionError("Reconstructed close is not finite")
        return float(result)
    if not np.isfinite(result).all():
        raise AssertionError("Reconstructed close contains non-finite values")
    return result


__all__ = ["reconstruct", "reconstruct_close"]
