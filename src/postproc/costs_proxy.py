"""Cost proxy utilities for evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def spread_bp_proxy(close: pd.Series, k: float, window: int = 64) -> pd.Series:
    """Estimate spread cost in basis points using rolling volatility."""

    close = close.astype(float)
    log_returns = np.log(close / close.shift(1)).fillna(0.0)
    rolling_std = log_returns.rolling(window, min_periods=1).std(ddof=0)
    proxy = k * rolling_std * 1e4
    return proxy


__all__ = ["spread_bp_proxy"]
