"""Volatility estimators for event labeling."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from pg.utils.logging_and_seed import get_logger


_LOG = get_logger(__name__)


_SUPPORTED_METHODS = {"ewma", "parkinson", "garman_klass"}


def _validate_inputs(
    df: pd.DataFrame,
    method: str,
    span_minutes: int,
    min_sigma: float,
    ts_col: str,
    ohlc_cols: Iterable[str],
) -> None:
    if df.empty:
        raise ValueError("input dataframe is empty")
    if method not in _SUPPORTED_METHODS:
        raise ValueError(f"method must be one of {_SUPPORTED_METHODS}, got {method}")
    if span_minutes <= 0:
        raise ValueError("span_minutes must be positive")
    if min_sigma < 1e-6:
        raise ValueError("min_sigma must be at least 1e-6")
    if ts_col not in df.columns:
        raise KeyError(f"ts column '{ts_col}' missing from dataframe")
    missing = [c for c in ohlc_cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing OHLC columns: {missing}")
    if "symbol" not in df.columns:
        raise KeyError("dataframe must contain a 'symbol' column")

    ts = df[ts_col]
    if not isinstance(ts.dtype, pd.DatetimeTZDtype):
        raise TypeError("ts column must be timezone-aware datetime")
    if str(ts.dt.tz) != "Asia/Kolkata":
        raise ValueError("ts column must have timezone Asia/Kolkata")

    # enforce symbol groups sorted by time
    grouped = df.groupby("symbol")[ts_col]
    for symbol, series in grouped:
        if not series.is_monotonic_increasing:
            raise ValueError(f"timestamps must be non-decreasing within symbol {symbol}")


def compute_sigma(
    df: pd.DataFrame,
    method: str,
    span_minutes: int,
    min_sigma: float,
    ts_col: str,
    ohlc_cols: Iterable[str],
) -> pd.Series:
    """Compute realized volatility per bar using past data only.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing symbol, timestamp, and OHLC fields.
    method : str
        Volatility estimator identifier ("ewma", "parkinson", or "garman_klass").
    span_minutes : int
        Lookback span in minutes for smoothing.
    min_sigma : float
        Minimum volatility floor to avoid degenerate values.
    ts_col : str
        Name of the timestamp column.
    ohlc_cols : Iterable[str]
        Iterable specifying order of OHLC column names.

    Returns
    -------
    pd.Series
        Estimated sigma per row, aligned with the input index.
    """

    _validate_inputs(df, method, span_minutes, min_sigma, ts_col, ohlc_cols)

    open_col, high_col, low_col, close_col = tuple(ohlc_cols)

    series_list = []
    for symbol, group in df.groupby("symbol", sort=False):
        _LOG.debug("computing sigma for symbol=%s with %s", symbol, method)
        if method == "ewma":
            log_returns = np.log(group[close_col]).diff()
            sigma = log_returns.ewm(span=span_minutes, adjust=False).std(bias=False)
        elif method == "parkinson":
            hl_term = np.log(group[high_col] / group[low_col]) ** 2
            roll = hl_term.rolling(window=span_minutes, min_periods=2)
            sigma = np.sqrt(roll.mean() / (4.0 * np.log(2.0)))
        else:  # garman_klass
            log_hl = np.log(group[high_col] / group[low_col])
            log_co = np.log(group[close_col] / group[open_col])
            prev_close = group[close_col].shift(1)
            log_oc = np.log(group[open_col] / prev_close)
            var = (
                0.5 * log_hl**2
                - (2.0 * np.log(2.0) - 1.0) * log_co**2
                + log_oc**2
            )
            sigma = np.sqrt(var.rolling(window=span_minutes, min_periods=2).mean())

        sigma = sigma.ffill().fillna(min_sigma)
        sigma = sigma.clip(lower=min_sigma)
        series_list.append(sigma)

    sigma_all = pd.concat(series_list).sort_index()
    sigma_all.name = "sigma"
    if len(sigma_all) != len(df):
        raise AssertionError("sigma series length mismatch")
    return sigma_all.loc[df.index]
