"""CUSUM event sampler for volatility-normalised moves."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from pg.utils.logging_and_seed import get_logger


_LOG = get_logger(__name__)


def cusum_filter(
    df: pd.DataFrame,
    ts_col: str,
    price_col: str,
    sigma: pd.Series,
    k_sigma: float,
    drift: float,
    min_spacing_minutes: int,
) -> pd.DatetimeIndex:
    """Return event timestamps where CUSUM exceeds k_sigma * sigma."""

    if df.empty:
        raise ValueError("input dataframe is empty")
    if ts_col not in df.columns:
        raise KeyError(f"timestamp column '{ts_col}' missing")
    if price_col not in df.columns:
        raise KeyError(f"price column '{price_col}' missing")
    if "symbol" not in df.columns:
        raise KeyError("dataframe must contain 'symbol'")
    if drift < 0:
        raise ValueError("drift must be non-negative")
    if k_sigma <= 0:
        raise ValueError("k_sigma must be positive")
    if min_spacing_minutes < 0:
        raise ValueError("min_spacing_minutes must be non-negative")
    if not sigma.index.equals(df.index):
        raise ValueError("sigma series must align with dataframe index")
    if not isinstance(df[ts_col].dtype, pd.DatetimeTZDtype):
        raise TypeError("timestamp column must be timezone-aware")

    events: List[pd.Timestamp] = []

    for symbol, group in df.groupby("symbol", sort=False):
        ts = group[ts_col]
        if not ts.is_monotonic_increasing:
            raise ValueError(f"timestamps must be sorted for symbol {symbol}")
        prices = group[price_col]
        if (prices <= 0).any():
            raise ValueError("prices must be positive for log returns")

        local_sigma = sigma.loc[group.index]
        threshold = k_sigma * local_sigma
        log_prices = np.log(prices)
        pos_sum = 0.0
        neg_sum = 0.0
        last_event_time: pd.Timestamp | None = None

        for i in range(1, len(group)):
            ret = log_prices.iloc[i] - log_prices.iloc[i - 1]
            pos_sum = max(0.0, pos_sum + ret - drift)
            neg_sum = min(0.0, neg_sum + ret + drift)
            ts_i = ts.iloc[i]
            h = threshold.iloc[i]

            triggered = False
            if pos_sum > h:
                triggered = True
            elif neg_sum < -h:
                triggered = True

            if triggered:
                if last_event_time is None or (
                    (ts_i - last_event_time).total_seconds() / 60.0 >= min_spacing_minutes
                ):
                    events.append(ts_i)
                    last_event_time = ts_i
                    _LOG.debug("CUSUM event detected symbol=%s ts=%s", symbol, ts_i)
                pos_sum = 0.0
                neg_sum = 0.0

    event_index = pd.DatetimeIndex(events)
    if event_index.empty:
        return pd.DatetimeIndex([], tz="Asia/Kolkata")
    event_index = event_index.sort_values()
    if not event_index.is_monotonic_increasing:
        raise AssertionError("event timestamps must be non-decreasing")
    if len(event_index.unique()) != len(event_index):
        raise AssertionError("event timestamps must be unique")
    if event_index.tz is None:
        event_index = event_index.tz_localize("Asia/Kolkata")
    else:
        event_index = event_index.tz_convert("Asia/Kolkata")
    return event_index
