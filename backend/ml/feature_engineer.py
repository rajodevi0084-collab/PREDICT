"""Feature engineering utilities for tabular market data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    """Configuration of the feature engineering pipeline.

    Parameters
    ----------
    windows:
        Rolling window sizes (in number of rows) used for statistical
        aggregations.
    horizon:
        Number of steps ahead used to construct supervision targets.
    epsilon:
        Dead-zone threshold for the classification target. Absolute
        regression deltas smaller than ``epsilon`` are mapped to the neutral
        class.
    include_spread:
        When ``True`` and the input data contains bid/ask quotes, include
        spread derived features.
    include_ofi:
        When ``True`` and order book size columns are present, include an
        order-flow imbalance indicator.
    """

    windows: tuple[int, ...] = field(default_factory=lambda: (5, 15, 60))
    horizon: int = 5
    epsilon: float = 0.0
    include_spread: bool = False
    include_ofi: bool = False


def compute_labels(df: pd.DataFrame, spec: FeatureSpec) -> tuple[pd.Series, pd.Series]:
    """Return classification and regression targets for the supplied data.

    The regression target is the future price delta computed over ``horizon``
    rows of the input. A positive delta indicates that the closing price has
    moved higher. The classification label is derived from the delta using an
    ``epsilon`` insensitive band: movements whose absolute value is below the
    threshold are mapped to the neutral class, while the remaining observations
    are labelled as up (+1) or down (-1).

    Parameters
    ----------
    df:
        Input frame containing at least ``symbol`` and ``close`` columns.
    spec:
        Feature specification describing the horizon and epsilon.
    """

    if "close" not in df.columns:
        raise KeyError("Input dataframe must contain a 'close' column")
    if "symbol" not in df.columns:
        raise KeyError("Input dataframe must contain a 'symbol' column")

    horizon = spec.horizon
    if horizon <= 0:
        raise ValueError("FeatureSpec.horizon must be a positive integer")

    grouped = df.groupby("symbol", sort=False)
    future_close = grouped["close"].shift(-horizon)
    delta = future_close - df["close"]

    cls = pd.Series(0, index=df.index, dtype=int)
    cls[delta > spec.epsilon] = 1
    cls[delta < -spec.epsilon] = -1

    cls.name = "target_cls"
    delta.name = "target_reg"

    return cls, delta


def make_features(df: pd.DataFrame, spec: FeatureSpec | None = None) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Construct model-ready features and aligned supervision targets.

    The feature matrix contains:

    - log returns and rolling statistics
    - OHLC geometry descriptors (candle body, shadows, ranges)
    - volume and flow indicators
    - calendar embeddings using sinusoidal encodings
    - optional spread and order-flow imbalance features

    The resulting matrices are aligned by the ``(symbol, timestamp)`` pair.
    Rows containing ``NaN`` or ``inf`` in either the features or the labels are
    dropped.
    """

    if spec is None:
        spec = FeatureSpec()

    if "symbol" not in df.columns:
        raise KeyError("Input dataframe must contain a 'symbol' column")
    if "timestamp" not in df.columns:
        raise KeyError("Input dataframe must contain a 'timestamp' column")
    if "close" not in df.columns:
        raise KeyError("Input dataframe must contain a 'close' column")

    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.dropna(subset=["timestamp"])
    work = work.sort_values(["symbol", "timestamp"], kind="mergesort")
    work = work.reset_index(drop=True)

    grouped = work.groupby("symbol", sort=False)

    # Base log returns
    work["log_close"] = np.log(work["close"].astype(float))
    log_return = grouped["log_close"].diff()
    work["log_return_1"] = log_return

    features: dict[str, pd.Series] = {
        "log_return_1": log_return,
    }

    # Rolling statistics over specified windows.
    for window in spec.windows:
        if window <= 0:
            raise ValueError("Window sizes must be positive integers")
        rolling = grouped["log_return_1"].transform(
            lambda s, w=window: s.rolling(window=w, min_periods=1).mean()
        )
        features[f"ret_mean_{window}"] = rolling
        rolling_std = grouped["log_return_1"].transform(
            lambda s, w=window: s.rolling(window=w, min_periods=1).std()
        )
        features[f"ret_std_{window}"] = rolling_std
        price_ema = grouped["close"].transform(
            lambda s, w=window: s.ewm(span=w, adjust=False).mean()
        )
        features[f"price_ema_ratio_{window}"] = work["close"] / price_ema - 1.0

    # OHLC geometry features if the necessary columns are present.
    ohlc_required = {"open", "high", "low", "close"}
    body = pd.Series(np.nan, index=work.index)
    if ohlc_required.issubset(work.columns):
        body = work["close"] - work["open"]
        upper = work["high"] - work[["open", "close"]].max(axis=1)
        lower = work[["open", "close"]].min(axis=1) - work["low"]
        total_range = work["high"] - work["low"]
        features["candle_body"] = body
        features["candle_range"] = total_range
        features["candle_upper_shadow"] = upper
        features["candle_lower_shadow"] = lower
        with np.errstate(divide="ignore", invalid="ignore"):
            features["body_to_range"] = np.where(
                total_range.to_numpy() != 0,
                body / total_range,
                0.0,
            )

    # Volume and flow based indicators.
    if "volume" in work.columns:
        volume = work["volume"].astype(float)
        work["volume"] = volume
        features["volume_log"] = np.log1p(volume)
        volume_change = grouped["volume"].pct_change()
        if isinstance(volume_change.index, pd.MultiIndex):
            volume_change = volume_change.reset_index(level=0, drop=True)
        features["volume_change"] = volume_change.replace([np.inf, -np.inf], np.nan)
        for window in spec.windows:
            if window <= 1:
                continue
            zscore = grouped["volume"].transform(
                lambda s, w=window: (
                    s - s.rolling(w, min_periods=1).mean()
                )
                / s.rolling(w, min_periods=1).std()
            )
            features[f"volume_zscore_{window}"] = zscore
        if ohlc_required.issubset(work.columns):
            sign = np.sign(body.fillna(0.0))
            features["volume_signed"] = sign * volume

    # Calendar embeddings.
    hour = work["timestamp"].dt.hour.to_numpy()
    minute = work["timestamp"].dt.minute.to_numpy()
    day_of_week = work["timestamp"].dt.dayofweek.to_numpy()
    # Convert to radians for smooth cyclic features.
    hour_angle = 2 * np.pi * (hour + minute / 60.0) / 24.0
    dow_angle = 2 * np.pi * day_of_week / 7.0
    features["time_hour_sin"] = np.sin(hour_angle)
    features["time_hour_cos"] = np.cos(hour_angle)
    features["time_dow_sin"] = np.sin(dow_angle)
    features["time_dow_cos"] = np.cos(dow_angle)

    # Optional spread features.
    spread_cols = {"bid", "ask"}
    if spec.include_spread and spread_cols.issubset(work.columns):
        bid = work["bid"].astype(float)
        ask = work["ask"].astype(float)
        mid = (bid + ask) / 2.0
        work["mid_price"] = mid
        spread = ask - bid
        features["spread"] = spread
        with np.errstate(divide="ignore", invalid="ignore"):
            features["spread_pct"] = spread / mid
        features["mid_log_return"] = grouped["mid_price"].transform(
            lambda s: np.log(s).diff()
        )

    # Optional Order Flow Imbalance (OFI).
    ofi_cols = {"bid_size", "ask_size"}
    if spec.include_ofi and ofi_cols.issubset(work.columns):
        bid_size = work["bid_size"].astype(float)
        ask_size = work["ask_size"].astype(float)
        work["bid_size"] = bid_size
        work["ask_size"] = ask_size
        bid_size_prev = grouped["bid_size"].shift(1)
        ask_size_prev = grouped["ask_size"].shift(1)
        features["order_flow_imbalance"] = (bid_size - bid_size_prev) - (ask_size - ask_size_prev)

    feature_frame = pd.DataFrame(features, index=work.index)
    feature_frame.replace([np.inf, -np.inf], np.nan, inplace=True)

    cls_target, reg_target = compute_labels(work, spec)

    combined = pd.concat(
        [
            work[["symbol", "timestamp"]],
            feature_frame,
            cls_target,
            reg_target,
        ],
        axis=1,
    )

    combined = combined.dropna(subset=list(feature_frame.columns) + [cls_target.name, reg_target.name])

    meta = combined[["symbol", "timestamp"]].copy()
    X = combined[feature_frame.columns]
    y_cls = combined[cls_target.name].astype(int)
    y_reg = combined[reg_target.name].astype(float)

    return X, y_cls, y_reg, meta


def build_feature_map(feature_columns: Sequence[str] | pd.Index) -> dict[str, int]:
    """Return a deterministic mapping from feature name to column index.

    Parameters
    ----------
    feature_columns:
        Sequence of feature names. Passing ``X.columns`` from :func:`make_features`
        preserves the order used during model fitting.
    """

    ordered: list[str]
    if isinstance(feature_columns, pd.Index):
        ordered = feature_columns.tolist()
    else:
        ordered = list(feature_columns)

    return {name: idx for idx, name in enumerate(ordered)}

