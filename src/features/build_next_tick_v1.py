"""Implementation of the ``next_tick_v1`` feature family."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .spec_loader import FeatureSpec


def make_price_features(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    price = df["mid"].astype(float)
    features: Dict[str, pd.Series] = {}

    for window in spec.group("price").get("returns", []):
        if window <= 0:
            continue
        feat = np.log(price / price.shift(window))
        features[f"ret_l{window}"] = feat

    for span in spec.group("price").get("ema_gaps", []):
        ema = price.ewm(span=span, adjust=False).mean()
        features[f"ema_gap_{span}"] = price - ema

    for window in spec.group("price").get("range_pct", []):
        rolling_high = price.rolling(window=window, min_periods=1).max()
        rolling_low = price.rolling(window=window, min_periods=1).min()
        rng = rolling_high - rolling_low
        features[f"range_pct_{window}"] = np.where(price != 0, rng / price, 0.0)

    for window in spec.group("price").get("realized_vol", []):
        returns = np.log(price / price.shift(1)).fillna(0.0)
        vol = returns.rolling(window=window, min_periods=1).std(ddof=0)
        features[f"rv_{window}"] = vol

    return pd.DataFrame(features)


def make_micro_features(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    groups = spec.group("micro")
    features: Dict[str, pd.Series] = {}

    if "vwap" in df.columns:
        vwap = df["vwap"].astype(float)
    else:
        vwap = df[["bid", "ask"]].astype(float).mean(axis=1)

    mid = df["mid"].astype(float)

    for window in groups.get("vwap_gap", []):
        features[f"vwap_gap_{window}"] = vwap - mid.rolling(window, min_periods=1).mean()

    for window in groups.get("spread", []):
        features[f"spread_{window}"] = (df["ask"].astype(float) - df["bid"].astype(float)).rolling(
            window, min_periods=1
        ).mean()

    imbalance_windows = groups.get("imbalance", [])
    if {"bid_size", "ask_size"}.issubset(df.columns):
        imbalance = (
            df["bid_size"].astype(float) - df["ask_size"].astype(float)
        ) / (df["bid_size"].astype(float) + df["ask_size"].astype(float)).replace(0, np.nan)
        imbalance = imbalance.fillna(0.0)
        for window in imbalance_windows:
            features[f"imbalance_{window}"] = imbalance.rolling(window, min_periods=1).mean()

    burst_windows = groups.get("trade_burst", [])
    if "volume" in df.columns:
        volume = df["volume"].astype(float)
        for window in burst_windows:
            features[f"trade_burst_{window}"] = volume.rolling(window, min_periods=1).sum()

    return pd.DataFrame(features)


def make_orderbook_features(orderbook: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    groups = spec.group("orderbook")
    features: Dict[str, pd.Series] = {}

    levels: Iterable[int] = groups.get("levels", [])
    for level in levels:
        bid_col = f"bid_{level}"
        ask_col = f"ask_{level}"
        if bid_col in orderbook.columns and ask_col in orderbook.columns:
            features[f"level_mid_{level}"] = (orderbook[bid_col] + orderbook[ask_col]) / 2.0

        bid_sz_col = f"bid_size_{level}"
        ask_sz_col = f"ask_size_{level}"
        if bid_sz_col in orderbook.columns and ask_sz_col in orderbook.columns:
            denom = orderbook[bid_sz_col] + orderbook[ask_sz_col]
            denom = denom.replace(0, np.nan)
            features[f"depth_skew_{level}"] = (
                (orderbook[bid_sz_col] - orderbook[ask_sz_col]) / denom
            ).fillna(0.0)

    if groups.get("depth_sums", False):
        bid_sum = orderbook[[c for c in orderbook.columns if c.startswith("bid_size_")]].sum(axis=1)
        ask_sum = orderbook[[c for c in orderbook.columns if c.startswith("ask_size_")]].sum(axis=1)
        features["depth_sum_bid"] = bid_sum
        features["depth_sum_ask"] = ask_sum

    if groups.get("skew", False) and {"depth_sum_bid", "depth_sum_ask"}.issubset(features):
        denom = features["depth_sum_bid"] + features["depth_sum_ask"]
        denom = denom.replace(0, np.nan)
        features["depth_skew"] = (
            (features["depth_sum_bid"] - features["depth_sum_ask"]) / denom
        ).fillna(0.0)

    if groups.get("slope", False):
        slopes = []
        for level in levels:
            bid_col = f"bid_{level}"
            ask_col = f"ask_{level}"
            if bid_col in orderbook.columns and ask_col in orderbook.columns:
                slopes.append(orderbook[ask_col] - orderbook[bid_col])
        if slopes:
            features["book_slope"] = pd.concat(slopes, axis=1).mean(axis=1)

    if groups.get("deltas"):
        for delta in groups.get("deltas", []):
            shifted = orderbook.filter(regex=r"^(bid|ask)_").diff(delta)
            for col in shifted.columns:
                features[f"{col}_delta{delta}"] = shifted[col]

    return pd.DataFrame(features)


def make_regime_features(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    groups = spec.group("regime")
    features: Dict[str, pd.Series] = {}

    if groups.get("vol_quantile"):
        returns = np.log(df["mid"].astype(float) / df["mid"].astype(float).shift(1)).fillna(0.0)
        vol = returns.rolling(window=64, min_periods=1).std(ddof=0)
        features["vol_quantile"] = vol.rank(pct=True)

    if groups.get("session_bucket"):
        timestamp = pd.to_datetime(df.index)
        features["session_bucket"] = timestamp.hour + timestamp.minute / 60.0

    return pd.DataFrame(features)


def apply_hygiene(X: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    hygiene = spec.group("hygiene", default={}) if hasattr(spec, "group") else {}
    X = X.copy()

    winsor_sigma = hygiene.get("winsor_sigma")
    if winsor_sigma:
        mean = X.mean()
        std = X.std(ddof=0).replace(0, np.nan)
        upper = mean + winsor_sigma * std
        lower = mean - winsor_sigma * std
        X = X.clip(lower=lower, upper=upper, axis=1)

    if hygiene.get("vol_standardize"):
        std = X.std(ddof=0).replace(0, 1.0)
        X = (X - X.mean()) / std

    corr_threshold = hygiene.get("drop_corr_gt")
    if corr_threshold is not None and not X.empty:
        keep_columns: list[str] = []
        corr_matrix = X.corr().abs()
        for column in corr_matrix.columns:
            if not keep_columns:
                keep_columns.append(column)
                continue
            if all(corr_matrix.loc[column, kept] <= corr_threshold for kept in keep_columns):
                keep_columns.append(column)
        X = X[keep_columns]

    keep_top_k = spec.keep_top_k()
    if keep_top_k is not None and keep_top_k < X.shape[1]:
        variances = X.var().sort_values(ascending=False)
        cols = variances.index[:keep_top_k]
        X = X[cols]

    return X.fillna(0.0)


__all__ = [
    "make_price_features",
    "make_micro_features",
    "make_orderbook_features",
    "make_regime_features",
    "apply_hygiene",
]
