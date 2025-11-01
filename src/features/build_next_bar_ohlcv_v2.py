"""Feature construction pipeline for the ``next_bar_ohlcv_v2`` spec."""

from __future__ import annotations

import itertools
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold

from ..data.price_source import typical_price
from .spec_loader import FeatureSpec


def assert_past_only(X: pd.DataFrame, price_index: pd.Index) -> None:
    """Ensure feature timestamps never extend beyond observed prices."""

    if X.empty:
        raise ValueError("Feature matrix is empty")

    price_index = pd.Index(price_index)
    if not X.index.is_monotonic_increasing:
        raise AssertionError("Feature index must be non-decreasing")

    if not X.index.isin(price_index).all():
        missing = X.index.difference(price_index)
        raise AssertionError(f"Feature index contains future timestamps: {missing[:5].tolist()}")

    if X.index.max() > price_index.max():
        raise AssertionError("Feature index extends beyond available price data")


def _safe_log_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    ratio = num / den.replace(0, np.nan)
    out = np.log(ratio.replace(0, np.nan))
    return out.replace([np.inf, -np.inf], np.nan)


def make_returns_trend(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    groups = spec.group("returns_trend", default={})
    close = df["close"].astype(float)
    high = df.get("high", close).astype(float)
    low = df.get("low", close).astype(float)
    typ = typical_price(df)

    features: Dict[str, pd.Series] = {}

    for lag in groups.get("log_returns_close_lags", []):
        if lag <= 0:
            continue
        features[f"log_ret_close_l{lag}"] = _safe_log_ratio(close, close.shift(lag))

    if groups.get("log_returns_high_low_typical", False):
        features["log_ret_high_low"] = _safe_log_ratio(high, low)
        features["log_ret_typical"] = _safe_log_ratio(typ, typ.shift(1))

    for span in groups.get("ema_gaps_close", []):
        ema = close.ewm(span=span, adjust=False).mean()
        features[f"ema_gap_close_{span}"] = close - ema

    for span in groups.get("ema_gaps_typical", []):
        ema_typ = typ.ewm(span=span, adjust=False).mean()
        features[f"ema_gap_typical_{span}"] = typ - ema_typ

    for span in groups.get("zscore_price_vs_ema", []):
        ema = close.ewm(span=span, adjust=False).mean()
        std = close.ewm(span=span, adjust=False).std().replace(0, np.nan)
        features[f"zclose_vs_ema_{span}"] = (close - ema) / std

    return pd.DataFrame(features)


def make_volatility_range(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    groups = spec.group("volatility_range", default={})
    close = df["close"].astype(float)
    high = df.get("high", close).astype(float)
    low = df.get("low", close).astype(float)

    features: Dict[str, pd.Series] = {}

    returns = np.log(close / close.shift(1)).fillna(0.0)
    for window in groups.get("realized_vol_windows", []):
        vol = returns.rolling(window=window, min_periods=1).std(ddof=0)
        features[f"rv_{window}"] = vol

    for window in groups.get("atr_windows", []):
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        features[f"atr_{window}"] = tr.rolling(window=window, min_periods=1).mean()

    for window in groups.get("range_pct_windows", []):
        high_roll = high.rolling(window=window, min_periods=1).max()
        low_roll = low.rolling(window=window, min_periods=1).min()
        rng = high_roll - low_roll
        features[f"range_pct_{window}"] = rng / close.replace(0, np.nan)

    if groups.get("park_gk_rs_yz", False):
        log_hl = _safe_log_ratio(high, low)
        log_co = _safe_log_ratio(close, close.shift(1))
        log_ho = _safe_log_ratio(high, close.shift(1))
        log_lo = _safe_log_ratio(low, close.shift(1))
        parkinson = (log_hl ** 2) / (4 * np.log(2))
        garman_klass = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        yz = log_co ** 2 + 0.164334 * (log_hl ** 2) + 0.25364 * log_co * log_hl
        features["vol_parkinson"] = parkinson
        features["vol_gk"] = garman_klass
        features["vol_rs"] = rs
        features["vol_yz"] = yz

    return pd.DataFrame(features)


def make_gaps_session(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    groups = spec.group("gaps_session", default={})
    features: Dict[str, pd.Series] = {}

    if groups.get("open_to_prev_close_gap", False) and {"open", "close"}.issubset(df.columns):
        prev_close = df["close"].shift(1)
        features["gap_open_prev_close"] = (df["open"] - prev_close) / prev_close.replace(0, np.nan)

    index = pd.to_datetime(df.index)
    if groups.get("session_bucket"):
        buckets: Sequence[str] = groups.get("session_bucket", [])
        hour = index.hour + index.minute / 60.0
        if "open" in buckets:
            features["session_open"] = (hour <= 1.0).astype(int)
        if "mid" in buckets:
            features["session_mid"] = ((hour > 1.0) & (hour < 6.0)).astype(int)
        if "close" in buckets:
            features["session_close"] = (hour >= 6.0).astype(int)

    if groups.get("dow_dom_month", False):
        features["day_of_week"] = index.dayofweek / 6.0
        features["day_of_month"] = (index.day - 1) / 30.0
        features["month_of_year"] = (index.month - 1) / 11.0

    if groups.get("expiry_distance", False):
        quarter_end = index.to_period("Q").end_time
        days_to_expiry = (quarter_end - index).days.astype(float)
        features["days_to_quarter_end"] = days_to_expiry

    return pd.DataFrame(features)


def make_volume_moneyflow(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    groups = spec.group("volume_moneyflow", default={})
    volume = df.get("volume", pd.Series(0.0, index=df.index)).astype(float)
    close = df["close"].astype(float)
    typ = typical_price(df)

    features: Dict[str, pd.Series] = {}

    for lag in groups.get("volume_returns_lags", []):
        features[f"vol_ret_l{lag}"] = _safe_log_ratio(volume, volume.shift(lag))

    for window in groups.get("volume_zscore_windows", []):
        rolling = volume.rolling(window=window, min_periods=1)
        z = (volume - rolling.mean()) / rolling.std(ddof=0).replace(0, np.nan)
        features[f"volume_z_{window}"] = z

    if groups.get("obv", False):
        direction = np.sign(close.diff().fillna(0.0))
        features["obv"] = (volume * direction).cumsum()

    for window in groups.get("mfi_windows", []):
        tp = typ
        money_flow = tp * volume
        pos_flow = money_flow.where(tp >= tp.shift(1), other=0.0)
        neg_flow = money_flow.where(tp < tp.shift(1), other=0.0)
        ratio = pos_flow.rolling(window, min_periods=1).sum() / neg_flow.rolling(
            window, min_periods=1
        ).sum().replace(0, np.nan)
        features[f"mfi_{window}"] = 100 - 100 / (1 + ratio)

    for window in groups.get("cmf_windows", []):
        high = df.get("high", close).astype(float)
        low = df.get("low", close).astype(float)
        mf = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
        features[f"cmf_{window}"] = (mf * volume).rolling(window, min_periods=1).sum() / volume.rolling(
            window, min_periods=1
        ).sum().replace(0, np.nan)

    for window in groups.get("price_volume_corr_windows", []):
        returns = np.log(close / close.shift(1))
        corr = returns.rolling(window, min_periods=1).corr(volume)
        features[f"price_volume_corr_{window}"] = corr

    return pd.DataFrame(features)


def make_oscillators_light(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    groups = spec.group("oscillators_light", default={})
    close = df["close"].astype(float)
    high = df.get("high", close).astype(float)
    low = df.get("low", close).astype(float)

    features: Dict[str, pd.Series] = {}

    diff = close.diff()
    for window in groups.get("rsi_windows", []):
        gain = diff.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
        loss = (-diff.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        features[f"rsi_{window}"] = 100 - 100 / (1 + rs)

    for k, d in [tuple(groups.get("stoch_kd", []))] if groups.get("stoch_kd") else []:
        lowest_low = low.rolling(window=k, min_periods=1).min()
        highest_high = high.rolling(window=k, min_periods=1).max()
        percent_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
        percent_d = percent_k.rolling(window=d, min_periods=1).mean()
        features[f"stoch_k_{k}"] = percent_k
        features[f"stoch_d_{k}_{d}"] = percent_d

    for window in groups.get("adx_windows", []):
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        tr = pd.concat(
            [(high - low).abs(), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(window, min_periods=1).mean()
        plus_dm_series = pd.Series(plus_dm, index=df.index)
        minus_dm_series = pd.Series(minus_dm, index=df.index)
        plus_di = 100 * plus_dm_series.rolling(window, min_periods=1).sum() / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm_series.rolling(window, min_periods=1).sum() / atr.replace(0, np.nan)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
        features[f"adx_{window}"] = dx.rolling(window, min_periods=1).mean()

    for window in groups.get("cci_windows", []):
        tp = typical_price(df)
        ma = tp.rolling(window, min_periods=1).mean()
        md = (tp - ma).abs().rolling(window, min_periods=1).mean()
        features[f"cci_{window}"] = (tp - ma) / (0.015 * md).replace(0, np.nan)

    for window in groups.get("williams_r_windows", []):
        highest_high = high.rolling(window, min_periods=1).max()
        lowest_low = low.rolling(window, min_periods=1).min()
        features[f"williams_r_{window}"] = -100 * (highest_high - close) / (
            highest_high - lowest_low
        ).replace(0, np.nan)

    return pd.DataFrame(features)


def make_patterns(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    groups = spec.group("patterns", default={})
    open_ = df.get("open", df["close"]).astype(float)
    close = df["close"].astype(float)
    high = df.get("high", close).astype(float)
    low = df.get("low", close).astype(float)

    features: Dict[str, pd.Series] = {}

    if groups.get("candle_ratios", False):
        body = (close - open_).abs()
        range_ = (high - low).replace(0, np.nan)
        features["body_to_range"] = body / range_
        features["upper_shadow_ratio"] = (high - close).abs() / range_
        features["lower_shadow_ratio"] = (open_ - low).abs() / range_

    if groups.get("engulf_harami_nr4_nr7_inside_outside", False):
        prev_open = open_.shift(1)
        prev_close = close.shift(1)
        engulf = ((close > prev_open) & (open_ < prev_close)) | (
            (close < prev_open) & (open_ > prev_close)
        )
        harami = ((close < prev_open) & (open_ > prev_close)) | (
            (close > prev_open) & (open_ < prev_close)
        )
        range_ = high - low
        nr4 = range_ < range_.rolling(4, min_periods=1).max().shift(1)
        nr7 = range_ < range_.rolling(7, min_periods=1).max().shift(1)
        inside = (high <= high.shift(1)) & (low >= low.shift(1))
        outside = (high >= high.shift(1)) & (low <= low.shift(1))
        features.update(
            {
                "pattern_engulf": engulf.astype(int),
                "pattern_harami": harami.astype(int),
                "pattern_nr4": nr4.astype(int),
                "pattern_nr7": nr7.astype(int),
                "pattern_inside": inside.astype(int),
                "pattern_outside": outside.astype(int),
            }
        )

    pattern_keys = [k for k in features if k.startswith("pattern_")]
    for window in groups.get("pattern_counts_windows", []):
        for key in pattern_keys:
            series = pd.Series(features[key], index=df.index)
            features[f"{key}_cnt_{window}"] = series.rolling(window, min_periods=1).sum()

    return pd.DataFrame(features)


def make_spectral_fractal(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    groups = spec.group("spectral_fractal", default={})
    close = df["close"].astype(float)
    features: Dict[str, pd.Series] = {}

    window = 64
    if groups.get("dct_lowk", 0):
        k = int(groups.get("dct_lowk", 0))
        coeffs = []
        for i in range(len(close)):
            segment = close.iloc[max(0, i - window + 1) : i + 1]
            if len(segment) < 4:
                coeffs.append([np.nan] * k)
                continue
            dct = np.real(np.fft.fft(segment - segment.mean()))
            coeffs.append(dct[:k])
        coeff_df = pd.DataFrame(coeffs, index=close.index)
        for col in coeff_df.columns:
            features[f"dct_c{col}"] = coeff_df[col]

    if groups.get("fft_lowk", 0):
        k = int(groups.get("fft_lowk", 0))
        fft_coeffs = []
        for i in range(len(close)):
            segment = close.iloc[max(0, i - window + 1) : i + 1]
            if len(segment) < 4:
                fft_coeffs.append([np.nan] * k)
                continue
            fft = np.fft.rfft(segment - segment.mean())
            fft_coeffs.append(np.abs(fft[:k]))
        fft_df = pd.DataFrame(fft_coeffs, index=close.index)
        for col in fft_df.columns:
            features[f"fft_c{col}"] = fft_df[col]

    for wavelet in groups.get("wavelet_db", []):
        span = int(wavelet) * 4
        sma = close.rolling(span, min_periods=1).mean()
        features[f"wavelet_db{wavelet}"] = close - sma

    if groups.get("hurst_dfa_entropy", False):
        returns = np.log(close / close.shift(1)).fillna(0.0)
        hurst = _rolling_hurst(returns)
        features["hurst"] = hurst
        features["dfa_alpha"] = _detrended_fluctuation(returns)
        entropy = -np.abs(returns.rolling(window, min_periods=1).mean())
        features["entropy_proxy"] = entropy

    return pd.DataFrame(features)


def _rolling_hurst(series: pd.Series, window: int = 128) -> pd.Series:
    hurst_vals = []
    for i in range(len(series)):
        segment = series.iloc[max(0, i - window + 1) : i + 1]
        if len(segment) < 16:
            hurst_vals.append(np.nan)
            continue
        lagged_diffs = []
        for lag in range(2, min(20, len(segment))):
            diff = segment.diff(lag).dropna()
            if diff.empty:
                continue
            lagged_diffs.append((lag, np.sqrt((diff ** 2).mean())))
        if not lagged_diffs:
            hurst_vals.append(np.nan)
            continue
        logs = np.log([ld[0] for ld in lagged_diffs])
        log_rs = np.log([ld[1] for ld in lagged_diffs])
        slope, _ = np.polyfit(logs, log_rs, 1)
        hurst_vals.append(slope)
    return pd.Series(hurst_vals, index=series.index)


def _detrended_fluctuation(series: pd.Series, window: int = 128) -> pd.Series:
    values = series.fillna(0.0).to_numpy()
    cumsum = np.cumsum(values - values.mean())
    dfa_vals = []
    for i in range(len(values)):
        segment = cumsum[max(0, i - window + 1) : i + 1]
        if len(segment) < 16:
            dfa_vals.append(np.nan)
            continue
        x = np.arange(len(segment))
        coeffs = np.polyfit(x, segment, 1)
        trend = np.polyval(coeffs, x)
        fluct = np.sqrt(np.mean((segment - trend) ** 2))
        dfa_vals.append(fluct)
    return pd.Series(dfa_vals, index=series.index)


def make_interactions(X: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    groups = spec.group("interactions", default={})
    top_k = int(groups.get("topk_pairwise", 0))
    if top_k <= 0 or X.empty:
        return pd.DataFrame(index=X.index)

    variances = X.var().sort_values(ascending=False)
    cols = variances.index[: min(top_k, len(variances))]
    combos = itertools.combinations(cols, 2)
    interactions = {}
    for a, b in itertools.islice(combos, 0, top_k * (top_k - 1) // 2):
        interactions[f"{a}_x_{b}"] = X[a] * X[b]
    if not interactions:
        return pd.DataFrame(index=X.index)
    return pd.DataFrame(interactions, index=X.index)


def apply_hygiene(X: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    hygiene = spec.group("hygiene", default={})
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
    if corr_threshold is not None and X.shape[1] > 1:
        keep: list[str] = []
        corr = X.corr().abs()
        for column in corr.columns:
            if not keep:
                keep.append(column)
                continue
            if all(corr.loc[column, k] <= corr_threshold for k in keep):
                keep.append(column)
        X = X[keep]

    return X


def _score_features_mic(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    if X.empty:
        return pd.Series(dtype=float)

    # Use out-of-fold mutual information as a proxy for MIC/SFI ranking.
    n_samples = len(X)
    n_splits = min(5, n_samples)
    X_filled = X.fillna(0.0)
    if n_splits < 2:
        mi = mutual_info_regression(X_filled, y, random_state=42)
        return pd.Series(mi, index=X.columns)

    scores = np.zeros(X.shape[1], dtype=float)
    counts = np.zeros_like(scores)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, _ in kf.split(X):
        X_train = X_filled.iloc[train_idx]
        y_train = y.iloc[train_idx]
        mi = mutual_info_regression(X_train, y_train, random_state=42)
        scores += mi
        counts += 1

    counts[counts == 0] = 1
    scores /= counts
    return pd.Series(scores, index=X.columns)


def rank_and_select(X: pd.DataFrame, y: pd.Series | None, spec: FeatureSpec) -> pd.DataFrame:
    keep_top_k = spec.keep_top_k()
    if keep_top_k is None or X.shape[1] <= keep_top_k:
        return X

    if y is None:
        variances = X.var().sort_values(ascending=False)
        selected = variances.index[:keep_top_k]
        return X[selected]

    scores = _score_features_mic(X, y)
    ranked = scores.sort_values(ascending=False)
    selected_cols = ranked.index[:keep_top_k]
    return X[selected_cols]


def build_feature_matrix(
    df: pd.DataFrame,
    spec: FeatureSpec,
) -> pd.DataFrame:
    """Construct the feature matrix using the ``next_bar_ohlcv_v2`` recipe."""

    frames = [
        make_returns_trend(df, spec),
        make_volatility_range(df, spec),
        make_gaps_session(df, spec),
        make_volume_moneyflow(df, spec),
        make_oscillators_light(df, spec),
        make_patterns(df, spec),
        make_spectral_fractal(df, spec),
    ]
    X = pd.concat(frames, axis=1)

    X = apply_hygiene(X, spec)

    X = X.reindex(df.index)
    assert_past_only(X, df.index)

    interactions = make_interactions(X, spec)
    if not interactions.empty:
        interactions = interactions.reindex(X.index)
        X = pd.concat([X, interactions], axis=1)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X


__all__ = [
    "make_returns_trend",
    "make_volatility_range",
    "make_gaps_session",
    "make_volume_moneyflow",
    "make_oscillators_light",
    "make_patterns",
    "make_spectral_fractal",
    "make_interactions",
    "apply_hygiene",
    "rank_and_select",
    "assert_past_only",
    "build_feature_matrix",
]
