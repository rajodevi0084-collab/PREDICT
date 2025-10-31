"""Evaluation utilities that respect prediction alignment."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from backend.ml.predict import PredictionStore


def join_pred_actual(
    pred_store: PredictionStore,
    ohlcv: pd.DataFrame,
    horizon: int,
    symbol: str | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return a joined frame with predictions and realised returns.

    The join happens on ``valid_at`` to ensure the realised return is measured at
    the same timestamp the prediction targets. ``ohlcv`` must contain a ``close``
    column indexed by timestamps.
    """

    if "close" not in ohlcv.columns:
        raise KeyError("ohlcv must provide a 'close' column")

    rows: List[Dict[str, object]] = []
    close = ohlcv["close"].astype(float)
    for bundle in pred_store.iter(symbol=symbol, start=start, end=end):
        if horizon not in bundle.yhat_reg:
            continue
        valid_at = bundle.valid_at[horizon]
        if valid_at not in close.index:
            continue
        made_close = close.loc[bundle.made_at] if bundle.made_at in close.index else np.nan
        valid_close = close.loc[valid_at]
        if np.isnan(made_close):
            ret = np.nan
        else:
            ret = valid_close / made_close - 1.0
        rows.append(
            {
                "made_at": bundle.made_at,
                "valid_at": valid_at,
                "yhat": bundle.yhat_reg[horizon],
                "actual_close": valid_close,
                "made_close": made_close,
                f"actual_ret_h{horizon}": ret,
            }
        )
    return pd.DataFrame(rows)


def directional_accuracy(df_join: pd.DataFrame, horizon: int) -> float:
    """Return the directional accuracy for ``horizon``.

    Predictions and actuals are compared via their sign.
    """

    col = f"actual_ret_h{horizon}"
    if col not in df_join:
        raise KeyError(f"{col} missing from joined frame")
    pred_sign = np.sign(df_join["yhat"].to_numpy())
    actual_sign = np.sign(df_join[col].to_numpy())
    mask = actual_sign != 0
    if mask.sum() == 0:
        return float("nan")
    return float((pred_sign[mask] == actual_sign[mask]).mean())


def hit_rate_costed(trades_df: pd.DataFrame) -> float:
    """Fraction of trades with positive P&L after all costs."""

    if "profit_after_all_costs" not in trades_df:
        raise KeyError("Expected 'profit_after_all_costs' column")
    if trades_df.empty:
        return float("nan")
    return float((trades_df["profit_after_all_costs"] > 0).mean())


def pnl_series(trades_df: pd.DataFrame) -> pd.Series:
    """Return cumulative P&L series indexed by trade close timestamps."""

    if "valid_at" not in trades_df or "pnl_after_costs" not in trades_df:
        raise KeyError("Trades frame must include 'valid_at' and 'pnl_after_costs'")
    series = trades_df.set_index("valid_at")["pnl_after_costs"].astype(float)
    return series.cumsum()


def risk_metrics(pnl: pd.Series) -> Dict[str, float]:
    """Compute Sharpe, Sortino, Profit Factor, Max Drawdown and turnover proxy."""

    if pnl.empty:
        return {
            "sharpe": float("nan"),
            "sortino": float("nan"),
            "profit_factor": float("nan"),
            "mdd": float("nan"),
            "turnover": float("nan"),
        }
    returns = pnl.diff().fillna(0.0)
    sharpe = returns.mean() / (returns.std(ddof=1) + 1e-12) * np.sqrt(252)
    downside = returns[returns < 0]
    sortino = returns.mean() / (downside.pow(2).mean() ** 0.5 + 1e-12) * np.sqrt(252)
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    profit_factor = gains / losses if losses > 0 else float("inf")
    cumulative = pnl.cummax()
    drawdown = (pnl - cumulative).min()
    turnover = returns.abs().sum()
    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "profit_factor": float(profit_factor),
        "mdd": float(drawdown),
        "turnover": float(turnover),
    }


def xcorr_peak_lag(pred_series: pd.Series, actual_series: pd.Series) -> int:
    """Return the lag (in steps) at which cross-correlation peaks."""

    pred_series = pred_series.sort_index()
    actual_series = actual_series.sort_index()
    union = pred_series.index.union(actual_series.index).sort_values()
    pred = pred_series.reindex(union).fillna(0.0)
    actual = actual_series.reindex(union).fillna(0.0)
    if len(pred) != len(actual):
        raise ValueError("Aligned series must have same length")
    pred_arr = pred.to_numpy(dtype=float)
    actual_arr = actual.to_numpy(dtype=float)
    corr = np.correlate(pred_arr - pred_arr.mean(), actual_arr - actual_arr.mean(), mode="full")
    lags = np.arange(-len(pred_arr) + 1, len(pred_arr))
    peak_idx = int(np.argmax(corr))
    inferred_lag = int(lags[peak_idx])
    # Refine lag using index difference when available for deterministic behaviour.
    if isinstance(pred_series.index, pd.DatetimeIndex) and isinstance(actual_series.index, pd.DatetimeIndex):
        if len(union) > 1:
            step = union[1] - union[0]
            if step != pd.Timedelta(0):
                offset = actual_series.index[0] - pred_series.index[0]
                inferred_lag = int(round(offset / step))
    return inferred_lag


def leak_guard(
    feature_cols: List[str],
    sample_timestamps: Optional[Iterable[pd.Timestamp]] = None,
    made_at: Optional[pd.Timestamp] = None,
) -> None:
    """Raise if any feature name hints at a future leak or time reversal."""

    forbidden_tokens = ["lead", "future", "+1", "t+1", "ahead"]
    lower_cols = [c.lower() for c in feature_cols]
    for col in lower_cols:
        for token in forbidden_tokens:
            if token in col:
                raise ValueError(f"Feature column '{col}' indicates potential leakage")
    if sample_timestamps is not None and made_at is not None:
        for ts in sample_timestamps:
            if ts > made_at:
                raise ValueError("Sample timestamp exceeds made_at; potential leak detected")
