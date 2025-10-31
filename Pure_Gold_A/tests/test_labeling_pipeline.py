from __future__ import annotations

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from pg.labeling.cusum import cusum_filter
from pg.labeling.triple_barrier import labels_from_events, make_events
from pg.labeling.volatility import compute_sigma
from pg.splits.purged_cv import purged_kfold


TZ = "Asia/Kolkata"


def _make_bars(close_values: list[float]) -> pd.DataFrame:
    ts = pd.date_range("2022-01-01 09:15", periods=len(close_values), freq="min", tz=TZ)
    data = {
        "symbol": ["AAA"] * len(close_values),
        "ts": ts,
        "open": close_values,
        "high": [c + 0.1 for c in close_values],
        "low": [c - 0.1 for c in close_values],
        "close": close_values,
        "volume": [1000.0] * len(close_values),
    }
    return pd.DataFrame(data)


def test_volatility_causal_positive():
    close = [100 + i * 0.01 for i in range(10)]
    bars = _make_bars(close)
    sigma_full = compute_sigma(
        bars,
        method="ewma",
        span_minutes=3,
        min_sigma=1e-4,
        ts_col="ts",
        ohlc_cols=["open", "high", "low", "close"],
    )
    assert (sigma_full >= 1e-4).all()

    sigma_partial = compute_sigma(
        bars.iloc[:5],
        method="ewma",
        span_minutes=3,
        min_sigma=1e-4,
        ts_col="ts",
        ohlc_cols=["open", "high", "low", "close"],
    )
    np.testing.assert_allclose(sigma_partial.values, sigma_full.iloc[:5].values)


def test_cusum_detects_jump():
    close = [100.0] * 5 + [102.0] + [102.5] * 4
    bars = _make_bars(close)
    sigma = pd.Series(0.01, index=bars.index)
    events = cusum_filter(
        bars,
        ts_col="ts",
        price_col="close",
        sigma=sigma,
        k_sigma=1.5,
        drift=0.0,
        min_spacing_minutes=1,
    )
    assert len(events) == 1
    assert events[0] == bars.loc[5, "ts"]


def test_triple_barrier_labels():
    close = [100.0, 101.8, 102.5, 100.0, 98.0, 100.0]
    bars = _make_bars(close)
    sigma = pd.Series(0.02, index=bars.index)
    events_ts = pd.DatetimeIndex([bars.loc[0, "ts"], bars.loc[3, "ts"], bars.loc[5, "ts"]])
    events_df = make_events(
        bars,
        events_ts=events_ts,
        ts_col="ts",
        symbol_col="symbol",
        sigma=sigma,
        pt_k=1.0,
        sl_k=1.0,
        max_horizon_minutes=5,
        min_move_k=0.1,
    )
    labels_df = labels_from_events(events_df, drop_ambiguous=False)
    assert set(labels_df["label"]) == {1, -1, 0}

    dropped_df = labels_from_events(events_df, drop_ambiguous=True)
    assert len(dropped_df) <= len(labels_df)
    assert set(dropped_df["label"]).issubset({-1, 0, 1})


def test_purged_cv_no_leakage():
    ts = pd.date_range("2022-01-01 09:15", periods=12, freq="10min", tz=TZ)
    events_df = pd.DataFrame(
        {
            "event_id": np.arange(6),
            "symbol": ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
            "t_event": ts[:6],
            "t_end": ts[:6] + pd.Timedelta(minutes=10),
            "pt_px": np.ones(6),
            "sl_px": np.ones(6),
            "t_touch": ts[:6],
            "label": [1, -1, 0, 1, -1, 0],
            "ret": np.zeros(6),
            "ambiguous": [False] * 6,
        }
    )
    events_df.attrs["n_blocks"] = 6
    splits = purged_kfold(events_df, n_folds=3, embargo_minutes=10)
    for fold in splits:
        train_idx = np.array(fold["train_idx"])
        val_idx = np.array(fold["val_idx"])
        assert np.intersect1d(train_idx, val_idx).size == 0
        val_start, val_end = [pd.Timestamp(x) for x in fold["val_window"]]
        purge_mask = (events_df["t_event"] <= val_end) & (events_df["t_end"] >= val_start)
        embargo_mask = (events_df["t_event"] <= val_end + pd.Timedelta(minutes=10)) & (
            events_df["t_end"] >= val_start - pd.Timedelta(minutes=10)
        )
        excluded = np.flatnonzero((purge_mask | embargo_mask).values)
        assert set(excluded).isdisjoint(train_idx)
