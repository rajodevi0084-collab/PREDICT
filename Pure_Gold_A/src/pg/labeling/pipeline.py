"""High-level labeling pipeline orchestrator."""
from __future__ import annotations

from glob import glob
from pathlib import Path
from time import perf_counter
from typing import Any, Dict

import pandas as pd

from pg.labeling.cusum import cusum_filter
from pg.labeling.triple_barrier import labels_from_events, make_events
from pg.labeling.volatility import compute_sigma
from pg.utils.logging_and_seed import get_logger


_LOG = get_logger(__name__)


def _load_bars(parquet_glob: str, cfg: dict[str, Any]) -> pd.DataFrame:
    files = sorted(glob(parquet_glob))
    if not files:
        raise FileNotFoundError(f"no parquet files matched glob: {parquet_glob}")

    dfs = []
    for path in files:
        start = perf_counter()
        df = pd.read_parquet(path)
        duration = perf_counter() - start
        _LOG.info("loaded %s with %d rows in %.2fs", path, len(df), duration)
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    ts_col = cfg["data"]["ts_col"]
    symbol_col = cfg["data"]["symbol_col"]

    if ts_col not in data.columns:
        raise KeyError(f"timestamp column '{ts_col}' missing from bars")

    data[ts_col] = pd.to_datetime(data[ts_col])

    tz = cfg["data"]["timezone"]
    if data[ts_col].dt.tz is None:
        data[ts_col] = data[ts_col].dt.tz_localize(tz)
    elif str(data[ts_col].dt.tz) != tz:
        data[ts_col] = data[ts_col].dt.tz_convert(tz)

    data = data.sort_values([symbol_col, ts_col]).reset_index(drop=True)
    if data.duplicated([symbol_col, ts_col]).any():
        raise ValueError("duplicate symbol/timestamp rows detected")

    required_cols = cfg["data"]["ohlc_cols"] + [symbol_col, ts_col, "volume"]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise KeyError(f"missing required columns: {missing}")

    return data[[symbol_col, ts_col, *cfg["data"]["ohlc_cols"], "volume"]]


def build_labels(parquet_glob: str, cfg: dict[str, Any]) -> Dict[str, Any]:
    """Execute the event labeling pipeline and persist outputs."""

    start_total = perf_counter()
    bars = _load_bars(parquet_glob, cfg)
    _LOG.info("loaded %d bars", len(bars))
    if bars.empty:
        raise RuntimeError("no bars loaded")

    ts_col = cfg["data"]["ts_col"]
    symbol_col = cfg["data"]["symbol_col"]
    ohlc_cols = cfg["data"]["ohlc_cols"]

    sigma = compute_sigma(
        bars,
        method=cfg["volatility"]["method"],
        span_minutes=cfg["volatility"]["span_minutes"],
        min_sigma=cfg["volatility"]["min_sigma"],
        ts_col=ts_col,
        ohlc_cols=ohlc_cols,
    )
    _LOG.info("sigma computed with mean %.6f", float(sigma.mean()))

    events_ts = cusum_filter(
        bars,
        ts_col=ts_col,
        price_col=ohlc_cols[-1],
        sigma=sigma,
        k_sigma=cfg["cusum"]["k_sigma"],
        drift=cfg["cusum"]["drift"],
        min_spacing_minutes=cfg["cusum"]["min_spacing_minutes"],
    )
    if events_ts.empty:
        raise RuntimeError("CUSUM produced no events")
    _LOG.info("CUSUM produced %d raw events", len(events_ts))

    events_df = make_events(
        bars,
        events_ts=events_ts,
        ts_col=ts_col,
        symbol_col=symbol_col,
        sigma=sigma,
        pt_k=cfg["triple_barrier"]["pt_k"],
        sl_k=cfg["triple_barrier"]["sl_k"],
        max_horizon_minutes=cfg["triple_barrier"]["max_horizon_minutes"],
        min_move_k=cfg["triple_barrier"]["min_move_k"],
    )
    labels_df = labels_from_events(events_df, cfg["triple_barrier"]["drop_ambiguous"])
    if labels_df.empty:
        raise RuntimeError("no labels after ambiguity filter")

    events_path = Path(cfg["io"]["events_parquet"])
    labels_path = Path(cfg["io"]["labels_parquet"])
    for path in [events_path, labels_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    events_df.to_parquet(events_path, index=False)
    labels_df.to_parquet(labels_path, index=False)
    _LOG.info("persisted events to %s and labels to %s", events_path, labels_path)

    positives = int((labels_df["label"] == 1).sum())
    negatives = int((labels_df["label"] == -1).sum())
    zeros = int((labels_df["label"] == 0).sum())

    summary = {
        "bars": int(len(bars)),
        "events_raw": int(len(events_df)),
        "events_after_min_move": int(len(labels_df)),
        "positives": positives,
        "negatives": negatives,
        "zeros": zeros,
        "dropped": int(len(events_df) - len(labels_df)),
        "duration_seconds": perf_counter() - start_total,
    }
    _LOG.info("pipeline summary: %s", summary)
    return summary
