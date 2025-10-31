"""Triple-barrier labeling utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from pg.utils.logging_and_seed import get_logger


_LOG = get_logger(__name__)


@dataclass
class _EventRecord:
    event_id: int
    symbol: str
    t_event: pd.Timestamp
    t_end: pd.Timestamp
    pt_px: float
    sl_px: float
    t_touch: pd.Timestamp | pd.NaT
    label: int
    ret: float
    ambiguous: bool


def _validate_make_events(
    df: pd.DataFrame,
    events_ts: pd.DatetimeIndex,
    ts_col: str,
    symbol_col: str,
    sigma: pd.Series,
    pt_k: float,
    sl_k: float,
    max_horizon_minutes: int,
    min_move_k: float,
) -> None:
    if df.empty:
        raise ValueError("bars dataframe is empty")
    if events_ts.empty:
        raise ValueError("events timestamp index is empty")
    if ts_col not in df.columns:
        raise KeyError(f"timestamp column '{ts_col}' missing")
    if symbol_col not in df.columns:
        raise KeyError(f"symbol column '{symbol_col}' missing")
    if not sigma.index.equals(df.index):
        raise ValueError("sigma series must align with dataframe index")
    if pt_k <= 0 or sl_k <= 0:
        raise ValueError("pt_k and sl_k must be positive")
    if max_horizon_minutes <= 0:
        raise ValueError("max_horizon_minutes must be positive")
    if min_move_k < 0:
        raise ValueError("min_move_k must be non-negative")
    if events_ts.tz is None or str(events_ts.tz) != "Asia/Kolkata":
        raise ValueError("events_ts must be timezone-aware Asia/Kolkata")
    if df.duplicated([symbol_col, ts_col]).any():
        raise ValueError("symbol/timestamp combinations must be unique")


def _price_to_barriers(close_px: float, sigma_event: float, pt_k: float, sl_k: float) -> tuple[float, float]:
    pt_px = close_px * float(np.exp(pt_k * sigma_event))
    sl_px = close_px * float(np.exp(-sl_k * sigma_event))
    return pt_px, sl_px


def make_events(
    df: pd.DataFrame,
    events_ts: pd.DatetimeIndex,
    ts_col: str,
    symbol_col: str,
    sigma: pd.Series,
    pt_k: float,
    sl_k: float,
    max_horizon_minutes: int,
    min_move_k: float,
) -> pd.DataFrame:
    """Construct triple-barrier event metadata and labels."""

    _validate_make_events(
        df, events_ts, ts_col, symbol_col, sigma, pt_k, sl_k, max_horizon_minutes, min_move_k
    )

    df_sorted = df.sort_values([symbol_col, ts_col])

    records: List[_EventRecord] = []

    for event_id, event_ts in enumerate(events_ts):
        matches = df_sorted[df_sorted[ts_col] == event_ts]
        if matches.empty:
            raise KeyError(f"event timestamp {event_ts} not found in bars")
        if len(matches) != 1:
            raise ValueError(
                f"event timestamp {event_ts} is not unique across symbols; ensure unique events"
            )
        row = matches.iloc[0]
        orig_idx = row.name
        symbol = row[symbol_col]
        close_px = float(row["close"])
        if close_px <= 0:
            raise ValueError("close price must be positive")

        sigma_event = float(sigma.loc[orig_idx])
        t_event = row[ts_col]
        if t_event.tz is None or str(t_event.tz) != "Asia/Kolkata":
            raise ValueError("event timestamps must preserve Asia/Kolkata timezone")
        t_end = t_event + pd.Timedelta(minutes=max_horizon_minutes)

        pt_px, sl_px = _price_to_barriers(close_px, sigma_event, pt_k, sl_k)

        future = df_sorted[
            (df_sorted[symbol_col] == symbol)
            & (df_sorted[ts_col] > t_event)
            & (df_sorted[ts_col] <= t_end)
        ]

        label = 0
        ret = 0.0
        t_touch: pd.Timestamp | pd.NaT = pd.NaT
        if not future.empty:
            for _, frow in future.iterrows():
                price = float(frow["close"])
                ts_future = frow[ts_col]
                if price >= pt_px:
                    label = 1
                    t_touch = ts_future
                    ret = float(np.log(price / close_px))
                    break
                if price <= sl_px:
                    label = -1
                    t_touch = ts_future
                    ret = float(np.log(price / close_px))
                    break
            else:
                last_row = future.iloc[-1]
                price = float(last_row["close"])
                t_touch = last_row[ts_col]
                ret = float(np.log(price / close_px))
        else:
            t_touch = t_end
            ret = 0.0

        ambiguous = label == 0 and abs(ret) < min_move_k * sigma_event

        records.append(
            _EventRecord(
                event_id=event_id,
                symbol=symbol,
                t_event=t_event,
                t_end=t_end,
                pt_px=pt_px,
                sl_px=sl_px,
                t_touch=t_touch,
                label=label,
                ret=ret,
                ambiguous=ambiguous,
            )
        )

    events_df = pd.DataFrame([r.__dict__ for r in records])
    events_df = events_df[
        ["event_id", "symbol", "t_event", "t_end", "pt_px", "sl_px", "t_touch", "label", "ret", "ambiguous"]
    ]
    _LOG.info("constructed %d events", len(events_df))
    return events_df


def labels_from_events(events_df: pd.DataFrame, drop_ambiguous: bool) -> pd.DataFrame:
    """Return labels dataframe based on ambiguity policy."""

    required = [
        "event_id",
        "symbol",
        "t_event",
        "t_end",
        "pt_px",
        "sl_px",
        "t_touch",
        "label",
        "ret",
        "ambiguous",
    ]
    missing = [c for c in required if c not in events_df.columns]
    if missing:
        raise KeyError(f"events_df missing columns: {missing}")

    if drop_ambiguous:
        filtered = events_df[~events_df["ambiguous"]].copy()
    else:
        filtered = events_df.copy()
    filtered.reset_index(drop=True, inplace=True)
    _LOG.info("labels count after ambiguity policy: %d", len(filtered))
    return filtered
