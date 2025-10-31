"""Helper utilities for time-ordered cross-validation splits."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def index_from_events(events_df: pd.DataFrame) -> np.ndarray:
    """Return a stable integer index spanning [0, N)."""

    if events_df.empty:
        return np.array([], dtype=int)
    return np.arange(len(events_df), dtype=int)


def slice_by_time(events_df: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp) -> np.ndarray:
    """Return event indices whose [t_event, t_end] lie within [t0, t1]."""

    if t0 > t1:
        raise ValueError("t0 must be <= t1")
    mask = (events_df["t_event"] >= t0) & (events_df["t_end"] <= t1)
    return np.flatnonzero(mask.values)


def apply_embargo(
    events_df: pd.DataFrame, idx: Iterable[int], embargo_minutes: int
) -> np.ndarray:
    """Return indices falling within the embargo window around selected events."""

    idx = np.asarray(list(idx), dtype=int)
    if idx.size == 0:
        return np.array([], dtype=int)
    if embargo_minutes < 0:
        raise ValueError("embargo_minutes must be non-negative")

    window_start = events_df.loc[idx, "t_event"].min() - pd.Timedelta(minutes=embargo_minutes)
    window_end = events_df.loc[idx, "t_end"].max() + pd.Timedelta(minutes=embargo_minutes)

    overlap = (events_df["t_event"] <= window_end) & (events_df["t_end"] >= window_start)
    return np.flatnonzero(overlap.values)
