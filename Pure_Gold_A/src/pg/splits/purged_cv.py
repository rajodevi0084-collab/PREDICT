"""Purged, embargoed cross-validation splitters."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from pg.splits.utils import apply_embargo, index_from_events, slice_by_time
from pg.utils.logging_and_seed import get_logger


_LOG = get_logger(__name__)


def make_time_blocks(events_df: pd.DataFrame, n_blocks: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Split the global time span into contiguous blocks."""

    if n_blocks <= 0:
        raise ValueError("n_blocks must be positive")
    if events_df.empty:
        raise ValueError("events dataframe is empty")

    t_start = events_df["t_event"].min()
    t_stop = events_df["t_end"].max()
    if t_start >= t_stop:
        raise ValueError("time range must have positive duration")

    total_seconds = (t_stop - t_start).total_seconds()
    block_seconds = total_seconds / n_blocks
    blocks: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

    for i in range(n_blocks):
        block_start = t_start + pd.Timedelta(seconds=block_seconds * i)
        if i == n_blocks - 1:
            block_end = t_stop
        else:
            block_end = t_start + pd.Timedelta(seconds=block_seconds * (i + 1))
        blocks.append((block_start, block_end))
    return blocks


def purged_kfold(
    events_df: pd.DataFrame,
    n_folds: int,
    embargo_minutes: int,
) -> List[Dict[str, Any]]:
    """Generate purged cross-validation splits with embargo."""

    if n_folds <= 1:
        raise ValueError("n_folds must be at least 2")
    if embargo_minutes < 0:
        raise ValueError("embargo_minutes must be non-negative")
    if events_df.empty:
        raise ValueError("events dataframe is empty")

    required_cols = {"event_id", "t_event", "t_end", "symbol"}
    missing = required_cols.difference(events_df.columns)
    if missing:
        raise KeyError(f"events_df missing columns: {sorted(missing)}")

    n_blocks = int(events_df.attrs.get("n_blocks", n_folds))
    if n_blocks < n_folds:
        n_blocks = n_folds
    blocks = make_time_blocks(events_df, n_blocks)
    val_blocks = blocks[-n_folds:]

    all_indices = index_from_events(events_df)
    splits: List[Dict[str, Any]] = []

    for fold_id, (val_start, val_end) in enumerate(val_blocks):
        val_idx = slice_by_time(events_df, val_start, val_end)
        if val_idx.size == 0:
            raise ValueError(f"validation fold {fold_id} is empty")

        purge_mask = (events_df["t_event"] <= val_end) & (events_df["t_end"] >= val_start)
        purge_idx = np.flatnonzero(purge_mask.values)

        train_idx = np.setdiff1d(all_indices, purge_idx, assume_unique=True)
        embargo_idx = apply_embargo(events_df, val_idx, embargo_minutes)
        train_idx = np.setdiff1d(train_idx, embargo_idx, assume_unique=True)

        if np.intersect1d(train_idx, val_idx).size != 0:
            raise AssertionError("train and validation indices overlap")

        splits.append(
            {
                "fold": fold_id,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "val_window": [val_start.isoformat(), val_end.isoformat()],
            }
        )
    return splits


def write_splits_to_json(splits: List[Dict[str, Any]], path: str | Path) -> None:
    """Persist splits to JSON with numpy arrays serialised as lists."""

    serialisable = []
    for item in splits:
        serialisable.append(
            {
                "fold": int(item["fold"]),
                "train_idx": item["train_idx"].tolist(),
                "val_idx": item["val_idx"].tolist(),
                "val_window": item["val_window"],
            }
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(serialisable, fp, indent=2)
    _LOG.info("wrote %d folds to %s", len(splits), path)
