"""Rolling time-series cross-validation helpers with purge and embargo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import pandas as pd


@dataclass
class RollingSplit:
    train_indices: Sequence[int]
    test_indices: Sequence[int]


def rolling_windows(
    index: pd.Index,
    train_size: int,
    test_size: int,
    *,
    purge_ticks: int = 0,
    embargo_ticks: int = 0,
) -> Iterator[RollingSplit]:
    n = len(index)
    start = 0
    while True:
        train_start = start
        train_end = train_start + train_size
        test_start = train_end + purge_ticks
        test_end = test_start + test_size
        embargo_end = test_end + embargo_ticks

        if embargo_end > n:
            break

        yield RollingSplit(
            train_indices=list(range(train_start, train_end)),
            test_indices=list(range(test_start, test_end)),
        )

        start += test_size


__all__ = ["RollingSplit", "rolling_windows"]
