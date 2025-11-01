"""Sequence dataset utilities with strict temporal alignment checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass
class SequenceSample:
    X_window: pd.DataFrame
    y_reg: float


class SequenceDataset:
    """Sliding-window dataset ensuring feature/label alignment at time ``t``."""

    def __init__(self, X: pd.DataFrame, y_reg: pd.Series, lookback: int) -> None:
        if lookback <= 0:
            raise ValueError("lookback must be positive")
        if X.empty or y_reg.empty:
            raise ValueError("Features and targets must contain data")

        self.lookback = int(lookback)
        self.y_reg = y_reg.sort_index()
        self.X = X.sort_index()

        if not self.X.index.equals(self.y_reg.index):
            missing = self.y_reg.index.difference(self.X.index)
            if not missing.empty:
                raise ValueError("Features missing indices present in targets")
            self.X = self.X.reindex(self.y_reg.index)

        self._positions = self.X.index.get_indexer(self.y_reg.index)
        if np.any(self._positions < 0):
            raise ValueError("Failed to align feature windows with targets")

    def __len__(self) -> int:
        return len(self.y_reg)

    def __iter__(self) -> Iterator[SequenceSample]:
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, index: int) -> SequenceSample:
        if index < 0 or index >= len(self):
            raise IndexError("SequenceDataset index out of range")
        label_idx = self.y_reg.index[index]
        end_pos = self._positions[index]
        start_pos = max(0, end_pos - self.lookback + 1)
        X_window = self.X.iloc[start_pos : end_pos + 1]
        if X_window.empty:
            raise ValueError("Computed empty window; check lookback and index ordering")
        assert (
            X_window.index.max() == label_idx
        ), "Feature window must terminate at the label timestamp"
        return SequenceSample(X_window=X_window, y_reg=float(self.y_reg.iloc[index]))


__all__ = ["SequenceDataset", "SequenceSample"]
