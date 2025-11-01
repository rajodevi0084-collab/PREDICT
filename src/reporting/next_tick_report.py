"""Reporting helpers for next-tick task."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


@dataclass
class FoldMetrics:
    name: str
    mae: float
    hit_rate: float
    brier: float


@dataclass
class NextTickReport:
    folds: List[FoldMetrics]
    dm_pvalue: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "folds": [fold.__dict__ for fold in self.folds],
            "dm_pvalue": self.dm_pvalue,
        }

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


__all__ = ["FoldMetrics", "NextTickReport"]
