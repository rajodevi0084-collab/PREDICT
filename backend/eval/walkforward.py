"""Walk-forward evaluation utilities."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


@dataclass
class Window:
    """Represents a single walk-forward window."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class WindowReport:
    """Report returned by :class:`WalkForwardRunner` for each window."""

    window: Window
    metrics: dict
    passed_gates: bool
    path: Path


class WalkForwardRunner:
    """Perform rolling walk-forward training/evaluation cycles."""

    def __init__(self, train_span: timedelta, test_span: timedelta, step: timedelta, tz: str) -> None:
        self.train_span = train_span
        self.test_span = test_span
        self.step = step
        self.tz = tz

    def split_ranges(self, calendar: Sequence[pd.Timestamp]) -> List[Window]:
        """Split the provided calendar into sequential windows."""

        if not calendar:
            return []
        calendar = sorted(calendar)
        windows: List[Window] = []
        start = calendar[0]
        while True:
            train_start = start
            train_end = train_start + self.train_span
            test_start = train_end
            test_end = test_start + self.test_span
            if test_end > calendar[-1]:
                break
            windows.append(
                Window(
                    train_start=pd.Timestamp(train_start, tz=self.tz),
                    train_end=pd.Timestamp(train_end, tz=self.tz),
                    test_start=pd.Timestamp(test_start, tz=self.tz),
                    test_end=pd.Timestamp(test_end, tz=self.tz),
                )
            )
            start = start + self.step
            if start + self.train_span >= calendar[-1]:
                break
        return windows

    def run(self, model_trainer, rl_trainer, data_loader) -> List[WindowReport]:
        """Execute walk-forward process and return per-window reports."""

        calendar = data_loader.calendar()
        windows = self.split_ranges(calendar)
        reports: List[WindowReport] = []
        for idx, window in enumerate(windows):
            train_data = data_loader.load(window.train_start, window.train_end)
            test_data = data_loader.load(window.test_start, window.test_end)
            model = model_trainer.train(train_data)
            predictions = model.predict(test_data)
            rl_trades = rl_trainer.evaluate(window)
            metrics = rl_trainer.evaluate_metrics(predictions, rl_trades)
            gates = (
                metrics.get("sharpe", 0) >= 1.5
                and metrics.get("sortino", 0) >= 2.0
                and metrics.get("profit_factor", 0) >= 1.3
                and metrics.get("mdd", 0) <= 0.15
            )
            out_dir = Path("artifacts") / "walkforward"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"window_{idx:02d}.json"
            with path.open("w", encoding="utf-8") as fh:
                json.dump({"window": window.__dict__, "metrics": metrics, "passed_gates": gates}, fh, default=str)
            reports.append(WindowReport(window=window, metrics=metrics, passed_gates=gates, path=path))
        return reports

    def aggregate(self, reports: Iterable[WindowReport]) -> dict:
        """Aggregate per-window reports into a summary dictionary."""

        metrics = [r.metrics for r in reports]
        if not metrics:
            return {}
        summary: dict[str, float] = {}
        for key in metrics[0].keys():
            values = [m.get(key, float("nan")) for m in metrics]
            arr = pd.Series(values, dtype=float).dropna()
            if arr.empty:
                continue
            mean = arr.mean()
            std = arr.std(ddof=1) if len(arr) > 1 else 0.0
            summary[key] = mean
            summary[f"{key}_ci95"] = 1.96 * std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        summary["num_windows"] = len(metrics)
        return summary

    def run_with_params(self, params: dict):  # pragma: no cover - hook for HPO
        raise NotImplementedError("Provide run_with_params when using Optuna search")
