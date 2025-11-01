"""Reporting utilities for next-bar models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

from ..metrics.next_bar import brier_3way, dm_test, hit_rate, mae_price


def build_next_bar_report(
    actual_close: Iterable[float],
    predicted_close: Iterable[float],
    baseline_last_close: Iterable[float],
    baseline_ema5: Iterable[float],
    probs: np.ndarray,
    labels: Iterable[int],
    conformal_bands: np.ndarray,
    output_dir: str | Path,
) -> Dict[str, float]:
    actual_arr = np.asarray(list(actual_close), dtype=float)
    pred_arr = np.asarray(list(predicted_close), dtype=float)
    last_close_arr = np.asarray(list(baseline_last_close), dtype=float)
    ema_arr = np.asarray(list(baseline_ema5), dtype=float)
    labels_arr = np.asarray(list(labels), dtype=int)

    mae_model = mae_price(actual_arr, pred_arr)
    mae_last = mae_price(actual_arr, last_close_arr)
    mae_ema = mae_price(actual_arr, ema_arr)
    brier = brier_3way(probs, labels_arr)
    hr = hit_rate(np.sign(actual_arr[1:] - actual_arr[:-1]), np.sign(pred_arr[1:] - pred_arr[:-1]))
    dm_last = dm_test(np.abs(actual_arr - pred_arr), np.abs(actual_arr - last_close_arr))
    dm_ema = dm_test(np.abs(actual_arr - pred_arr), np.abs(actual_arr - ema_arr))

    coverage = np.mean(
        (actual_arr >= conformal_bands[:, 0]) & (actual_arr <= conformal_bands[:, 1])
    )

    report = {
        "mae_model": mae_model,
        "mae_last_close": mae_last,
        "mae_ema5": mae_ema,
        "brier_3way": brier,
        "hit_rate": hr,
        "dm_vs_last": dm_last,
        "dm_vs_ema5": dm_ema,
        "conformal_coverage": float(coverage),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return report


__all__ = ["build_next_bar_report"]
