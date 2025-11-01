"""High-level orchestration for the next-bar OHLCV task."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

from ..calibration.calibrate_cls import TemperatureScaler
from ..calibration.calibrate_reg import LinearCalibration
from ..features.spec_loader import load_feature_spec, resolve_feature_builder
from ..labels.next_bar_targets import build_next_bar_targets
from ..metrics.next_bar import mae_price
from ..reporting.next_bar_report import build_next_bar_report
from ..validation.leakage_checks import assert_no_future_features, shuffle_target_sanity
from ..validation.tscv_rolling import rolling_windows


def _load_project_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def train_next_bar(
    df: pd.DataFrame,
    *,
    config_path: str | Path = "configs/project.yaml",
) -> Dict[str, Any]:
    """Run a lightweight pipeline to validate the next-bar configuration."""

    config = _load_project_config(config_path)
    task_cfg = config["tasks"]["next_bar_ohlcv"]

    spec = load_feature_spec(task_cfg["feature_set"])
    labels = build_next_bar_targets(df, dead_zone_abs_bp=task_cfg["dead_zone_abs_bp"])

    builder = resolve_feature_builder(spec.name)
    X = builder.build_feature_matrix(df, spec, labels=labels)
    assert_no_future_features(X, labels.index)

    shuffle_target_sanity(X, labels["y_reg"], tolerance=0.1, n_permutations=3)

    close = df["close"].astype(float)
    actual_next_close = close.shift(-1).iloc[:-1]
    baseline_last = close.shift(1).iloc[1 : len(labels) + 1]
    ema5 = close.ewm(span=5, adjust=False).mean().shift(1).iloc[1 : len(labels) + 1]
    ema5 = ema5.reindex(labels.index, method="nearest")

    predicted_close = baseline_last.reindex(labels.index).fillna(method="bfill")
    mae_vs_last = mae_price(actual_next_close.reindex(labels.index), predicted_close)

    # Calibrators are placeholders for now.
    cls_calibrator = TemperatureScaler()
    reg_calibrator = LinearCalibration()
    cls_calibrator.fit(np.zeros((len(labels), 3)), np.zeros(len(labels)))
    reg_calibrator.fit(labels["y_reg"], labels["y_reg"])

    probs = np.full((len(labels), 3), 1 / 3)
    conformal_bands = np.column_stack(
        [predicted_close * 0.99, predicted_close, predicted_close * 1.01]
    )

    report_dir = Path("reports/next_bar")
    report = build_next_bar_report(
        actual_next_close.reindex(labels.index, method="nearest"),
        predicted_close,
        baseline_last.reindex(labels.index, method="nearest"),
        ema5,
        probs,
        labels["y_cls"],
        conformal_bands[:, [0, 2]],
        report_dir,
    )

    splits = list(
        rolling_windows(
            labels.index,
            train_size=max(len(labels) // 2, 1),
            test_size=max(len(labels) // 4, 1),
            purge_bars=task_cfg.get("purge_bars", 0),
            embargo_bars=task_cfg.get("embargo_bars", 0),
        )
    )

    summary = {
        "n_samples": int(len(labels)),
        "n_features": int(X.shape[1]),
        "baseline_mae": mae_vs_last,
        "report": report,
        "n_splits": len(splits),
    }
    return summary


__all__ = ["train_next_bar"]
