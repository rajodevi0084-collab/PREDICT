"""Simplified training loop wiring together feature and label builders."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml

from ..data.price_source import get_price_series
from ..features.build_next_tick_v1 import (
    apply_hygiene,
    make_micro_features,
    make_orderbook_features,
    make_price_features,
    make_regime_features,
)
from ..features.spec_loader import FeatureSpec, load_feature_spec
from ..labels.next_tick_targets import build_next_tick_labels


def _load_project_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def train_next_tick(
    df: pd.DataFrame,
    *,
    orderbook: Optional[pd.DataFrame] = None,
    config_path: str | Path = "configs/project.yaml",
) -> Dict[str, Any]:
    """Run a lightweight training procedure for demonstration and testing."""

    config = _load_project_config(config_path)
    task_cfg = config["tasks"]["next_tick"]

    price_source = task_cfg.get("price_source", "mid")
    price_series = get_price_series(df, price_source)
    df = df.copy()
    df["mid"] = price_series

    spec = load_feature_spec(task_cfg["feature_set"])

    labels = build_next_tick_labels(
        df,
        tick_size=task_cfg["tick_size"],
        dead_zone_ticks=task_cfg["dead_zone_ticks"],
        price_col="mid",
    )

    price_features = make_price_features(df, spec)
    micro_features = make_micro_features(df, spec)
    regime_features = make_regime_features(df, spec)

    feature_frames = [price_features, micro_features, regime_features]
    if orderbook is not None:
        feature_frames.append(make_orderbook_features(orderbook, spec))

    X = pd.concat(feature_frames, axis=1).iloc[: len(labels)]
    X = apply_hygiene(X, spec)

    summary = {
        "n_samples": int(len(labels)),
        "n_features": int(X.shape[1]),
        "labels": labels.describe().to_dict(),
    }
    return summary


__all__ = ["train_next_tick"]
