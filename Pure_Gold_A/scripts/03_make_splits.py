"""CLI to construct purged, embargoed cross-validation splits."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

from pg.splits.purged_cv import purged_kfold, write_splits_to_json
from pg.utils.logging_and_seed import get_logger


_LOG = get_logger(__name__)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs/labeling.yaml"
    cfg = load_config(config_path)

    labels_path = Path(cfg["io"]["labels_parquet"])
    if not labels_path.exists():
        _LOG.error("labels parquet not found at %s", labels_path)
        return 1

    events_df = pd.read_parquet(labels_path)
    if events_df.empty:
        _LOG.error("labels dataframe is empty")
        return 1

    n_folds = int(cfg["splits"]["n_folds"])
    embargo_minutes = int(cfg["splits"]["embargo_minutes"])
    n_blocks = max(int(cfg["splits"]["backtest_blocks"]), n_folds)
    events_df.attrs["n_blocks"] = n_blocks

    splits = purged_kfold(events_df, n_folds=n_folds, embargo_minutes=embargo_minutes)

    for fold in splits:
        val_count = len(fold["val_idx"])
        train_count = len(fold["train_idx"])
        print(f"fold={fold['fold']} train={train_count} val={val_count} window={fold['val_window']}")
        if val_count < 500:
            _LOG.error("validation count %d < 500 for fold %d", val_count, fold["fold"])
            return 1

    splits_path = cfg["io"]["splits_json"]
    write_splits_to_json(splits, splits_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
