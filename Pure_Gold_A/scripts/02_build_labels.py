"""CLI to build events and labels from minute bars."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from pg.labeling.pipeline import build_labels
from pg.utils.logging_and_seed import get_logger, set_seed


_LOG = get_logger(__name__)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs/labeling.yaml"
    cfg = load_config(config_path)

    set_seed(int(cfg["reproducibility"]["seed"]))

    summary = build_labels(cfg["data"]["input_parquet_glob"], cfg)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = project_root / "logs/labels"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{run_id}.jsonl"

    record = {"run_id": run_id, **summary}
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record) + "\n")
    _LOG.info("wrote run log to %s", log_path)

    print(json.dumps(summary, indent=2))

    events_after = summary["events_after_min_move"]
    positives = summary["positives"]
    negatives = summary["negatives"]
    fail = False
    if events_after < 1000:
        _LOG.error("insufficient events_after_min_move=%d", events_after)
        fail = True
    if events_after > 0:
        pos_rate = positives / events_after
        neg_rate = negatives / events_after
        if pos_rate < 0.05 or neg_rate < 0.05:
            _LOG.error("class balance too skewed (pos=%.3f, neg=%.3f)", pos_rate, neg_rate)
            fail = True

    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
