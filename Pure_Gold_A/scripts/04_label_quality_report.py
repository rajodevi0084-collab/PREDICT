"""Baseline diagnostics for label quality gate."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression

from pg.metrics.brier_auc import auc_binary, brier_score, calibration_summary
from pg.utils.logging_and_seed import get_logger, set_seed


_LOG = get_logger(__name__)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _prepare_features(labels_df: pd.DataFrame) -> pd.Series:
    labels_df = labels_df.sort_values(["symbol", "t_event"]).copy()
    feat = labels_df.groupby("symbol")["ret"].shift(1)
    feat = feat.fillna(0.0)
    return np.sign(feat)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs/labeling.yaml"
    cfg = load_config(config_path)
    set_seed(int(cfg["reproducibility"]["seed"]))

    labels_path = Path(cfg["io"]["labels_parquet"])
    splits_path = Path(cfg["io"]["splits_json"])
    events_path = Path(cfg["io"]["events_parquet"])

    if not labels_path.exists() or not splits_path.exists():
        _LOG.error("required inputs missing (labels or splits)")
        return 1

    labels_df = pd.read_parquet(labels_path)
    splits = json.loads(splits_path.read_text(encoding="utf-8"))

    fold0 = next((fold for fold in splits if fold["fold"] == 0), None)
    if fold0 is None:
        _LOG.error("fold 0 not found in splits")
        return 1

    features = _prepare_features(labels_df)
    labels_df = labels_df.assign(ret_sign_prev1=features.values)

    X = labels_df[["ret_sign_prev1"]].values
    y = (labels_df["label"] == 1).astype(int).values

    train_idx = np.array(fold0["train_idx"], dtype=int)
    val_idx = np.array(fold0["val_idx"], dtype=int)

    model = LogisticRegression(solver="lbfgs")
    model.fit(X[train_idx], y[train_idx])
    prob_val = model.predict_proba(X[val_idx])[:, 1]
    y_val = y[val_idx]

    brier = brier_score(prob_val, y_val)
    auc = auc_binary(prob_val, y_val)
    calibration_summary(prob_val, y_val)

    pos_ratio = float(y_val.mean())
    neg_ratio = float(1.0 - pos_ratio)

    ambiguous_dropped = 0
    if events_path.exists():
        events_df = pd.read_parquet(events_path)
        ambiguous_total = int(events_df["ambiguous"].sum())
        ambiguous_remaining = int(labels_df.get("ambiguous", pd.Series(dtype=bool)).sum())
        ambiguous_dropped = max(ambiguous_total - ambiguous_remaining, 0)

    gate_pass = auc > 0.60 and brier < 0.21

    report_path = Path(cfg["io"]["quality_report_md"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("# Label Quality Report\n\n")
        fh.write(f"Fold: 0\n\n")
        fh.write(f"- Brier Score: {brier:.4f}\n")
        fh.write(f"- ROC AUC: {auc:.4f}\n")
        fh.write(f"- Positive ratio: {pos_ratio:.3f}\n")
        fh.write(f"- Negative ratio: {neg_ratio:.3f}\n")
        fh.write(f"- Ambiguous dropped: {ambiguous_dropped}\n")
        fh.write(f"- Gate Status: {'PASS' if gate_pass else 'FAIL'} (AUC>0.60 & Brier<0.21)\n")
        if not gate_pass:
            fh.write("\n## Next Checks\n")
            fh.write("- Verify label alignment.\n")
            fh.write("- Inspect volatility scale.\n")
            fh.write("- Adjust triple_barrier.min_move_k (0.25–0.6).\n")
            fh.write("- Tune cusum.k_sigma (0.6–0.9).\n")
            fh.write("- Increase embargo_minutes if leakage suspected.\n")

    print(
        json.dumps(
            {
                "fold": 0,
                "brier": brier,
                "auc": auc,
                "pos_ratio": pos_ratio,
                "neg_ratio": neg_ratio,
                "ambiguous_dropped": ambiguous_dropped,
                "gate_pass": gate_pass,
            },
            indent=2,
        )
    )

    if not gate_pass:
        print(
            "Gate failed: inspect label alignment, sigma scaling, min_move_k, cusum threshold, or embargo settings.",
            file=sys.stderr,
        )
    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
