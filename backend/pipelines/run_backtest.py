"""Unified backtest entry-point wiring together data, ML and RL layers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from backend.data.labels import LabelGenerator
from backend.eval.metrics import join_pred_actual, leak_guard, xcorr_peak_lag
from backend.logging.jsonl import open_run_logger
from backend.ml.predict import InMemoryPredictionStore, PredictionBundle
from backend.oms.costs import CostModel


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_alignment(pred_store: InMemoryPredictionStore, ohlcv: pd.DataFrame, horizons: list[int]) -> None:
    for h in horizons:
        joined = join_pred_actual(pred_store, ohlcv, horizon=h)
        if joined.empty:
            continue
        lag = xcorr_peak_lag(joined["yhat"], joined[f"actual_ret_h{h}"])
        if lag != h:
            raise AssertionError(f"Cross correlation peak lag {lag} != {h}")


def run_backtest(config_path: str, data_loader, forecaster, rl_trainer) -> Dict[str, Any]:
    config = load_config(config_path)
    horizons = list(config["data"]["horizons"])
    epsilon = float(config["data"]["neutral_epsilon_bp"])
    candle_conv = config["data"]["candle_time_convention"]
    cost_model = CostModel(config["costs"])

    df = data_loader.load()
    leak_guard(list(df.columns))
    label_gen = LabelGenerator(horizons=horizons, epsilon_bp=epsilon, candle_time_convention=candle_conv)
    labels = label_gen.generate(df)

    prediction_store = InMemoryPredictionStore()
    for features in data_loader.iter_features(df):
        bundle: PredictionBundle = forecaster.predict(features)
        prediction_store.append(bundle)

    ensure_alignment(prediction_store, df, horizons)

    run_id = data_loader.run_id()
    reports_dir = Path("artifacts") / "backtest" / run_id
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open_run_logger(run_id) as logger:
        for trade in rl_trainer.run(prediction_store, labels, cost_model):
            logger.write(trade)

    summary_path = reports_dir / "summary.json"
    summary = rl_trainer.summary()
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return {"summary_path": summary_path, "run_id": run_id}


def main() -> None:
    raise RuntimeError("This module expects to be orchestrated by the application; provide concrete data loaders and trainers.")


if __name__ == "__main__":  # pragma: no cover
    main()
