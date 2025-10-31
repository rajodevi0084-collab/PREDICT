# Labels & Splits Pipeline Overview

This document describes the end-to-end process for generating supervised event labels and backtest splits for NSE 1-minute equities. The pipeline consumes raw minute bars, produces event labels suitable for classification, and enforces strict quality gates before any downstream modeling may proceed.

## Inputs
- Minute-bar parquet files matching `configs/labeling.yaml:data.input_parquet_glob`.
- Each parquet must contain columns `[symbol, ts, open, high, low, close, volume]`.
- Timestamps are timezone-aware (`Asia/Kolkata`).

## Processing Order
1. **02_build_labels.py** — loads bars, estimates volatility, samples events via CUSUM, applies triple-barrier labeling, and writes:
   - `events_parquet`
   - `labels_parquet`
   - JSONL summary log under `logs/labels/`.
2. **03_make_splits.py** — ingests labels and constructs purged, embargoed cross-validation folds saved to `splits_json`.
3. **04_label_quality_report.py** — runs a sanity-check baseline on fold 0, logs metrics, and writes the quality report markdown.

## Outputs
- **Events parquet**: per-event metadata and barrier prices.
- **Labels parquet**: event outcomes, returns, and ambiguity flags.
- **Splits JSON**: purged time-ordered CV assignment for each event.
- **Quality report**: markdown summarizing baseline diagnostics and gate status.

## Quality Gate
The pipeline must achieve the following metrics on the first purged fold (fold 0):
- Area Under ROC Curve (AUC) > 0.60
- Brier Score < 0.21

Processing may continue only if both thresholds are met; otherwise, consult the troubleshooting guidance in `OPERATIONS_labels_splits.md`.

## New Files
- `configs/labeling.yaml` — declarative configuration for labeling CLI; load with `python scripts/02_build_labels.py`.
- `src/pg/labeling/volatility.py` — volatility estimators; imported by the labeling pipeline.
- `src/pg/labeling/cusum.py` — CUSUM event sampler; used within the pipeline.
- `src/pg/labeling/triple_barrier.py` — triple-barrier event construction and labeling utilities; consumed by the pipeline.
- `src/pg/labeling/pipeline.py` — orchestrates the labeling workflow; invoked by CLI.
- `src/pg/splits/purged_cv.py` — purged K-fold splitter with embargo; called by splits CLI.
- `src/pg/splits/utils.py` — helper utilities for index management and embargo logic.
- `src/pg/metrics/brier_auc.py` — metric helpers for the quality report; import into diagnostics.
- `src/pg/utils/logging_and_seed.py` — central logging and deterministic seeding; used by all scripts.
- `scripts/02_build_labels.py` — CLI entrypoint to build events and labels.
- `scripts/03_make_splits.py` — CLI entrypoint to generate purged CV splits.
- `scripts/04_label_quality_report.py` — CLI entrypoint for baseline diagnostics and gate evaluation.
- `tests/test_labeling_pipeline.py` — unit tests covering volatility, CUSUM, triple-barrier, and purged CV.
- `OPERATIONS_labels_splits.md` — operational runbook for executing the pipeline.
- `contracts/labels_contract.md` — schema contracts for inputs and outputs.
