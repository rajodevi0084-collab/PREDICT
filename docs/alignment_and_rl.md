# Alignment, Reinforcement Learning, and Costs

This document summarises the updated alignment strategy, RL stack, and cost model introduced in the refactor. It is intended for new contributors who need a concise overview before diving into the code.

## Horizon semantics

- **made_at** is the timestamp at which the forecast is produced (bar *t*).
- **valid_at** is the timestamp where the targeted event materialises (bar *t+h*).
- The [`LabelGenerator`](../backend/data/labels.py) emits forward returns `ret_h{h}` and classes `cls_h{h}` measured at `made_at`, alongside a `valid_at` index (shifted by `h` for end-of-bar candles).
- Downstream consumers (forecaster, charts, evaluation) must reindex predictions to `valid_at` before comparing with realised returns. This eliminates the orange “tail” that previously lagged the blue price line.

```
made_at (t)     valid_at (t+h)
    |-------------------------|
close[t]                 close[t+h]
```

## Why the chart looked shifted

Earlier versions plotted predictions at `made_at`. Because realised prices are only known at `valid_at`, the orange curve trailed the blue close series by `h` bars. The revised [`PriceChart`](../frontend/components/PriceChart.tsx) renders predictions at `valid_at`, highlights the default horizon, and warns when timebases differ so misalignments are immediately visible.

## Reinforcement learning design

- **State**: concatenation of feature vector, per-horizon forecasts (`yhat_reg`, class probabilities), current position, normalised cash, time progress, and recent return.
- **Actions**: `Flat`, `Long1`, `Long2`, `Short1`, `Short2` mapping to discrete exposure levels. See [`ACTION_MAP`](../backend/rl/env.py).
- **Reward**: delta in marked-to-market portfolio minus trading costs, turnover penalty (`turnover_lambda · |Δpos|`), and drawdown penalty (`dd_lambda · drawdown`).
- **Constraints**: configurable kill-switch (`max_daily_loss_pct`), per-symbol exposure caps (apply in policy layer), optional EOD flattening in the [`ExecutionStub`](../backend/oms/execution_stub.py).
- **Logging**: each step returns a cost breakdown and kill-switch flag in the `info` dict for downstream attribution.

## India cost model

[`CostModel`](../backend/oms/costs.py) ingests parameters from `config/model.yaml` and expands every trade into a [`CostBreakdown`](../backend/oms/costs.py). Components include brokerage, STT, GST, stamp duty, exchange charges, half-spread slippage, and market-impact costs (`impact_coeff × qty / ADV`). Tests verify that costs grow with order size.

## Walk-forward evaluation

[`WalkForwardRunner`](../backend/eval/walkforward.py) splits the trading calendar into rolling windows (`train_span`, `test_span`, `step`). Each window produces a JSON report containing:

- Metrics (Sharpe, Sortino, Profit Factor, MDD, Turnover, Hit-rate, etc.).
- Gate status: Sharpe ≥ 1.5, Sortino ≥ 2.0, PF ≥ 1.3, MDD ≤ 15%.
- Stored under `artifacts/walkforward/window_XX.json`.

`aggregate()` adds confidence intervals across windows, and the Optuna wrapper expects a `run_with_params` hook to evaluate hyper-parameter candidates.

## Coverage vs Return curve

The new [`MetricsPanel`](../frontend/components/MetricsPanel.tsx) fetches `/api/report`, renders directional accuracy by horizon, and plots “Coverage vs Return”. Each point reflects the average realised return conditioned on a minimum confidence threshold. Use it to tune `coverage_tau` in [`PolicyGate`](../backend/rl/policy_gating.py): targeting ~65–70% accuracy usually requires selective coverage with higher confidence cut-offs.

## Putting it together

`backend/pipelines/run_backtest.py` orchestrates the full flow:

1. Load `config/model.yaml` and instantiate `LabelGenerator`.
2. Train/load the forecaster, generate [`PredictionBundle`](../backend/ml/predict.py) objects, and store them with `made_at`/`valid_at` metadata.
3. Run RL training via [`RLTrainer`](../backend/rl/train_rl.py) on walk-forward windows.
4. Apply the full India cost model and log every trade using [`open_run_logger`](../backend/logging/jsonl.py).
5. Fail fast if alignment is broken (`xcorr_peak_lag != h`) or `leak_guard` detects future-looking features.

With these changes the system maintains consistent timing semantics end-to-end, enabling trustworthy performance diagnostics and policy gating.
