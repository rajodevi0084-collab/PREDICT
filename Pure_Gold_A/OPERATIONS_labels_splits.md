# Labels & Splits Runbook

1. Edit `configs/labeling.yaml` with the desired data glob and parameter tweaks.
2. Run `python scripts/02_build_labels.py` to produce events and labels.
3. Run `python scripts/03_make_splits.py` to create purged CV folds and write the JSON splits.
4. Run `python scripts/04_label_quality_report.py` to evaluate the baseline gate on fold 0.

## Gate Criteria
Proceed only if the quality report shows **AUC > 0.60** and **Brier < 0.21** for fold 0.

## Troubleshooting & Parameter Nudges
- Lower `cusum.k_sigma` to the 0.6–0.9 range to increase event density when counts are low.
- Adjust `triple_barrier.min_move_k` between 0.25–0.6 to balance noise filtering versus coverage.
- Increase `volatility.span_minutes` to smooth σ estimates if labels appear too jumpy.
- Increase `splits.embargo_minutes` when leakage is suspected.

## Reading Outputs
- **Build labels CLI** prints counts for bars, raw events, post-filter events, and label balance. A surge in dropped events signals an aggressive `min_move_k`.
- **Split CLI** prints per-fold train/validation sizes; zero leakage is enforced internally.
- **Quality report** lists gate metrics, class balance, and ambiguous counts. Failure typically traces to misaligned labels, volatility scaling, or overly strict thresholds.

## Failure Signatures
- **Low event counts**: raise from CUSUM threshold; reduce `cusum.k_sigma`.
- **Poor AUC**: inspect feature alignment, verify sigma smoothing, and consider lowering `min_move_k`.
- **High Brier**: check for extreme probability calibration, revisit triple-barrier parameters, or confirm the embargo prevents leakage.
