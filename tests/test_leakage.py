import numpy as np

from src.validation.leakage_checks import shuffle_target_sanity_check


def test_shuffle_target_drives_metric_to_zero():
    rng = np.random.default_rng(42)
    y_true = rng.normal(size=128)
    y_pred = rng.normal(size=128)

    def correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.corrcoef(y_true, y_pred)[0, 1])

    baseline, shuffled = shuffle_target_sanity_check(y_true, y_pred, correlation, tolerance=0.25)
    assert np.all(np.abs(shuffled) < 0.25)
