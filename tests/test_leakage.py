import numpy as np

from src.validation.leakage_checks import shuffle_target_sanity


def test_shuffle_target_drives_metric_to_zero():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(128, 4))
    y = rng.normal(size=128)

    result = shuffle_target_sanity(X, y, tolerance=0.25, n_permutations=5, random_state=7)
    shuffle_scores = result["shuffle_scores"]
    metric = result["metric"]

    if metric == "auc":
        expected = 0.5
    else:
        expected = 0.0
    assert np.allclose(np.nanmean(shuffle_scores), expected, atol=0.25)
