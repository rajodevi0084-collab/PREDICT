import numpy as np

from src.metrics.next_bar import brier_3way, dm_test, hit_rate, mae_price


def test_mae_price():
    actual = np.array([10.0, 11.0, 12.0])
    pred = np.array([9.5, 10.5, 12.5])
    assert np.isclose(mae_price(actual, pred), 0.5)


def test_brier_and_hit_rate():
    probs = np.array([[0.2, 0.5, 0.3], [0.1, 0.7, 0.2]])
    labels = np.array([-1, 0])
    score = brier_3way(probs, labels)
    assert 0.0 <= score <= 1.0

    actual = np.array([1, -1, 1])
    predicted = np.array([0.5, -0.1, 0.9])
    hr = hit_rate(actual, predicted)
    assert 0.0 <= hr <= 1.0


def test_dm_test():
    model_err = np.array([0.1, 0.2, 0.3, 0.2])
    benchmark_err = np.array([0.2, 0.2, 0.4, 0.3])
    p_value = dm_test(model_err, benchmark_err)
    assert 0.0 <= p_value <= 1.0
