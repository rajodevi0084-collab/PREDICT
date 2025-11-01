import numpy as np

from src.metrics.next_tick import brier_3way, dm_test, hit_rate, tick_mae


def test_tick_mae():
    actual = np.array([100.0, 100.5, 101.0])
    pred = np.array([100.1, 100.4, 100.9])
    assert np.isclose(tick_mae(actual, pred, tick_size=0.05), 2.0)


def test_hit_rate():
    actual = np.array([1, -1, 0, 1])
    pred = np.array([1, 1, 0, -1])
    assert np.isclose(hit_rate(actual, pred), 1 / 3)


def test_brier_3way():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    labels = np.array([0, 1])
    score = brier_3way(probs, labels)
    assert 0 <= score <= 1


def test_dm_test():
    model = np.array([0.1, 0.2, -0.1])
    benchmark = np.zeros_like(model)
    p_value = dm_test(model, benchmark)
    assert 0 <= p_value <= 1
