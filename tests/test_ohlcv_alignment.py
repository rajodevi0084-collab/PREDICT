import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from src.data.sequence_dataset import SequenceDataset
from src.features.build_next_bar_ohlcv_v2 import build_feature_matrix
from src.features.spec_loader import load_feature_spec
from src.labels.next_bar_targets import build_next_bar_targets


def test_labels_follow_log_return_shift():
    index = pd.date_range("2023-01-01", periods=12, freq="h")
    close = pd.Series(np.linspace(100.0, 110.0, len(index)), index=index)
    df = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000 + np.arange(len(index)),
        },
        index=index,
    )

    labels = build_next_bar_targets(df, dead_zone_abs_bp=1.0)
    expected_y_reg = np.log(df["close"].shift(-1) / df["close"]).dropna()
    assert_series_equal(labels["y_reg"], expected_y_reg, check_names=False)

    threshold = 1.0 / 1e4
    expected_cls = np.sign(expected_y_reg)
    expected_cls = expected_cls.mask(expected_y_reg.abs() < threshold, 0).astype(int)
    assert_series_equal(labels["y_cls"], expected_cls, check_names=False)


def test_features_align_with_labels_and_sequence_windows():
    index = pd.date_range("2023-01-01", periods=20, freq="h")
    base = np.arange(20, dtype=float)
    df = pd.DataFrame(
        {
            "open": base + 100,
            "high": base + 101,
            "low": base + 99,
            "close": base + 100.5,
            "volume": 1000 + base,
        },
        index=index,
    )

    spec = load_feature_spec("next_bar_ohlcv_v2")
    labels = build_next_bar_targets(df, dead_zone_abs_bp=1.0)
    X = build_feature_matrix(df, spec, labels=labels)

    assert X.index.difference(labels.index).empty
    assert labels.index.difference(X.index).empty

    dataset = SequenceDataset(X, labels["y_reg"], lookback=4)
    sample = dataset[5]
    assert sample.X_window.index.max() == labels.index[5]
