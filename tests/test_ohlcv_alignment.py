import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from src.features.build_next_bar_ohlcv_v2 import build_feature_matrix, rank_and_select
from src.features.spec_loader import load_feature_spec
from src.labels.next_bar_targets import build_next_bar_targets


def _sample_ohlcv(n: int = 32) -> pd.DataFrame:
    index = pd.date_range("2023-01-01", periods=n, freq="h")
    close = pd.Series(np.linspace(100.0, 110.0, n), index=index)
    df = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000 + np.arange(n),
        },
        index=index,
    )
    return df


def test_labels_follow_log_return_shift():
    df = _sample_ohlcv(16)
    y_reg, y_cls = build_next_bar_targets(df, dead_zone_abs_bp=1.0, horizon_bars=1)

    expected_y_reg = np.log(df["close"].shift(-1) / df["close"]).dropna()
    assert_series_equal(y_reg, expected_y_reg, check_names=False)

    threshold = 1.0 / 1e4
    expected_cls = np.sign(expected_y_reg)
    expected_cls = expected_cls.mask(expected_y_reg.abs() < threshold, 0.0).astype(int)
    assert_series_equal(y_cls, expected_cls, check_names=False)


def test_features_align_with_labels_after_selection():
    df = _sample_ohlcv(40)
    spec = load_feature_spec("next_bar_ohlcv_v2")
    y_reg, _ = build_next_bar_targets(df, dead_zone_abs_bp=1.0, horizon_bars=1)

    X_raw = build_feature_matrix(df, spec)
    X = X_raw.loc[y_reg.index]
    X = rank_and_select(X, y_reg, spec)

    assert len(X) == len(y_reg)
    assert (X.index == y_reg.index).all()
