import numpy as np
import pandas as pd

from src.features.build_next_tick_v1 import make_price_features
from src.features.spec_loader import load_feature_spec
from src.labels.next_tick_targets import build_next_tick_labels
from src.validation.leakage_checks import assert_no_future_features


def test_labels_are_shifted_once():
    index = pd.date_range("2024-01-01", periods=10, freq="s", tz="UTC")
    df = pd.DataFrame(
        {
            "bid": 100 + np.arange(10, dtype=float),
            "ask": 100.5 + np.arange(10, dtype=float),
        },
        index=index,
    )
    df["mid"] = (df["bid"] + df["ask"]) / 2

    labels = build_next_tick_labels(df, tick_size=0.05, dead_zone_ticks=1)
    assert labels.index[0] == df.index[0]
    assert df.index[-1] not in labels.index

    spec = load_feature_spec("next_tick_v1")
    features = make_price_features(df, spec).iloc[: len(labels)]
    assert_no_future_features(features, labels.index)
