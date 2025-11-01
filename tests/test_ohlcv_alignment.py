import numpy as np
import pandas as pd

from src.features.build_next_bar_ohlcv_v2 import build_feature_matrix
from src.features.spec_loader import load_feature_spec
from src.labels.next_bar_targets import build_next_bar_targets


def test_features_align_with_next_bar_labels():
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
    labels = build_next_bar_targets(df, dead_zone_abs_bp=2.0)
    X = build_feature_matrix(df, spec, labels=labels)

    assert len(X) == len(labels)
    assert labels.index.max() < df.index.max()
