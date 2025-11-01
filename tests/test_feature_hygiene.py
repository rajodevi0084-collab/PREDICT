import numpy as np
import pandas as pd

from src.features.build_next_bar_ohlcv_v2 import apply_hygiene
from src.features.spec_loader import FeatureSpec


def test_apply_hygiene_limits_outliers_and_correlations():
    spec = FeatureSpec(
        name="test",
        raw={
            "groups": {
                "hygiene": {
                    "winsor_sigma": 1.0,
                    "vol_standardize": True,
                    "drop_corr_gt": 0.5,
                }
            },
            "keep_top_k": 2,
        },
    )

    X = pd.DataFrame(
        {
            "a": [0, 0, 0, 100],
            "b": [1, 1, 1, 1],
            "c": [0, 1, 2, 3],
        }
    )
    cleaned = apply_hygiene(X, spec)

    assert cleaned.abs().max().max() <= 5  # winsorised
    assert cleaned.shape[1] <= 3
