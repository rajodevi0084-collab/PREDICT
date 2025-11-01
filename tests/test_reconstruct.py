import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from src.postproc.reconstruct_price import reconstruct_close


def test_reconstruct_close_identity_series():
    index = pd.date_range("2023-01-01", periods=5, freq="h")
    close = pd.Series(np.linspace(100.0, 104.0, len(index)), index=index)
    y_reg_hat = pd.Series(0.0, index=index)

    reconstructed = reconstruct_close(close, y_reg_hat)
    assert_series_equal(reconstructed, close)


def test_reconstruct_close_scalar():
    assert reconstruct_close(101.5, 0.0) == 101.5
