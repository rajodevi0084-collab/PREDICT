import numpy as np
import pandas as pd

from src.data.sequence_dataset import SequenceDataset


def test_sequence_window_aligns_with_label_index():
    index = pd.date_range("2023-02-01", periods=12, freq="h")
    X = pd.DataFrame({"feature": np.arange(12, dtype=float)}, index=index)
    y = pd.Series(np.linspace(-0.01, 0.01, 12), index=index)

    dataset = SequenceDataset(X, y, lookback=4)

    for i in range(len(dataset)):
        sample = dataset[i]
        assert sample.X_window.index.max() == y.index[i]
