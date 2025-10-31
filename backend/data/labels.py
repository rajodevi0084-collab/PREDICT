"""Label generation utilities with strict alignment guarantees.

The module provides :class:`LabelGenerator` which produces multi-horizon
regression and classification labels together with the ``valid_at`` timestamp
for each horizon. Time semantics follow the convention that labels are
``made_at`` time *t* and become valid at *t + h*.

Example
-------
>>> import pandas as pd
>>> idx = pd.date_range("2024-01-01", periods=5, freq="1min")
>>> df = pd.DataFrame({"close": [100, 101, 102, 103, 104]}, index=idx)
>>> gen = LabelGenerator(horizons=[1, 2], epsilon_bp=8, candle_time_convention="end")
>>> ret_df = gen.make_returns(df)
>>> ret_df.filter(like="ret_")  # doctest:+ELLIPSIS
                     ret_h1   ret_h2
2024-01-01 00:00:00  0.0100  0.0200
2024-01-01 00:01:00  0.0099  0.0198
>>> cls_df = gen.make_classes(ret_df)
>>> cls_df.filter(like="cls_").iloc[0].to_dict()
{'cls_h1': 1, 'cls_h2': 1}
>>> gen.attach_valid_at_index(df, h=2)[0]
Timestamp('2024-01-01 00:02:00', freq='T')
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class LabelGenerator:
    """Generate aligned regression and classification labels.

    Parameters
    ----------
    horizons:
        Sequence of forecast horizons expressed in bars.
    epsilon_bp:
        Threshold in basis points to consider a return *neutral*.
    candle_time_convention:
        Either ``"start"`` or ``"end"``; controls where ``valid_at`` is placed.
    """

    horizons: List[int]
    epsilon_bp: float
    candle_time_convention: str = "end"

    def __post_init__(self) -> None:
        if not self.horizons:
            raise ValueError("At least one horizon is required")
        if any(h <= 0 for h in self.horizons):
            raise ValueError("Horizons must be positive integers")
        if self.candle_time_convention not in {"start", "end"}:
            raise ValueError("candle_time_convention must be 'start' or 'end'")

    def make_returns(self, df_ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with forward returns for each horizon.

        The returned frame includes the original OHLCV columns plus
        ``ret_h{h}`` columns representing ``close[t+h] / close[t] - 1``.
        """

        if "close" not in df_ohlcv.columns:
            raise KeyError("Input frame must contain a 'close' column")

        out = df_ohlcv.copy()
        close = out["close"].astype(float)
        for h in self.horizons:
            shifted = close.shift(-h)
            out[f"ret_h{h}"] = shifted / close - 1.0
        return out

    def make_classes(self, ret_df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with ternary classes per horizon.

        Each class column ``cls_h{h}`` takes values in ``{-1, 0, +1}`` based on
        the sign of the corresponding ``ret_h{h}`` and the ``epsilon_bp``
        tolerance. ``epsilon_bp`` is converted to a decimal return threshold.
        """

        epsilon = self.epsilon_bp / 1e4
        out = ret_df.copy()
        for h in self.horizons:
            col = f"ret_h{h}"
            if col not in out:
                raise KeyError(f"Missing {col} column; call make_returns first")
            values = out[col].to_numpy(copy=False)
            classes = np.zeros_like(values, dtype=int)
            classes[values > epsilon] = 1
            classes[values < -epsilon] = -1
            out[f"cls_h{h}"] = classes
        return out

    def validate_no_lookahead(self, ret_df: pd.DataFrame) -> None:
        """Ensure that forward returns only depend on future prices.

        The function asserts that ``ret_h{h}`` equals
        ``close.shift(-h) / close - 1`` for each horizon. Any deviation implies
        a look-ahead leak or data corruption.
        """

        close = ret_df["close"].astype(float)
        for h in self.horizons:
            expected = close.shift(-h) / close - 1.0
            actual = ret_df[f"ret_h{h}"]
            if not np.allclose(actual.fillna(np.nan), expected.fillna(np.nan), equal_nan=True):
                raise AssertionError(f"ret_h{h} does not match forward return definition")

    def attach_valid_at_index(self, df: pd.DataFrame, h: int) -> pd.Series:
        """Return the ``valid_at`` index for horizon ``h``.

        When ``candle_time_convention == 'end'`` the ``valid_at`` index is simply
        the input index shifted by +h steps. With ``'start'`` the timestamps are
        interpreted as bar starts and the valid time is shifted by ``h+1`` to
        capture the close.
        """

        if h not in self.horizons:
            raise ValueError(f"Horizon {h} not configured")
        shift = h if self.candle_time_convention == "end" else h + 1
        return pd.Series(df.index, index=df.index).shift(-shift)

    def generate(self, df_ohlcv: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Convenience wrapper returning returns, classes and valid indices."""

        ret_df = self.make_returns(df_ohlcv)
        self.validate_no_lookahead(ret_df)
        cls_df = self.make_classes(ret_df)
        valid_at = {
            h: self.attach_valid_at_index(df_ohlcv, h)
            for h in self.horizons
        }
        return {"returns": ret_df, "classes": cls_df, "valid_at": valid_at}
