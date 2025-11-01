"""Label generation for next-bar OHLCV prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd


class NextBarAlignmentError(RuntimeError):
    """Raised when next-bar target alignment invariants are violated."""


def build_next_bar_targets(
    df: pd.DataFrame,
    *,
    price_col: str = "close",
    dead_zone_abs_bp: float = 0.0,
) -> pd.DataFrame:
    """Return regression and classification targets for the next bar.

    Parameters
    ----------
    df:
        Input dataframe indexed by bar timestamp and containing the ``price_col``.
    price_col:
        Column to use for price, defaults to ``"close"``.
    dead_zone_abs_bp:
        Absolute dead-zone expressed in basis points. Classification labels within
        ``±dead_zone_abs_bp`` are mapped to the neutral class ``0``.
    """

    if price_col not in df.columns:
        raise KeyError(f"Price column '{price_col}' not found in dataframe")

    price = df[price_col].astype(float)
    if price.isna().any():
        raise ValueError("Price column contains NaNs; cannot compute next-bar targets")

    y_reg = np.log(price.shift(-1) / price)
    threshold = abs(dead_zone_abs_bp) / 1e4

    y_cls = pd.Series(np.sign(y_reg), index=y_reg.index)
    if threshold > 0:
        neutral = y_reg.abs() < threshold
        y_cls.loc[neutral] = 0

    y_reg = y_reg.dropna()
    y_cls = y_cls.loc[y_reg.index].fillna(0).astype(int)

    labels = pd.DataFrame({"y_reg": y_reg, "y_cls": y_cls}, index=y_reg.index)

    if labels.isna().any().any():
        raise ValueError("Constructed next-bar targets contain NaNs")

    if labels.index.max() == df.index.max():
        raise NextBarAlignmentError("Next-bar targets should not include the final bar")

    return labels


__all__ = ["build_next_bar_targets", "NextBarAlignmentError"]
