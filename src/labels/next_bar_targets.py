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

    next_price = price.shift(-1)
    y_reg = np.log(next_price / price)
    threshold = abs(dead_zone_abs_bp) / 1e4

    y_cls = np.sign(y_reg).astype(float)
    dead_zone_mask = y_reg.abs() < threshold
    y_cls[dead_zone_mask] = 0.0
    y_cls[~dead_zone_mask] = np.sign(y_reg[~dead_zone_mask])

    # Drop the final bar where the shift introduced NaNs.
    y_reg = y_reg.iloc[:-1]
    y_cls = y_cls.iloc[:-1].fillna(0).astype(int)

    labels = pd.DataFrame({"y_reg": y_reg, "y_cls": y_cls}, index=df.index[:-1])

    if labels.isna().any().any():
        raise ValueError("Constructed next-bar targets contain NaNs")

    # Ensure the final timestamp from the original data is not present in targets.
    if len(labels.index.intersection(df.index[-1:])) != 0:
        raise NextBarAlignmentError("Next-bar targets should not include the final bar")

    return labels


__all__ = ["build_next_bar_targets", "NextBarAlignmentError"]
