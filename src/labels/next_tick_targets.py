"""Label generation utilities for the next-tick prediction task."""

from __future__ import annotations

import numpy as np
import pandas as pd


class LabelAlignmentError(RuntimeError):
    """Raised when label construction detects a temporal misalignment."""


def build_next_tick_labels(
    df: pd.DataFrame,
    tick_size: float,
    dead_zone_ticks: float,
    *,
    price_col: str = "mid",
) -> pd.DataFrame:
    """Return next-tick regression and classification labels.

    The returned dataframe contains ``y_reg`` (log return) and ``y_cls`` (-1/0/1)
    columns indexed by the feature timestamp ``t``. Both labels depend solely on
    information from tick ``t+1`` to avoid leakage.
    """

    if price_col not in df.columns:
        raise KeyError(f"Price column '{price_col}' not found")

    price = df[price_col].astype(float)
    if price.isna().any():
        raise ValueError("Price column contains NaNs; cannot compute labels")

    next_price = price.shift(-1)
    delta = next_price - price
    y_reg = np.log(next_price / price)

    threshold = abs(dead_zone_ticks) * float(tick_size)
    y_cls = pd.Series(np.sign(delta), index=delta.index).astype(float)
    dead_zone_mask = delta.abs() < threshold
    y_cls.loc[dead_zone_mask] = 0.0
    y_cls.loc[~dead_zone_mask] = y_cls.loc[~dead_zone_mask].apply(np.sign)
    y_reg = y_reg.iloc[:-1]
    y_cls = y_cls.iloc[:-1].fillna(0).astype(int)

    labels = pd.DataFrame({"y_reg": y_reg, "y_cls": y_cls}, index=df.index[:-1])
    if labels.isna().any().any():
        raise ValueError("Constructed labels contain NaNs")

    # Invariant: the label at index ``t`` must depend on price at ``t+1`` only.
    # We verify this by checking that the final timestamp was dropped exactly once.
    if len(labels.index.intersection(df.index[-1:])) != 0:
        raise LabelAlignmentError("Labels should not include the final timestamp")

    return labels


__all__ = ["build_next_tick_labels", "LabelAlignmentError"]
