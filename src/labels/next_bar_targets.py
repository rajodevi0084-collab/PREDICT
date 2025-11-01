"""Label generation for next-bar OHLCV prediction with strict H=1 shift."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_next_bar_targets(
    df_ohlcv: pd.DataFrame,
    *,
    dead_zone_abs_bp: float,
    horizon_bars: int = 1,
) -> tuple[pd.Series, pd.Series]:
    """Return log-return regression and classification targets for H=1."""

    if "close" not in df_ohlcv.columns:
        raise KeyError("'close' column is required to build next-bar targets")

    if horizon_bars != 1:
        raise AssertionError("This run must be H=1")

    close = df_ohlcv["close"].astype(float)
    if close.isna().any():
        raise ValueError("Close column contains NaNs; cannot build targets")

    y_reg_full = np.log(close.shift(-horizon_bars) / close)

    threshold = abs(dead_zone_abs_bp) / 1e4
    y_cls_full = np.sign(y_reg_full)
    if threshold > 0:
        neutral_mask = y_reg_full.abs() < threshold
        y_cls_full = y_cls_full.mask(neutral_mask, 0.0)

    y_reg = y_reg_full.dropna()
    y_cls = y_cls_full.loc[y_reg.index].fillna(0.0).astype(int)

    if not y_reg.index.equals(y_cls.index):
        raise ValueError("Classification targets misaligned with regression targets")

    return y_reg, y_cls


__all__ = ["build_next_bar_targets"]
