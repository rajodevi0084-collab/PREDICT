"""Utilities for retrieving price series used during labelling and inference."""

from __future__ import annotations

from typing import Literal

import pandas as pd

PriceSource = Literal["mid", "last", "close", "typical"]


def get_price_series(df: pd.DataFrame, source: PriceSource = "mid") -> pd.Series:
    """Return the price series specified by ``source``.

    Parameters
    ----------
    df:
        Input dataframe containing at least ``bid``/``ask`` or ``last`` columns.
    source:
        Either ``"mid"`` to compute ``(bid + ask) / 2`` or ``"last"`` to use the
        last traded price directly.

    Returns
    -------
    pandas.Series
        Price series aligned with ``df``'s index.

    Raises
    ------
    KeyError
        If the requested source cannot be constructed from ``df``.
    ValueError
        If the resulting series contains missing values.
    """

    source = cast_price_source(source)
    if source == "mid":
        if {"bid", "ask"}.issubset(df.columns):
            price = (df["bid"].astype(float) + df["ask"].astype(float)) / 2.0
        elif {"best_bid", "best_ask"}.issubset(df.columns):
            price = (df["best_bid"].astype(float) + df["best_ask"].astype(float)) / 2.0
        else:
            if "last" not in df.columns:
                raise KeyError("Cannot compute mid price: missing bid/ask columns")
            price = df["last"].astype(float)
    elif source == "close":
        if "close" not in df.columns:
            raise KeyError("Cannot retrieve close price: column 'close' missing")
        price = df["close"].astype(float)
    elif source == "typical":
        price = typical_price(df)
    else:
        if "last" not in df.columns:
            raise KeyError("Cannot retrieve last price: column 'last' missing")
        price = df["last"].astype(float)

    if price.isna().any():
        raise ValueError("Price series contains NaNs; ensure input data is cleaned")

    price = price.rename("price")
    return price


def cast_price_source(source: str) -> PriceSource:
    """Validate and normalise a price source string."""

    normalized = source.strip().lower()
    if normalized not in {"mid", "last", "close", "typical"}:
        raise ValueError(f"Unsupported price source '{source}'")
    return normalized  # type: ignore[return-value]


def typical_price(df: pd.DataFrame) -> pd.Series:
    """Return the typical price ``(high + low + close) / 3`` as a float series."""

    required = {"high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(f"Cannot compute typical price: missing columns {missing_str}")
    price = (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3.0
    price = price.rename("price")
    if price.isna().any():
        raise ValueError("Typical price contains NaNs; ensure input data is cleaned")
    return price


__all__ = ["get_price_series", "cast_price_source", "typical_price", "PriceSource"]
