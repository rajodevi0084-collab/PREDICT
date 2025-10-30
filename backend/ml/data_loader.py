"""Utilities for loading and normalising tabular datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path("data")

SUPPORTED_EXTENSIONS = {".csv", ".parquet", ".pq"}

_TIMESTAMP_CANDIDATES = ("timestamp", "datetime", "date", "time")
_SYMBOL_CANDIDATES = ("symbol", "symbols", "ticker", "asset")


@dataclass(frozen=True)
class DatasetMetadata:
    rows: int
    symbols: list[str]
    date_min: Optional[str]
    date_max: Optional[str]
    dtypes: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "symbols": self.symbols,
            "date_min": self.date_min,
            "date_max": self.date_max,
            "dtypes": self.dtypes,
        }


def list_symbols(file_ids: Sequence[str | Path]) -> list[str]:
    """Return the sorted list of unique ticker symbols across ``file_ids``."""

    frame = load_dataset(file_ids)
    if "symbol" not in frame.columns:
        return []
    symbols = frame["symbol"].dropna().astype(str).str.upper().str.strip().unique()
    return sorted(symbols.tolist())


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise timestamps, symbols, and ordering of the dataframe."""

    df = df.copy()

    timestamp_column = _find_timestamp_column(df)
    if timestamp_column is not None:
        df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors="coerce", utc=True)
        df = df.dropna(subset=[timestamp_column])
        if timestamp_column != "timestamp":
            df = df.rename(columns={timestamp_column: "timestamp"})
        timestamp_column = "timestamp"

    symbol_column = _find_symbol_column(df)
    if symbol_column is not None:
        df[symbol_column] = df[symbol_column].astype(str).str.upper().str.strip()
        if symbol_column != "symbol":
            df = df.rename(columns={symbol_column: "symbol"})
        symbol_column = "symbol"

    subset: Optional[list[str]] = None
    if timestamp_column and symbol_column:
        subset = [timestamp_column, symbol_column]
    elif timestamp_column:
        subset = [timestamp_column]

    if subset:
        df = df.drop_duplicates(subset=subset)
    else:
        df = df.drop_duplicates()

    if timestamp_column:
        sort_columns = [timestamp_column]
        if symbol_column:
            sort_columns.insert(0, symbol_column)
        df = df.sort_values(sort_columns, kind="mergesort")
    df = df.reset_index(drop=True)

    # Normalise price/volume columns to numeric floats for downstream consumers.
    for column in ("open", "high", "low", "close", "volume"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def load_dataset(
    file_ids: Sequence[str | Path] | str | Path,
    *,
    symbols: Optional[Iterable[str]] = None,
    start: Optional[pd.Timestamp | str] = None,
    end: Optional[pd.Timestamp | str] = None,
) -> pd.DataFrame:
    """Load one or more datasets, normalise them, and apply optional filters."""

    paths = _coerce_file_ids(file_ids)
    frames = [_load_single(path) for path in paths]
    if not frames:
        raise ValueError("No datasets could be loaded")

    frame = pd.concat(frames, ignore_index=True, sort=False)

    if symbols:
        normalized = {symbol.upper().strip() for symbol in symbols if symbol}
        if normalized:
            if "symbol" not in frame.columns:
                raise ValueError("Dataset does not contain a 'symbol' column")
            frame = frame[frame["symbol"].astype(str).str.upper().isin(sorted(normalized))]

    if start is not None or end is not None:
        frame = slice_time(frame, start=start, end=end)

    if {"symbol", "timestamp"}.issubset(frame.columns):
        frame = frame.sort_values(["symbol", "timestamp"], kind="mergesort")
    elif "timestamp" in frame.columns:
        frame = frame.sort_values("timestamp")

    return frame.reset_index(drop=True)


def slice_time(
    df: pd.DataFrame,
    *,
    start: Optional[pd.Timestamp | str] = None,
    end: Optional[pd.Timestamp | str] = None,
) -> pd.DataFrame:
    """Return a time-sliced view of ``df`` using the ``timestamp`` column."""

    if "timestamp" not in df.columns:
        return df.copy()

    ts = df["timestamp"]
    if not pd.api.types.is_datetime64tz_dtype(ts):
        ts = pd.to_datetime(ts, errors="coerce", utc=True)

    mask = pd.Series(True, index=df.index, dtype=bool)
    if start is not None:
        start_ts = _ensure_timestamp(start)
        mask &= ts >= start_ts
    if end is not None:
        end_ts = _ensure_timestamp(end)
        mask &= ts <= end_ts

    return df.loc[mask].copy()


def _ensure_timestamp(value: pd.Timestamp | str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _pyarrow_type_mapper(arrow_type: pa.DataType) -> Any:
    """Map pyarrow types to pandas extension types where appropriate."""

    if pa.types.is_timestamp(arrow_type):
        return pd.ArrowDtype(arrow_type)
    return None


def _build_metadata(df: pd.DataFrame, schema: pa.Schema, num_rows: int) -> DatasetMetadata:
    timestamp_column = "timestamp" if "timestamp" in df.columns else None

    date_min: Optional[str] = None
    date_max: Optional[str] = None
    if timestamp_column:
        if not pd.api.types.is_datetime64tz_dtype(df[timestamp_column]):
            timestamps = pd.to_datetime(df[timestamp_column], errors="coerce", utc=True)
        else:
            timestamps = df[timestamp_column]
        if not timestamps.empty:
            date_min = timestamps.min().isoformat()
            date_max = timestamps.max().isoformat()

    symbol_column = _find_symbol_column(df)
    if symbol_column is not None:
        symbols_series = df[symbol_column].dropna().astype(str).str.upper().str.strip()
        symbols = sorted(symbols_series.unique().tolist())
    else:
        symbols = []

    dtypes = {}
    for field in schema:
        dtypes[field.name] = str(field.type)
    for column, dtype in df.dtypes.items():
        dtypes.setdefault(column, str(dtype))

    return DatasetMetadata(
        rows=num_rows,
        symbols=symbols,
        date_min=date_min,
        date_max=date_max,
        dtypes=dtypes,
    )


def load_with_metadata(path: Path) -> tuple[pd.DataFrame, DatasetMetadata]:
    """Load a single dataset and return it alongside metadata."""

    df = _load_single(path)
    schema = pa.Table.from_pandas(df, preserve_index=False).schema
    metadata = _build_metadata(df, schema, len(df))
    return df, metadata


def _find_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    lower_names = {name.lower(): name for name in df.columns}
    for candidate in _TIMESTAMP_CANDIDATES:
        if candidate in lower_names:
            return lower_names[candidate]
    for column in df.columns:
        if "time" in column.lower() or "date" in column.lower():
            return column
    return None


def _find_symbol_column(df: pd.DataFrame) -> Optional[str]:
    lower_names = {name.lower(): name for name in df.columns}
    for candidate in _SYMBOL_CANDIDATES:
        if candidate in lower_names:
            return lower_names[candidate]
    for column in df.columns:
        if "symbol" in column.lower() or "ticker" in column.lower():
            return column
    return None


def _load_single(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {suffix}")

    if suffix == ".csv":
        df = pd.read_csv(path)
    else:
        parquet_file = pq.ParquetFile(path)
        table = parquet_file.read()
        df = table.to_pandas(types_mapper=_pyarrow_type_mapper)

    return _normalize_df(df)


def _coerce_file_ids(file_ids: Sequence[str | Path] | str | Path) -> list[Path]:
    if isinstance(file_ids, (str, Path)):
        items: Sequence[str | Path] = [file_ids]
    else:
        items = file_ids

    paths: list[Path] = []
    for item in items:
        path = _resolve_file(item)
        paths.append(path)
    return paths


def _resolve_file(item: str | Path) -> Path:
    path = Path(item)
    if path.exists():
        return path
    # Treat bare identifiers as entries in the data directory.
    if not path.suffix:
        for suffix in sorted(SUPPORTED_EXTENSIONS):
            candidate = DATA_DIR / f"{path}{suffix}"
            if candidate.exists():
                return candidate
    matches = list(DATA_DIR.glob(f"{path}.*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Dataset '{item}' not found")


__all__ = [
    "DATA_DIR",
    "SUPPORTED_EXTENSIONS",
    "DatasetMetadata",
    "list_symbols",
    "load_dataset",
    "load_with_metadata",
    "slice_time",
]
