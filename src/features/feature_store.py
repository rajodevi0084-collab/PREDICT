"""Simple on-disk feature store with namespace support."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

_BASE_DIR = Path("data/feature_store")


@dataclass
class FeatureHandle:
    namespace: str
    symbol: str
    version: str

    @property
    def path(self) -> Path:
        return _BASE_DIR / self.namespace / self.symbol / self.version


def _meta_path(handle: FeatureHandle) -> Path:
    return handle.path / "metadata.json"


def store_features(handle: FeatureHandle, features: pd.DataFrame, *, overwrite: bool = False) -> None:
    path = handle.path
    path.mkdir(parents=True, exist_ok=True)

    feature_path = path / "features.parquet"
    if feature_path.exists() and not overwrite:
        raise FileExistsError(f"Features already exist for {handle}")

    features.to_parquet(feature_path)

    metadata = {"rows": int(features.shape[0]), "cols": int(features.shape[1])}
    with _meta_path(handle).open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


def load_features(handle: FeatureHandle) -> Optional[pd.DataFrame]:
    feature_path = handle.path / "features.parquet"
    if not feature_path.exists():
        return None
    return pd.read_parquet(feature_path)


def has_features(handle: FeatureHandle) -> bool:
    return (handle.path / "features.parquet").exists()


__all__ = ["FeatureHandle", "store_features", "load_features", "has_features"]
