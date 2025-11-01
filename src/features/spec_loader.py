"""Feature specification loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from importlib import import_module
from types import ModuleType
from typing import Any, Dict, Iterable, Iterator

import yaml


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    raw: Dict[str, Any]

    def group(self, key: str, default: Any | None = None) -> Any:
        groups = self.raw.get("groups", {})
        if key not in groups:
            if default is not None:
                return default
            raise KeyError(f"Feature group '{key}' not found in spec '{self.name}'")
        return groups[key]

    def hygiene(self) -> Dict[str, Any]:
        return self.group("hygiene", default={})

    def keep_top_k(self) -> int | None:
        return self.raw.get("keep_top_k")


def load_feature_spec(name: str, *, base_dir: str | Path = "configs/features") -> FeatureSpec:
    path = Path(base_dir) / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Feature spec '{name}' not found at {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return FeatureSpec(name=name, raw=data)


def iter_feature_groups(spec: FeatureSpec) -> Iterator[tuple[str, Any]]:
    groups = spec.raw.get("groups", {})
    for key, value in groups.items():
        if key == "hygiene":
            continue
        yield key, value


_FEATURE_BUILDERS = {
    "next_tick_v1": "src.features.build_next_tick_v1",
    "next_bar_ohlcv_v2": "src.features.build_next_bar_ohlcv_v2",
}


def resolve_feature_builder(name: str) -> ModuleType:
    """Return the module implementing the requested feature builder."""

    if name not in _FEATURE_BUILDERS:
        raise KeyError(f"No feature builder registered for spec '{name}'")
    module_path = _FEATURE_BUILDERS[name]
    return import_module(module_path)


__all__ = [
    "FeatureSpec",
    "load_feature_spec",
    "iter_feature_groups",
    "resolve_feature_builder",
]
