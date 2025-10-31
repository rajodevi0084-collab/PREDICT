"""Prediction helpers that keep track of timing metadata."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Mapping, Optional

import numpy as np
import pandas as pd


@dataclass
class PredictionBundle:
    """Container describing a multi-horizon prediction made at time ``t``."""

    made_at: pd.Timestamp
    yhat_reg: Dict[int, float]
    probs_cls: Dict[int, Dict[str, float]]
    valid_at: Dict[int, pd.Timestamp]

    def __post_init__(self) -> None:
        for h, ts in self.valid_at.items():
            if ts <= self.made_at:
                raise ValueError(f"valid_at for horizon {h} must be strictly after made_at")


class PredictionStore:
    """Interface for storing and retrieving :class:`PredictionBundle` objects."""

    def append(self, bundle: PredictionBundle) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def iter(
        self,
        symbol: Optional[str] = None,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> Iterator[PredictionBundle]:  # pragma: no cover - interface
        raise NotImplementedError


class InMemoryPredictionStore(PredictionStore):
    """Simple in-memory store primarily used for testing and prototyping."""

    def __init__(self) -> None:
        self._items: list[PredictionBundle] = []

    def append(self, bundle: PredictionBundle) -> None:
        self._items.append(bundle)

    def iter(
        self,
        symbol: Optional[str] = None,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> Iterator[PredictionBundle]:
        for bundle in self._items:
            if start and bundle.made_at < start:
                continue
            if end and bundle.made_at > end:
                continue
            yield bundle


@dataclass
class FeatureSnapshot:
    """Minimal snapshot passed into :func:`predict_multi_horizon`.

    The snapshot carries the trained model callable, the feature vector and
    contextual information such as horizons and a calendar utility. The callable
    must accept the feature vector and return a mapping with ``reg`` and ``cls``
    entries.
    """

    model: object
    features: Mapping[str, float]
    horizons: Iterable[int]
    made_at: pd.Timestamp
    calendar: Mapping[int, pd.Timestamp]


def _softmax(logits: Iterable[float]) -> list[float]:
    values = np.asarray(list(logits), dtype=float)
    if values.size == 0:
        raise ValueError("Softmax requires at least one value")
    values = values - values.max()
    exp = np.exp(values)
    exp_sum = exp.sum()
    if exp_sum == 0:
        raise ValueError("Softmax underflow; all exponentials zero")
    return (exp / exp_sum).tolist()


def predict_multi_horizon(snapshot: FeatureSnapshot) -> PredictionBundle:
    """Run the forecaster and return a :class:`PredictionBundle`.

    The model callable is expected to return a mapping with ``reg`` and ``cls``
    outputs keyed by horizon. ``reg`` should already be in return space while
    ``cls`` can be either probabilities or logits. ``made_at`` is propagated and
    ``valid_at`` is taken from the provided calendar mapping.
    """

    raw = snapshot.model(snapshot.features)
    yhat_reg = {int(h): float(raw["reg"][h]) for h in snapshot.horizons}

    probs_cls: Dict[int, Dict[str, float]] = {}
    for h in snapshot.horizons:
        cls_values = raw["cls"][h]
        if isinstance(cls_values, Mapping):
            probs = {str(k): float(v) for k, v in cls_values.items()}
        else:
            sm = _softmax(cls_values)
            probs = {k: v for k, v in zip(["down", "flat", "up"], sm)}
        probs_cls[int(h)] = probs

    valid_at = {int(h): pd.Timestamp(snapshot.calendar[h]) for h in snapshot.horizons}

    return PredictionBundle(
        made_at=pd.Timestamp(snapshot.made_at),
        yhat_reg=yhat_reg,
        probs_cls=probs_cls,
        valid_at=valid_at,
    )


def store_predictions(bundle: PredictionBundle, store: PredictionStore) -> None:
    """Persist ``bundle`` to ``store`` ensuring timestamps are recorded."""

    store.append(bundle)
