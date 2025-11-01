"""Simple ridge blender for stacking model predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


@dataclass
class BlenderResult:
    feature_names: list[str]
    component_weights: np.ndarray


class RidgeBlender:
    def __init__(self, alpha: float = 1.0, weight_metric: str = "mae_inverse") -> None:
        self.alpha = alpha
        self.weight_metric = weight_metric
        self.model = Ridge(alpha=self.alpha)
        self.feature_names_: list[str] | None = None
        self.component_weights_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: Iterable[float]) -> "RidgeBlender":
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.to_numpy(dtype=float, copy=False)
        else:
            X_arr = np.asarray(X, dtype=float)
            self.feature_names_ = list(range(X_arr.shape[1]))
        y_arr = np.asarray(list(y), dtype=float)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("Mismatched feature rows and target length")

        self.model.fit(X_arr, y_arr)
        self.component_weights_ = self._compute_component_weights(X_arr, y_arr)
        return self

    def _compute_component_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.weight_metric == "mae_inverse":
            errors = np.mean(np.abs(X - y[:, None]), axis=0)
            weights = 1.0 / np.maximum(errors, 1e-6)
        else:
            weights = np.ones(X.shape[1])
        weights = weights / np.sum(weights)
        return weights

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy(dtype=float, copy=False)
        else:
            X_arr = np.asarray(X, dtype=float)
        return self.model.predict(X_arr)

    def describe(self) -> BlenderResult:
        if self.feature_names_ is None or self.component_weights_ is None:
            raise RuntimeError("Blender has not been fitted")
        return BlenderResult(feature_names=list(self.feature_names_), component_weights=self.component_weights_)


__all__ = ["RidgeBlender", "BlenderResult"]
