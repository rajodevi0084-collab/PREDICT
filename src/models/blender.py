"""Model ensembling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class RidgeBlender:
    alpha: float = 1.0

    def __post_init__(self) -> None:
        self.coef_: np.ndarray | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> "RidgeBlender":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        gram = X.T @ X + self.alpha * np.eye(X.shape[1])
        self.coef_ = np.linalg.solve(gram, X.T @ y)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Blender has not been fitted")
        X = np.asarray(X, dtype=float)
        return X @ self.coef_


__all__ = ["RidgeBlender"]
