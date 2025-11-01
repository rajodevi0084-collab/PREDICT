"""Consistency wrappers mimicking LightGBM style interfaces."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


class TabularLGBMRegressor:
    def __init__(self, **kwargs) -> None:
        self.model = GradientBoostingRegressor(**kwargs)
        self.feature_names_: Optional[list[str]] = None

    def fit(self, X, y) -> "TabularLGBMRegressor":
        self.feature_names_ = list(getattr(X, "columns", range(X.shape[1])))
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)


class TabularLGBMClassifier:
    def __init__(self, **kwargs) -> None:
        self.model = GradientBoostingClassifier(**kwargs)
        self.feature_names_: Optional[list[str]] = None

    def fit(self, X, y) -> "TabularLGBMClassifier":
        self.feature_names_ = list(getattr(X, "columns", range(X.shape[1])))
        self.model.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        proba = self.model.predict_proba(X)
        if proba.shape[1] == 2:
            # convert to 3-way by inserting flat class with zeros
            zeros = np.zeros((proba.shape[0], 1))
            proba = np.column_stack([proba[:, 0], zeros, proba[:, 1]])
        return proba


__all__ = ["TabularLGBMRegressor", "TabularLGBMClassifier"]
