"""Consistency wrappers mimicking LightGBM style interfaces."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


class TabularLGBMRegressor:
    def __init__(self, **kwargs) -> None:
        params = dict(kwargs)
        if params.get("objective") == "huber":
            params.setdefault("loss", "huber")
            params.pop("objective", None)
        params.setdefault("n_iter_no_change", 20)
        params.setdefault("validation_fraction", 0.1)
        self.model = GradientBoostingRegressor(**params)
        self.feature_names_: Optional[list[str]] = None

    def fit(self, X, y) -> "TabularLGBMRegressor":
        self.feature_names_ = list(getattr(X, "columns", range(X.shape[1])))
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)


class TabularLGBMClassifier:
    def __init__(self, **kwargs) -> None:
        params = dict(kwargs)
        self.objective = params.pop("objective", "multiclass")
        if self.objective not in {"multiclass", "multiclass_3way"}:
            raise ValueError(f"Unsupported objective '{self.objective}'")
        params.setdefault("n_iter_no_change", 20)
        params.setdefault("validation_fraction", 0.1)
        self.model = GradientBoostingClassifier(**params)
        self.feature_names_: Optional[list[str]] = None
        self.class_order_ = np.array([-1, 0, 1])

    def fit(self, X, y) -> "TabularLGBMClassifier":
        self.feature_names_ = list(getattr(X, "columns", range(X.shape[1])))
        y_arr = np.asarray(y)
        if self.objective == "multiclass_3way":
            mapping = {cls: idx for idx, cls in enumerate(self.class_order_)}
            if not np.isin(np.unique(y_arr), self.class_order_).all():
                raise ValueError("Labels must be subset of {-1,0,1} for multiclass_3way")
            y_encoded = np.vectorize(mapping.get)(y_arr)
            self.model.fit(X, y_encoded)
        else:
            self.model.fit(X, y_arr)
        return self

    def predict_proba(self, X) -> np.ndarray:
        proba = self.model.predict_proba(X)
        if self.objective == "multiclass_3way":
            if proba.shape[1] != len(self.class_order_):
                raise RuntimeError("Classifier did not produce 3-way probabilities")
            return proba
        if proba.shape[1] == 2:
            zeros = np.zeros((proba.shape[0], 1))
            proba = np.column_stack([proba[:, 0], zeros, proba[:, 1]])
        return proba

    def predict(self, X) -> np.ndarray:
        if self.objective == "multiclass_3way":
            encoded = self.model.predict(X)
            return self.class_order_[encoded]
        return self.model.predict(X)


__all__ = ["TabularLGBMRegressor", "TabularLGBMClassifier"]
