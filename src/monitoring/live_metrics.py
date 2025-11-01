"""Runtime monitoring helpers for the next-tick service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class LiveMetrics:
    latency_ms: float
    psi_features: float
    psi_labels: float
    conformal_coverage: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "latency_ms": self.latency_ms,
            "psi_features": self.psi_features,
            "psi_labels": self.psi_labels,
            "conformal_coverage": self.conformal_coverage,
        }


def compute_psi(expected: np.ndarray, observed: np.ndarray) -> float:
    expected = np.asarray(expected, dtype=float)
    observed = np.asarray(observed, dtype=float)
    hist_expected, bins = np.histogram(expected, bins=10, density=True)
    hist_observed, _ = np.histogram(observed, bins=bins, density=True)
    hist_expected = np.clip(hist_expected, 1e-8, None)
    hist_observed = np.clip(hist_observed, 1e-8, None)
    return float(((hist_observed - hist_expected) * np.log(hist_observed / hist_expected)).sum())


__all__ = ["LiveMetrics", "compute_psi"]
