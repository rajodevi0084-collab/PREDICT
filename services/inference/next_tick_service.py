"""Inference helpers for the next-tick endpoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np

from src.calibration.calibrate_cls import TemperatureScaler
from src.calibration.calibrate_reg import LinearCalibration
from src.conformal.conformal_reg import ConformalRegressor
from src.postproc.reconstruct_price import reconstruct
from src.postproc.tick_snap import clip_log_return


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max()
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum()


@dataclass
class NextTickService:
    tick_size: float
    sigma_train: float = 0.01

    def __post_init__(self) -> None:
        self.cls_calibrator = TemperatureScaler()
        self.reg_calibrator = LinearCalibration()
        self.conformal = ConformalRegressor(coverage=0.9)
        self.conformal.residuals_ = np.array([self.sigma_train])

    def predict(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        logits = np.asarray(snapshot.get("logits", [0.0, 0.0, 0.0]), dtype=float)
        logits = self.cls_calibrator.transform(logits)
        probs = _softmax(logits)

        y_reg_hat = float(snapshot.get("y_reg_hat", 0.0))
        y_reg_calibrated = self.reg_calibrator.transform(np.array([y_reg_hat]))[0]
        y_reg_clipped = clip_log_return(np.array([y_reg_calibrated]), 3.0, self.sigma_train)[0]

        mid = float(snapshot["mid"])
        next_price = reconstruct(mid, y_reg_clipped, self.tick_size)
        lower, upper = self.conformal.interval(np.array([y_reg_clipped]))
        bands = reconstruct(mid, np.array([lower[0], y_reg_clipped, upper[0]]), self.tick_size)

        return {
            "p_down": float(probs[0]),
            "p_flat": float(probs[1]),
            "p_up": float(probs[2]),
            "y_reg_hat": float(y_reg_clipped),
            "m_next_hat": float(next_price),
            "bands": bands.tolist() if hasattr(bands, "tolist") else [float(b) for b in bands],
        }


def predict(features_snapshot: Dict[str, Any], *, tick_size: float = 0.05) -> Dict[str, Any]:
    service = NextTickService(tick_size=tick_size)
    return service.predict(features_snapshot)


__all__ = ["NextTickService", "predict"]
