"""Inference helper for the next-bar OHLCV student model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from src.calibration.calibrate_cls import TemperatureScaler
from src.calibration.calibrate_reg import LinearCalibration
from src.conformal.conformal_reg import ConformalInterval, apply_conformal
from src.postproc.reconstruct_price import reconstruct_close


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum()


@dataclass
class NextBarService:
    coverage: float = 0.9

    def __post_init__(self) -> None:
        self.cls_calibrator = TemperatureScaler()
        self.reg_calibrator = LinearCalibration()
        self.conformal = ConformalInterval(alpha=1 - self.coverage, qhat=0.002)

    def predict(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        logits = np.asarray(snapshot.get("logits", [0.0, 0.0, 0.0]), dtype=float)
        logits = self.cls_calibrator.transform(logits)
        probs = _softmax(logits)

        y_reg_hat = float(snapshot.get("y_reg_hat", 0.0))
        y_reg_calibrated = self.reg_calibrator.transform(np.array([y_reg_hat]))[0]

        close_price = float(snapshot.get("close", snapshot.get("price", 0.0)))
        next_close = reconstruct_close(close_price, y_reg_calibrated)

        conformal_interval = apply_conformal(np.array([y_reg_calibrated]), self.conformal)[0]
        lower_price = reconstruct_close(close_price, conformal_interval[0])
        upper_price = reconstruct_close(close_price, conformal_interval[1])

        return {
            "p_down": float(probs[0]),
            "p_flat": float(probs[1]),
            "p_up": float(probs[2]),
            "y_reg_hat": float(y_reg_calibrated),
            "next_close_hat": float(next_close),
            "bands": [float(lower_price), float(next_close), float(upper_price)],
        }


def predict(snapshot: Dict[str, Any], coverage: float = 0.9) -> Dict[str, Any]:
    service = NextBarService(coverage=coverage)
    return service.predict(snapshot)


__all__ = ["NextBarService", "predict"]
