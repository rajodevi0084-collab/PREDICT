"""Inference helper for the next-bar OHLCV student model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

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

        y_reg_hat_raw = float(snapshot.get("y_reg_hat", 0.0))
        y_reg_calibrated = self.reg_calibrator.transform(np.array([y_reg_hat_raw]))[0]

        close_price = float(snapshot.get("close", snapshot.get("price", 0.0)))
        next_close = reconstruct_close(close_price, y_reg_calibrated)

        conformal_interval = apply_conformal(np.array([y_reg_calibrated]), self.conformal)[0]
        lower_price = reconstruct_close(close_price, conformal_interval[0])
        upper_price = reconstruct_close(close_price, conformal_interval[1])

        obs_time = _normalise_timestamp(
            snapshot.get("obs_time") or snapshot.get("timestamp")
        )
        target_time = _normalise_timestamp(
            snapshot.get("target_time") or snapshot.get("next_timestamp")
        )

        calibration_map = self.reg_calibrator.model.as_mapping()

        return {
            "symbol": snapshot.get("symbol"),
            "obs_time": obs_time,
            "target_time": target_time,
            "c_t": close_price,
            "p_down": float(probs[0]),
            "p_flat": float(probs[1]),
            "p_up": float(probs[2]),
            "y_reg_hat": float(y_reg_calibrated),
            "y_reg_hat_raw": y_reg_hat_raw,
            "next_close_hat": float(next_close),
            "bands": {
                "lo": float(lower_price),
                "med": float(next_close),
                "hi": float(upper_price),
            },
            "calibration": calibration_map,
        }


def predict(snapshot: Dict[str, Any], coverage: float = 0.9) -> Dict[str, Any]:
    service = NextBarService(coverage=coverage)
    return service.predict(snapshot)


def _normalise_timestamp(value: Optional[Any]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.astimezone().isoformat()
    if isinstance(value, np.datetime64):
        return str(value)
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        return iso()
    return str(value)


__all__ = ["NextBarService", "predict"]
