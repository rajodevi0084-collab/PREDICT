"""Inference helper for the next-bar OHLCV student model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

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
    horizon: int = 1

    def __post_init__(self) -> None:
        self.cls_calibrator = TemperatureScaler()
        self.reg_calibrator = LinearCalibration()
        self.conformal = ConformalInterval(alpha=1 - self.coverage, qhat=0.002)

    def predict(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        if self.horizon != 1:
            raise AssertionError("This run must be H=1")

        bars = _bars_to_frame(snapshot.get("bars"))
        if len(bars) <= self.horizon:
            raise ValueError("Not enough bars to compute next-bar prediction")

        obs_time = bars.index[-1]
        target_time = bars.index[-1 + self.horizon]

        if pd.Timestamp(target_time) <= pd.Timestamp(obs_time):
            raise AssertionError("target_time must be strictly greater than obs_time")

        obs_ts = _normalise_timestamp(obs_time)
        target_ts = _normalise_timestamp(target_time)

        close_price = float(bars["close"].iloc[-1])

        logits = np.asarray(snapshot.get("logits", [0.0, 0.0, 0.0]), dtype=float)
        logits = self.cls_calibrator.transform(logits)
        probs = _softmax(logits)

        y_reg_hat_raw = float(snapshot.get("y_reg_hat", 0.0))
        y_reg_calibrated = self.reg_calibrator.transform(np.array([y_reg_hat_raw]))[0]

        next_close = reconstruct_close(close_price, y_reg_calibrated)

        conformal_interval = apply_conformal(np.array([y_reg_calibrated]), self.conformal)[0]
        lower_price = reconstruct_close(close_price, conformal_interval[0])
        upper_price = reconstruct_close(close_price, conformal_interval[1])

        return {
            "symbol": snapshot.get("symbol"),
            "horizon": self.horizon,
            "obs_time": obs_ts,
            "target_time": target_ts,
            "c_t": close_price,
            "p_down": float(probs[0]),
            "p_flat": float(probs[1]),
            "p_up": float(probs[2]),
            "y_reg_hat": float(y_reg_calibrated),
            "next_close_hat": float(next_close),
            "bands": {
                "lo": float(lower_price),
                "med": float(next_close),
                "hi": float(upper_price),
            },
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
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is None:
            return value.tz_localize("UTC").isoformat()
        return value.tz_convert("UTC").isoformat()
    if isinstance(value, np.datetime64):
        return str(value)
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        return iso()
    return str(value)


def _bars_to_frame(bars: Optional[Any]) -> pd.DataFrame:
    if bars is None:
        raise KeyError("bars payload is required")

    if isinstance(bars, pd.DataFrame):
        frame = bars.copy()
    elif isinstance(bars, Sequence):
        frame = pd.DataFrame(list(bars))
    else:
        raise TypeError("bars payload must be a DataFrame or sequence of records")

    if frame.empty:
        raise ValueError("bars payload is empty")

    if "timestamp" in frame.columns:
        index = pd.to_datetime(frame.pop("timestamp"), utc=True)
        frame.index = index
    elif not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index, utc=True)

    if "close" not in frame.columns:
        raise KeyError("bars payload must include a 'close' column")

    return frame


__all__ = ["NextBarService", "predict"]
