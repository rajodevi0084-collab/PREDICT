"""Policy gating based on calibration confidence."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from backend.ml.predict import PredictionBundle


@dataclass
class GateDecision:
    made_at: str
    valid_at: str
    horizon: int
    original_action: str
    gated_action: str
    reason: str


def gate_action(
    pred_bundle: PredictionBundle,
    raw_action: Dict[int, str],
    coverage_tau: float,
    z_tau: Optional[float] = None,
) -> Dict[int, str]:
    """Return gated action dictionary keyed by horizon."""

    result: Dict[int, str] = {}
    for horizon, action in raw_action.items():
        probs = pred_bundle.probs_cls.get(horizon, {})
        confidence = max(probs.values()) if probs else 0.0
        gated = action
        reason = ""
        if confidence < coverage_tau:
            gated = "Flat"
            reason = f"confidence {confidence:.3f} below tau {coverage_tau:.3f}"
        elif z_tau is not None:
            ret = abs(pred_bundle.yhat_reg.get(horizon, 0.0))
            if ret < z_tau:
                gated = "Flat"
                reason = f"|ret| {ret:.4f} below z_tau {z_tau:.4f}"
        if gated != action:
            reason = reason or "forced flat"
        result[horizon] = gated
    return result


@dataclass
class PolicyGate:
    coverage_tau: float
    z_tau: Optional[float] = None
    decisions: List[GateDecision] = field(default_factory=list)

    def gate(self, pred_bundle: PredictionBundle, raw_action: Dict[int, str]) -> Dict[int, str]:
        gated = gate_action(pred_bundle, raw_action, self.coverage_tau, self.z_tau)
        for horizon, action in raw_action.items():
            new_action = gated[horizon]
            if new_action != action:
                reason = "confidence"
                probs = pred_bundle.probs_cls.get(horizon, {})
                confidence = max(probs.values()) if probs else 0.0
                if self.z_tau is not None:
                    ret = abs(pred_bundle.yhat_reg.get(horizon, 0.0))
                    if ret < self.z_tau:
                        reason = "regression_z"
                self.decisions.append(
                    GateDecision(
                        made_at=str(pred_bundle.made_at),
                        valid_at=str(pred_bundle.valid_at[horizon]),
                        horizon=horizon,
                        original_action=action,
                        gated_action=new_action,
                        reason=f"{reason}: {confidence:.3f}",
                    )
                )
        return gated

    def summary(self) -> Dict[str, float]:
        total = len(self.decisions)
        counts: Dict[str, int] = {}
        for decision in self.decisions:
            counts[decision.reason.split(":")[0]] = counts.get(decision.reason.split(":")[0], 0) + 1
        return {"total": total, **{k: v / total for k, v in counts.items()}} if total else {"total": 0}
