"""Cost modelling primitives for Indian equities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping


@dataclass
class CostBreakdown:
    """Detailed breakdown returned by :class:`CostModel`. Values are in currency."""

    brokerage: float
    stt: float
    gst: float
    stamp: float
    exchange: float
    slippage_bp: float
    slippage_value: float
    impact_value: float
    total_value: float


class CostModel:
    """Apply regulatory and micro-structure costs for Indian markets."""

    def __init__(self, params_from_yaml: Mapping[str, float]) -> None:
        required = [
            "brokerage_bps",
            "stt_bps",
            "gst_bps",
            "stamp_bps",
            "exchange_bps",
            "slippage_half_spread_bp",
            "impact_coeff",
        ]
        for key in required:
            if key not in params_from_yaml:
                raise KeyError(f"Missing cost parameter: {key}")
        self.params = dict(params_from_yaml)

    def quote_half_spread(self, symbol: str, t) -> float:
        """Return half-spread in basis points. Currently static."""

        return float(self.params.get("slippage_half_spread_bp", 0.0))

    def estimate_slippage(self, bp_half_spread: float, qty: float, adv: float, impact_coeff: float | None = None) -> float:
        """Estimate total slippage in basis points given size and liquidity."""

        if adv <= 0:
            raise ValueError("ADV must be positive for slippage estimation")
        coeff = impact_coeff if impact_coeff is not None else float(self.params["impact_coeff"])
        market_impact = coeff * (qty / adv)
        return bp_half_spread * 2 + market_impact * 1e4

    def apply_all(self, price: float, qty: float, side: str, symbol: str, t, adv: float | None = None) -> CostBreakdown:
        """Apply all cost components to a trade."""

        trade_value = price * abs(qty)
        bp = {k: float(v) for k, v in self.params.items() if k.endswith("_bps")}
        brokerage = trade_value * bp["brokerage_bps"] / 1e4
        stt = trade_value * bp["stt_bps"] / 1e4
        gst = (brokerage + stt) * bp["gst_bps"] / 1e4
        stamp = trade_value * bp["stamp_bps"] / 1e4
        exchange = trade_value * bp["exchange_bps"] / 1e4
        half_spread = self.quote_half_spread(symbol, t)
        adv = adv or max(abs(qty), 1.0)
        slippage_bp = self.estimate_slippage(half_spread, abs(qty), adv)
        slippage_value = trade_value * slippage_bp / 1e4
        impact_value = trade_value * (self.params["impact_coeff"] * (abs(qty) / adv))
        total = brokerage + stt + gst + stamp + exchange + slippage_value + impact_value
        return CostBreakdown(
            brokerage=brokerage,
            stt=stt,
            gst=gst,
            stamp=stamp,
            exchange=exchange,
            slippage_bp=slippage_bp,
            slippage_value=slippage_value,
            impact_value=impact_value,
            total_value=total,
        )

    def apply_to_trade(self, trade: Mapping[str, float]) -> Dict[str, float]:
        """Return a new trade dict with cost breakdown embedded."""

        breakdown = self.apply_all(
            price=float(trade["price"]),
            qty=float(trade["qty"]),
            side=str(trade.get("side", "buy")),
            symbol=str(trade.get("symbol", "")),
            t=trade.get("ts"),
            adv=float(trade.get("adv", abs(trade["qty"]))),
        )
        enriched = dict(trade)
        enriched["cost_breakdown"] = breakdown
        enriched["total_costs"] = breakdown.total_value
        return enriched
