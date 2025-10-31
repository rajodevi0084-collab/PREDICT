"""Idempotent execution stub used for backtesting."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class Order:
    client_order_id: str
    symbol: str
    side: str
    qty: float
    ts: datetime


def build_coid(run_id: str, symbol: str, ts: datetime, action: str) -> str:
    return f"{run_id}-{symbol}-{ts.isoformat()}-{action}"


class ExecutionStub:
    def __init__(self) -> None:
        self.sent: Dict[str, Order] = {}
        self.fills: List[Dict] = []

    def send(self, order: Order) -> Optional[Dict]:
        if order.client_order_id in self.sent:
            return None
        self.sent[order.client_order_id] = order
        fill = {
            "client_order_id": order.client_order_id,
            "symbol": order.symbol,
            "side": order.side,
            "qty": order.qty,
            "filled_qty": order.qty,
            "status": "filled",
            "ts": order.ts,
        }
        self.fills.append(fill)
        return fill

    def eod_flatten(self, symbol: str, price: float, position: float, ts: datetime) -> Optional[Dict]:
        if position == 0:
            return None
        side = "sell" if position > 0 else "buy"
        order = Order(
            client_order_id=build_coid("EOD", symbol, ts, "flatten"),
            symbol=symbol,
            side=side,
            qty=abs(position),
            ts=ts,
        )
        return self.send(order)
