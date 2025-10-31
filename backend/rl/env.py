"""Gymnasium trading environment with strict alignment semantics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - fallback for documentation builds
    gym = None
    spaces = None


ACTION_MAP = {
    0: ("Flat", 0.0),
    1: ("Long1", 1.0),
    2: ("Long2", 2.0),
    3: ("Short1", -1.0),
    4: ("Short2", -2.0),
}


@dataclass
class PositionState:
    position: float = 0.0
    cash: float = 0.0
    equity_peak: float = 0.0

    def mark_to_market(self, price: float) -> float:
        return self.cash + self.position * price

    def drawdown(self, equity: float) -> float:
        self.equity_peak = max(self.equity_peak, equity)
        return (self.equity_peak - equity) / (self.equity_peak + 1e-9)

    def apply_action(self, target_exposure: float, price: float) -> float:
        delta = target_exposure - self.position
        self.cash -= delta * price
        self.position = target_exposure
        return abs(delta)


class TradingEnv(gym.Env if gym else object):  # type: ignore[misc]
    """A small discrete-action trading environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: Dict[str, np.ndarray],
        turnover_lambda: float = 0.0,
        dd_lambda: float = 0.0,
        max_daily_loss_pct: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        if gym is None:  # pragma: no cover - guard for optional dependency
            raise ImportError("gymnasium is required to use TradingEnv")
        super().__init__()
        self.data = data
        self.features = data["features"]
        self.returns = data["returns"]
        self.predictions = data.get("predictions")
        self.prices = data["prices"]
        self.turnover_lambda = turnover_lambda
        self.dd_lambda = dd_lambda
        self.max_daily_loss_pct = max_daily_loss_pct
        self.state = PositionState()
        self.idx = 0
        self.random_generator = np.random.default_rng(seed)
        feat_dim = self.features.shape[1]
        self.action_space = spaces.Discrete(len(ACTION_MAP))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(feat_dim + 5,), dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        if self.predictions is None:
            pred = np.zeros(3, dtype=float)
        else:
            pred = self.predictions[self.idx]
        feature_vec = self.features[self.idx]
        obs = np.concatenate(
            [
                feature_vec,
                np.asarray(pred, dtype=float),
                np.asarray(
                    [
                        self.state.position,
                        self.state.cash,
                        float(self.idx) / len(self.prices),
                        self.prices[self.idx],
                        self.returns[self.idx - 1] if self.idx > 0 else 0.0,
                    ]
                ),
            ]
        )
        return obs.astype(np.float32)

    def seed(self, seed: Optional[int] = None) -> None:
        self.random_generator = np.random.default_rng(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):  # type: ignore[override]
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
        self.idx = 0
        self.state = PositionState()
        self.state.equity_peak = 0.0
        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):  # type: ignore[override]
        label, exposure = ACTION_MAP[action]
        price = self.prices[self.idx]
        turnover = self.state.apply_action(exposure, price)
        next_price = self.prices[self.idx + 1] if self.idx + 1 < len(self.prices) else price
        pnl = self.state.position * (next_price - price)
        reward = pnl - self.turnover_lambda * turnover
        equity = self.state.mark_to_market(next_price)
        drawdown = self.state.drawdown(equity)
        reward -= self.dd_lambda * drawdown
        info = {
            "action_label": label,
            "turnover": turnover,
            "drawdown": drawdown,
            "kill_switch": False,
        }
        self.idx += 1
        terminated = self.idx >= len(self.prices) - 1
        truncated = False
        if self.max_daily_loss_pct > 0:
            reference = self.prices[0]
            if equity < reference * (1 - self.max_daily_loss_pct / 100):
                info["kill_switch"] = True
                terminated = True
        next_index = min(self.idx, len(self.prices) - 1)
        self.idx = next_index
        obs = self._get_obs()
        return obs, reward, terminated, truncated, info
