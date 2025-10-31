"""Training wrapper around PPO for discrete trading policies."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd

try:  # pragma: no cover - optional dependency
    from stable_baselines3 import PPO
except ImportError:  # pragma: no cover
    PPO = None


@dataclass
class RLSnapshot:
    policy_path: Path
    metrics: Dict[str, float]


class RLTrainer:
    """Wrapper coordinating RL training and evaluation."""

    def __init__(self, env_factory: Callable[..., Any], algo_cfg: Dict[str, Any]) -> None:
        self.env_factory = env_factory
        self.algo_cfg = algo_cfg
        self.snapshot: Optional[RLSnapshot] = None

    def train(self, train_window) -> RLSnapshot:
        if PPO is None:  # pragma: no cover - ensures deterministic behaviour when dependency missing
            raise ImportError("stable-baselines3 is required for RL training")
        env = self.env_factory(window=train_window, mode="train")
        cfg = dict(self.algo_cfg)
        total_timesteps = int(cfg.pop("total_timesteps", 10_000))
        policy_kwargs = cfg.pop("policy_kwargs", {"net_arch": [cfg.pop("net_width", 64)]})
        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, **cfg)
        model.learn(total_timesteps=total_timesteps)
        snapshot_path = Path("artifacts") / "rl" / f"ppo_{train_window.train_start:%Y%m%d}.zip"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(snapshot_path.as_posix())
        self.snapshot = RLSnapshot(policy_path=snapshot_path, metrics={"episodes": total_timesteps})
        return self.snapshot

    def evaluate(self, test_window) -> pd.DataFrame:
        if PPO is None:
            raise ImportError("stable-baselines3 is required for RL evaluation")
        if self.snapshot is None:
            raise RuntimeError("Model not trained before evaluation")
        env = self.env_factory(window=test_window, mode="test")
        model = PPO.load(self.snapshot.policy_path.as_posix())
        obs, _ = env.reset()
        records = []
        done = False
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            records.append({"reward": reward, "info": info})
            done = terminated or truncated
        return pd.DataFrame(records)

    def evaluate_metrics(self, predictions, trades: pd.DataFrame) -> Dict[str, float]:
        if trades.empty:
            return {}
        pnl = trades["reward"].cumsum()
        return {
            "sharpe": float(pnl.mean()),
            "sortino": float(pnl.mean()),
            "profit_factor": float(pnl[pnl > 0].sum() / (abs(pnl[pnl < 0].sum()) + 1e-9)),
            "mdd": float((pnl.cummax() - pnl).max()),
        }
