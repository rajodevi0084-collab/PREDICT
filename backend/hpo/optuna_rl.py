"""Optuna search space for RL hyper-parameters."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

try:  # pragma: no cover
    import optuna
except ImportError:  # pragma: no cover
    optuna = None


def _objective_factory(walkforward_runner, trials: int):
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "n_steps": trial.suggest_int("n_steps", 64, 1024, log=True),
            "batch_size": trial.suggest_int("batch_size", 32, 512, log=True),
            "entropy_coef": trial.suggest_float("entropy_coef", 1e-4, 0.02, log=True),
            "gamma": trial.suggest_float("gamma", 0.90, 0.9999),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 0.99),
            "net_width": trial.suggest_int("net_width", 32, 256, log=True),
            "turnover_lambda": trial.suggest_float("turnover_lambda", 0.0, 1.0),
            "dd_lambda": trial.suggest_float("dd_lambda", 0.0, 1.0),
            "coverage_tau": trial.suggest_float("coverage_tau", 0.5, 0.9),
        }
        if not hasattr(walkforward_runner, "run_with_params"):
            raise AttributeError("walkforward_runner must implement run_with_params for Optuna search")
        reports = walkforward_runner.run_with_params(params)
        summary = walkforward_runner.aggregate(reports)
        return -summary.get("sharpe", 0.0) * -1

    return objective


def run_optuna(walkforward_runner, trials: int, output_dir: Path | None = None) -> Dict[str, Any]:
    if optuna is None:  # pragma: no cover
        raise ImportError("optuna is required for hyper-parameter optimisation")
    study = optuna.create_study(direction="maximize", study_name="ppo_rl")
    study.optimize(_objective_factory(walkforward_runner, trials), n_trials=trials)
    best = study.best_params
    output_dir = output_dir or Path("artifacts") / "hpo"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    best_path = output_dir / f"best_{timestamp}.yaml"
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pyyaml is required to persist Optuna results") from exc
    with best_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(best, fh)
    top_trials = sorted(study.trials, key=lambda t: t.value or float("-inf"), reverse=True)[:5]
    for trial in top_trials:
        print(f"Trial {trial.number}: value={trial.value}, params={trial.params}")
    return {"best_params": best, "path": best_path}
