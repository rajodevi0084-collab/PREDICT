"""Inference utilities for trained temporal models."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from backend.ml.feature_engineer import FeatureSpec, make_features
from backend.ml.model import SSMEncoder, TemporalTransformer, TemporalTransformerConfig
from backend.services import Registry, RunRegistry

ARTIFACTS_DIR = Path("artifacts")
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"


@dataclass(frozen=True)
class LoadedModel:
    """Container bundling a model with its ancillary metadata."""

    model: nn.Module
    run: dict[str, Any]
    feature_spec: FeatureSpec
    temperature: float
    coverage_target: float
    device: torch.device


def load_model(
    run_id: str | None = None,
    *,
    registry: RunRegistry | None = None,
    device: str | torch.device | None = None,
) -> LoadedModel:
    """Load a trained model checkpoint and supporting metadata."""

    registry = registry or Registry()

    run: Optional[dict[str, Any]]
    if run_id:
        run = _get_run_by_id(registry, run_id)
        if run is None:
            raise ValueError(f"Run '{run_id}' not found in registry")
    else:
        run = registry.resolve_active()
        if run is None:
            runs = registry.list_runs()
            run = max(runs, key=_run_updated_at, default=None)
        if run is None:
            raise RuntimeError("No runs available in registry")

    checkpoint = _resolve_checkpoint_path(Path(run["checkpoint_path"]))

    metadata = run.get("metadata", {}) if isinstance(run.get("metadata"), dict) else {}
    model_config = metadata.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("Run metadata must contain a 'model_config' mapping")

    feature_spec_cfg = metadata.get("feature_spec", {})
    feature_spec = _build_feature_spec(feature_spec_cfg)

    target_device = _resolve_device(device)
    model = _build_model(model_config, metadata.get("ssm_config"))
    state = torch.load(checkpoint, map_location=target_device)
    state_dict = state.get("model_state", state)
    model.load_state_dict(state_dict)
    model.to(target_device)
    model.eval()

    temperature = _extract_temperature(run)
    coverage_target = _extract_coverage_target(metadata, run)

    return LoadedModel(
        model=model,
        run=run,
        feature_spec=feature_spec,
        temperature=temperature,
        coverage_target=coverage_target,
        device=target_device,
    )


def predict(
    frame: pd.DataFrame,
    *,
    run_id: str | None = None,
    registry: RunRegistry | None = None,
    coverage_target: float | None = None,
    temperature: float | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Generate predictions for ``frame`` using the specified model run."""

    loaded = load_model(run_id, registry=registry, device=device)
    model = loaded.model
    run = loaded.run

    cov_target = float(coverage_target if coverage_target is not None else loaded.coverage_target)
    cov_target = float(np.clip(cov_target, 0.0, 1.0))

    temp = float(temperature if temperature is not None else loaded.temperature)
    if not math.isfinite(temp) or temp <= 0:
        temp = 1.0

    X, _, _, meta = make_features(frame, loaded.feature_spec)
    if X.empty:
        raise ValueError("No rows available after feature engineering")

    features = torch.from_numpy(X.to_numpy(dtype=np.float32, copy=True)).unsqueeze(1)
    features = features.to(loaded.device)

    with torch.no_grad():
        outputs = model(features)
        logits = outputs["logits_cls"]
        reg_pred = outputs["y_reg"].squeeze(-1)

    logits = logits.to(torch.float32) / temp
    probs = torch.softmax(logits, dim=-1)
    if probs.shape[1] < 3:
        raise ValueError("Model must output at least three classes (down, neutral, up)")

    confidences, pred_indices = torch.max(probs, dim=-1)
    tau = _select_tau_for_coverage(confidences, cov_target)
    abstain_mask = confidences < tau

    probs_cpu = probs.detach().cpu()
    reg_cpu = reg_pred.detach().cpu().to(torch.float32)

    base_prices = frame.loc[X.index, "close"].astype(float).to_numpy(copy=True)
    y_pred_price = base_prices + reg_cpu.numpy()

    predictions = pd.DataFrame(
        {
            "timestamp": meta["timestamp"].to_numpy(),
            "symbol": meta["symbol"].to_numpy(),
            "p_down": probs_cpu[:, 0].numpy(),
            "p_up": probs_cpu[:, -1].numpy(),
            "p_neutral": probs_cpu[:, 1].numpy() if probs_cpu.shape[1] > 2 else np.nan,
            "y_pred_price": y_pred_price,
            "abstain": abstain_mask.cpu().numpy(),
            "confidence": confidences.detach().cpu().numpy(),
            "predicted_class": pred_indices.detach().cpu().numpy(),
            "tau": tau,
            "coverage_target": cov_target,
            "run_id": run["id"],
        }
    )

    output_dir = PREDICTIONS_DIR / run["id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)

    manifest = {
        "run_id": run["id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_rows": int(len(predictions)),
        "tau": float(tau),
        "coverage_target": float(cov_target),
        "temperature": float(temp),
        "feature_columns": list(X.columns),
        "artifacts": {
            "predictions": f"/artifacts/{predictions_path.relative_to(ARTIFACTS_DIR).as_posix()}",
        },
    }

    manifest_path = output_dir / "manifest.json"
    manifest["artifacts"]["manifest"] = f"/artifacts/{manifest_path.relative_to(ARTIFACTS_DIR).as_posix()}"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "run_id": run["id"],
        "predictions_path": predictions_path,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "num_rows": int(len(predictions)),
        "tau": float(tau),
        "coverage_target": float(cov_target),
        "temperature": float(temp),
        "feature_columns": list(X.columns),
    }


def _resolve_checkpoint_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_dir():
        best = path / "best.pt"
        if best.exists():
            return best
        candidates = sorted(path.glob("*.pt"))
        if candidates:
            return candidates[0]
        raise FileNotFoundError(f"No checkpoint files found in directory: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {path}")
    return path


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _build_model(config_dict: dict[str, Any], ssm_cfg: Any) -> nn.Module:
    config = TemporalTransformerConfig(**config_dict)
    ssm_encoder = None
    if isinstance(ssm_cfg, dict):
        ssm_encoder = SSMEncoder(**ssm_cfg)
    return TemporalTransformer(config, ssm_encoder=ssm_encoder)


def _build_feature_spec(spec_dict: Any) -> FeatureSpec:
    if not isinstance(spec_dict, dict):
        return FeatureSpec()
    payload = spec_dict.copy()
    windows = payload.get("windows")
    if isinstance(windows, Iterable) and not isinstance(windows, (str, bytes)):
        payload["windows"] = tuple(int(w) for w in windows)
    return FeatureSpec(**payload)


def _get_run_by_id(registry: RunRegistry, run_id: str) -> Optional[dict[str, Any]]:
    for run in registry.list_runs():
        if run.get("id") == run_id:
            return run
    return None


def _run_updated_at(run: dict[str, Any]) -> str:
    updated = run.get("updated_at")
    if isinstance(updated, str):
        return updated
    return ""


def _extract_temperature(run: dict[str, Any]) -> float:
    metadata = run.get("metadata")
    if isinstance(metadata, dict):
        temp = metadata.get("temperature")
        if isinstance(temp, (int, float)):
            return float(temp)
    metrics = run.get("metrics")
    if isinstance(metrics, dict):
        temp = metrics.get("temperature")
        if isinstance(temp, (int, float)):
            return float(temp)
    return 1.0


def _extract_coverage_target(metadata: dict[str, Any], run: dict[str, Any]) -> float:
    candidates = [
        metadata.get("coverage_target") if isinstance(metadata, dict) else None,
        metadata.get("default_coverage_target") if isinstance(metadata, dict) else None,
    ]
    metrics = run.get("metrics")
    if isinstance(metrics, dict):
        candidates.append(metrics.get("coverage_target"))
    for value in candidates:
        if isinstance(value, (int, float)):
            return float(np.clip(value, 0.0, 1.0))
    return 0.75


def _select_tau_for_coverage(confidences: Tensor, coverage: float) -> float:
    if confidences.numel() == 0:
        return 1.0
    coverage = float(np.clip(float(coverage), 0.0, 1.0))
    if coverage <= 0:
        return float(confidences.max().item())
    sorted_conf, _ = torch.sort(confidences.detach().cpu(), descending=True)
    k = max(int(math.ceil(len(sorted_conf) * coverage)) - 1, 0)
    return float(sorted_conf[k].item())


__all__ = ["LoadedModel", "load_model", "predict"]
