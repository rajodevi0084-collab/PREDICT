"""Training endpoints for orchestrating model runs."""

from __future__ import annotations

import asyncio
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field, validator

from backend.ml.data_loader import load_dataset
from backend.ml.feature_engineer import FeatureSpec, build_feature_map, make_features
from backend.ml.model import TemporalTransformer, TemporalTransformerConfig
from backend.ml.trainer import Trainer, TrainerConfig
from backend.services import Registry

try:  # Avoid circular imports during tooling
    from backend.main import reporter
except ModuleNotFoundError:  # pragma: no cover - during static analysis
    reporter = None  # type: ignore


router = APIRouter(prefix="/train", tags=["train"])


class TrainStartRequest(BaseModel):
    files: list[str] = Field(..., min_items=1, description="Uploaded dataset identifiers")
    symbols: Optional[list[str]] = Field(None, description="Subset of symbols to train on")
    horizon: int = Field(5, ge=1, description="Forecast horizon in steps")
    epochs: int = Field(25, ge=1, description="Maximum number of training epochs")
    feature_budget: int = Field(128, ge=16, description="Hidden dimension for the transformer")
    coverage: float = Field(0.75, ge=0.0, le=1.0, description="Target coverage for abstention")

    @validator("symbols", pre=True)
    def _normalize_symbols(cls, value: Optional[Iterable[str]]) -> Optional[list[str]]:  # noqa: N805
        if value is None:
            return None
        symbols = [symbol.upper().strip() for symbol in value if symbol]
        return symbols or None


class PromoteRequest(BaseModel):
    run_id: str = Field(..., description="Run identifier to promote")


@router.post("/start")
async def start_training(payload: TrainStartRequest, background_tasks: BackgroundTasks) -> dict[str, str]:
    if reporter is None:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail="Reporter service unavailable")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M-TRAIN")
    registry = Registry()
    checkpoint_dir = Path("artifacts") / "models" / run_id
    metadata = {
        "status": "queued",
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "coverage_target": payload.coverage,
    }
    registry.add_run(run_id, checkpoint_dir, metadata=metadata)
    registry.update_run(run_id, status="running")

    background_tasks.add_task(
        _execute_training,
        run_id,
        payload.dict(),
    )

    return {"run_id": run_id}


@router.post("/promote")
async def promote_run(request: PromoteRequest) -> dict[str, Any]:
    registry = Registry()
    try:
        run = registry.promote(request.run_id)
    except (FileNotFoundError, KeyError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"run_id": run["id"], "status": "active"}


@router.get("/runs")
async def list_runs() -> dict[str, Any]:
    registry = Registry()
    active = registry.resolve_active()
    return {"runs": registry.list_runs(), "active": active.get("id") if active else None}


def _execute_training(run_id: str, payload: Dict[str, Any]) -> None:
    registry = Registry()
    try:
        files = payload.get("files", [])
        symbols = payload.get("symbols")
        horizon = int(payload.get("horizon", 5))
        epochs = int(payload.get("epochs", 25))
        feature_budget = int(payload.get("feature_budget", 128))
        coverage = float(payload.get("coverage", 0.75))

        reporter.log(run_id, {"event": "loading_data", "files": files})
        frame = load_dataset(files, symbols=symbols)
        if frame.empty:
            raise ValueError("No rows available after loading datasets")

        feature_spec = FeatureSpec(horizon=horizon)
        features, _, _, _ = make_features(frame, feature_spec)
        if features.empty:
            raise ValueError("Feature engineering produced an empty matrix")

        input_dim = features.shape[1]
        model_dim = max(feature_budget, 32)
        num_heads = max(1, model_dim // 32)
        model_config = TemporalTransformerConfig(
            input_dim=input_dim,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=4,
            ff_dim=model_dim * 4,
            num_classes=3,
            regression_dim=1,
        )
        model = TemporalTransformer(model_config)

        trainer_config = TrainerConfig(max_epochs=epochs)
        trainer = Trainer(
            model,
            feature_spec,
            reporter=reporter,
            run_id=run_id,
            config=trainer_config,
        )

        registry.update_run(
            run_id,
            metadata={
                "model_config": asdict(model_config),
                "feature_spec": {
                    "windows": list(feature_spec.windows),
                    "horizon": feature_spec.horizon,
                    "epsilon": feature_spec.epsilon,
                },
                "coverage_target": coverage,
                "feature_columns": list(features.columns),
            },
        )

        metrics = trainer.fit(frame)
        trainer.save_model_card(
            feature_map=build_feature_map(features.columns),
            data_ranges=_summarise_data_ranges(frame),
        )

        registry.update_run(
            run_id,
            status="completed",
            metrics=metrics,
            metadata={
                "temperature": metrics.get("temperature"),
                "coverage_target": coverage,
            },
        )
        reporter.log(run_id, {"event": "run_completed", "metrics": metrics})
    except Exception as exc:  # pragma: no cover - defensive
        registry.update_run(
            run_id,
            status="failed",
            metadata={"error": str(exc), "traceback": traceback.format_exc()},
        )
        if reporter is not None:
            reporter.log(run_id, {"event": "run_failed", "error": str(exc)})
    finally:
        if reporter is not None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(reporter.close(run_id))
            else:
                loop.create_task(reporter.close(run_id))


def _summarise_data_ranges(frame: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if "timestamp" in frame.columns:
        timestamps = frame["timestamp"].dropna()
        if not timestamps.empty:
            summary["timestamp"] = {
                "min": timestamps.min().isoformat(),
                "max": timestamps.max().isoformat(),
            }
    if "symbol" in frame.columns:
        symbols = sorted({str(symbol) for symbol in frame["symbol"].dropna().unique()})
        summary["symbols"] = symbols
    return summary
