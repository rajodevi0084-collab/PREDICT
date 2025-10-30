"""Prediction endpoints orchestrating inference workflows."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.ml.data_loader import (
    DATA_DIR,
    SUPPORTED_EXTENSIONS,
    load_with_metadata,
    slice_time,
)
from backend.ml.infer import predict as run_prediction
from backend.services import Registry

router = APIRouter(prefix="/predict", tags=["predict"])


class PredictRequest(BaseModel):
    file_id: str = Field(..., description="Identifier of the uploaded dataset")
    run_id: Optional[str] = Field(None, description="Model run identifier to use for inference")
    start: Optional[datetime] = Field(None, description="Optional start timestamp filter")
    end: Optional[datetime] = Field(None, description="Optional end timestamp filter")
    symbols: Optional[list[str]] = Field(None, description="Restrict inference to these symbols")
    coverage_target: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Override the default coverage target for abstention",
    )


def _resolve_data_file(file_id: str) -> Path:
    for suffix in sorted(SUPPORTED_EXTENSIONS):
        candidate = DATA_DIR / f"{file_id}{suffix}"
        if candidate.exists():
            return candidate
    matches = list(DATA_DIR.glob(f"{file_id}.*"))
    if matches:
        return matches[0]
    raise HTTPException(status_code=404, detail="Dataset not found")


@router.post("/run")
async def run_prediction_endpoint(payload: PredictRequest) -> dict[str, Any]:
    path = _resolve_data_file(payload.file_id)
    frame, metadata = load_with_metadata(path)

    if payload.start or payload.end:
        frame = slice_time(frame, start=payload.start, end=payload.end)

    if payload.symbols:
        if "symbol" not in frame.columns:
            raise HTTPException(status_code=400, detail="Dataset does not contain a 'symbol' column")
        symbols = {symbol.upper().strip() for symbol in payload.symbols if symbol}
        if not symbols:
            raise HTTPException(status_code=400, detail="No valid symbols provided")
        frame = frame[frame["symbol"].astype(str).str.upper().isin(sorted(symbols))]

    if frame.empty:
        raise HTTPException(status_code=400, detail="No rows available after applying filters")

    registry = Registry()
    try:
        result = run_prediction(
            frame,
            run_id=payload.run_id,
            registry=registry,
            coverage_target=payload.coverage_target,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    manifest = result["manifest"].copy()
    artifacts = manifest.get("artifacts", {})

    return {
        "run_id": result["run_id"],
        "num_rows": result["num_rows"],
        "tau": result["tau"],
        "coverage_target": result["coverage_target"],
        "temperature": result["temperature"],
        "feature_columns": result["feature_columns"],
        "artifacts": artifacts,
        "dataset": {
            "file_id": payload.file_id,
            "filename": path.name,
            "metadata": metadata.to_dict(),
        },
    }


@router.get("/latest")
async def latest_prediction() -> dict[str, Any]:
    predictions_root = Path("artifacts") / "predictions"
    manifest_files = sorted(predictions_root.glob("*/manifest.json"))
    if not manifest_files:
        raise HTTPException(status_code=404, detail="No prediction manifests found")

    latest = max(manifest_files, key=lambda path: path.stat().st_mtime)
    manifest = json.loads(latest.read_text(encoding="utf-8"))

    predictions_path = latest.parent / "predictions.parquet"
    manifest.setdefault("artifacts", {})
    manifest["artifacts"].setdefault(
        "manifest",
        f"/artifacts/{latest.relative_to(Path('artifacts')).as_posix()}",
    )
    manifest["artifacts"].setdefault(
        "predictions",
        f"/artifacts/{predictions_path.relative_to(Path('artifacts')).as_posix()}",
    )

    manifest["path"] = str(latest)
    manifest["predictions_path"] = str(predictions_path)
    return manifest
