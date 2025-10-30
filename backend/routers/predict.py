"""Prediction endpoints for running inference and accessing manifests."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator

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
    files: list[str] = Field(..., min_items=1, description="Uploaded dataset identifiers")
    run_id: Optional[str] = Field(None, description="Model run identifier or 'active'")
    symbols: Optional[list[str]] = Field(None, description="Restrict inference to these symbols")
    start: Optional[datetime] = Field(None, description="Optional UTC start timestamp filter")
    end: Optional[datetime] = Field(None, description="Optional UTC end timestamp filter")
    coverage_target: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Override the target coverage when selecting an abstention threshold",
    )
    tau: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Directly provide an abstention threshold instead of using coverage",
    )

    @validator("symbols", pre=True)
    def _normalize_symbols(cls, value: Optional[list[str]]) -> Optional[list[str]]:  # noqa: N805
        if value is None:
            return None
        normalized = [symbol.upper().strip() for symbol in value if symbol]
        return normalized or None


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
            tau=payload.tau,
        )
    except FileNotFoundError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    predictions_df: pd.DataFrame = result.pop("predictions")
    predictions_path = str(result.pop("predictions_path"))
    manifest_path = str(result.pop("manifest_path"))
    preview = predictions_df.head(200).to_dict(orient="records")

    manifest = result["manifest"].copy()
    artifacts = manifest.get("artifacts", {})

    dataset_summary = _summarise_dataset(frame)

    return {
        **result,
        "predictions_path": predictions_path,
        "manifest_path": manifest_path,
        "artifacts": artifacts,
        "dataset": {
            "files": payload.files,
            **dataset_summary,
        },
        "preview": preview,
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
    preview: list[dict[str, Any]] = []
    if predictions_path.exists():
        predictions_df = pd.read_parquet(predictions_path)
        preview = predictions_df.head(200).to_dict(orient="records")

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
    manifest["preview"] = preview
    return manifest


def _summarise_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"rows": int(len(frame))}
    if "timestamp" in frame.columns:
        timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
        if not timestamps.empty:
            summary["timestamp"] = {
                "min": timestamps.min().isoformat(),
                "max": timestamps.max().isoformat(),
            }
    if "symbol" in frame.columns:
        summary["symbols"] = sorted({str(symbol) for symbol in frame["symbol"].dropna().unique()})
    return summary
