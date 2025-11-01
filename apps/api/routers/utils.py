"""Utility endpoints for data conversion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter(prefix="/utils", tags=["utils"])


@router.post("/csv-to-parquet")
async def csv_to_parquet(symbol: str, file: UploadFile = File(...)) -> dict[str, str]:
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol must be provided")

    try:
        content = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(content))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {exc}") from exc

    if "timestamp" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must include a 'timestamp' column")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"])

    output_dir = Path("data/parquet")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{symbol.upper()}.parquet"
    df.to_parquet(path)

    return {"path": str(path)}
