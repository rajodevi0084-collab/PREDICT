"""FastAPI router exposing inference endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .next_tick_service import predict as predict_next_tick

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("/next-tick")
def next_tick(payload: dict) -> dict:
    try:
        return predict_next_tick(payload)
    except KeyError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=str(exc)) from exc
