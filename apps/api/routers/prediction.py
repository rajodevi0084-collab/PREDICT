"""API router bridging the inference services."""

from __future__ import annotations

from fastapi import APIRouter

from services.inference.next_tick_service import predict as predict_next_tick

router = APIRouter(prefix="/prediction", tags=["prediction"])


@router.post("/next-tick")
def next_tick(payload: dict) -> dict:
    return predict_next_tick(payload)
