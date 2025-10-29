"""Application factory for the PREDICT backend service."""

from __future__ import annotations

import importlib
import pkgutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import APIRouter, FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect

from backend.services.reporting import Reporter

ARTIFACTS_ROOT = Path("artifacts")
REQUIRED_DIRECTORIES = [
    ARTIFACTS_ROOT / "models",
    ARTIFACTS_ROOT / "metrics",
    ARTIFACTS_ROOT / "predictions",
    Path("data"),
]

reporter = Reporter()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Ensure required directories exist before serving requests."""

    for directory in REQUIRED_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)
    yield


def create_app() -> FastAPI:
    """Instantiate and configure the FastAPI application."""

    app = FastAPI(lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.mount("/artifacts", StaticFiles(directory=ARTIFACTS_ROOT, check_dir=False), name="artifacts")

    _register_routers(app)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.websocket("/ws/train/{run_id}")
    async def training_updates(websocket: WebSocket, run_id: str) -> None:
        await websocket.accept()
        try:
            async with reporter.subscribe(run_id) as queue:
                while True:
                    payload = await queue.get()
                    await websocket.send_json(payload)
        except WebSocketDisconnect:
            pass

    return app


def _register_routers(app: FastAPI) -> None:
    """Discover and register routers from the ``backend.routers`` package."""

    try:
        routers_pkg = importlib.import_module("backend.routers")
    except ModuleNotFoundError:
        return

    if not hasattr(routers_pkg, "__path__"):
        return

    for module_info in pkgutil.walk_packages(routers_pkg.__path__, routers_pkg.__name__ + "."):
        if module_info.ispkg:
            continue
        module = importlib.import_module(module_info.name)
        router = getattr(module, "router", None)
        if isinstance(router, APIRouter):
            app.include_router(router)


__all__ = ["create_app", "reporter"]
