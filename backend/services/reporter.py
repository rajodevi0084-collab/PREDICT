"""Asynchronous event reporter used to broadcast training updates."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, DefaultDict, Set


class Reporter:
    """Manage WebSocket subscribers and publish structured log payloads."""

    def __init__(self) -> None:
        self._subscribers: DefaultDict[str, Set[asyncio.Queue[dict[str, Any]]]] = defaultdict(set)
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def subscribe(self, run_id: str) -> AsyncIterator[asyncio.Queue[dict[str, Any]]]:
        """Yield a queue that receives messages for the given ``run_id``."""

        queue: "asyncio.Queue[dict[str, Any]]" = asyncio.Queue()
        async with self._lock:
            self._subscribers[run_id].add(queue)
        try:
            yield queue
        finally:
            await self._remove(run_id, queue)

    def publish(self, run_id: str, event: str, payload: dict[str, Any] | None = None) -> None:
        """Publish an event to all listeners of ``run_id``."""

        message = {
            "run_id": run_id,
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": payload or {},
        }
        coro = self._broadcast(run_id, message)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(coro)
        else:
            loop.create_task(coro)

    def log(self, run_id: str, payload: dict[str, Any]) -> None:
        """Convenience wrapper to publish a log event."""

        self.publish(run_id, event="log", payload=payload)

    async def close(self, run_id: str) -> None:
        """Close all queues associated with ``run_id``."""

        async with self._lock:
            subscribers = list(self._subscribers.pop(run_id, set()))
        for queue in subscribers:
            await queue.put({"event": "close", "run_id": run_id})

    async def _broadcast(self, run_id: str, message: dict[str, Any]) -> None:
        async with self._lock:
            queues = list(self._subscribers.get(run_id, ()))
        for queue in queues:
            await queue.put(message)

    async def _remove(self, run_id: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        async with self._lock:
            subscribers = self._subscribers.get(run_id)
            if not subscribers:
                return
            subscribers.discard(queue)
            if not subscribers:
                self._subscribers.pop(run_id, None)


__all__ = ["Reporter"]
