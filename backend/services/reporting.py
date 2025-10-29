"""Utilities for broadcasting training updates to connected clients."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, DefaultDict, Set


class Reporter:
    """Broadcast JSON-serialisable payloads to subscribers by run identifier."""

    def __init__(self) -> None:
        self._subscribers: DefaultDict[str, Set[asyncio.Queue[Any]]] = defaultdict(set)
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def subscribe(self, run_id: str) -> AsyncIterator[asyncio.Queue[Any]]:
        """Yield a queue that receives events for the given ``run_id``."""

        queue: "asyncio.Queue[Any]" = asyncio.Queue()
        async with self._lock:
            self._subscribers[run_id].add(queue)
        try:
            yield queue
        finally:
            async with self._lock:
                subscribers = self._subscribers.get(run_id)
                if subscribers and queue in subscribers:
                    subscribers.remove(queue)
                    if not subscribers:
                        self._subscribers.pop(run_id, None)

    async def broadcast(self, run_id: str, payload: Any) -> None:
        """Send ``payload`` to all subscribers of ``run_id``."""

        async with self._lock:
            recipients = list(self._subscribers.get(run_id, ()))
        for queue in recipients:
            await queue.put(payload)
