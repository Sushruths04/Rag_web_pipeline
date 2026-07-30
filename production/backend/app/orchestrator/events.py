"""Asyncio fan-out of persisted events to WebSocket subscribers.

The pump thread calls publish_threadsafe(); subscribers are asyncio queues
owned by websocket handlers. Until a loop is bound (server startup), publish
is a silent no-op — events are still persisted, subscribers replay from the
store on connect, so nothing is lost.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Optional


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[str, set[asyncio.Queue]] = defaultdict(set)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self, run_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subs[run_id].add(q)
        return q

    def unsubscribe(self, run_id: str, q: asyncio.Queue) -> None:
        self._subs[run_id].discard(q)

    def publish_threadsafe(self, event: dict) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self._publish, event)

    def _publish(self, event: dict) -> None:
        for q in list(self._subs.get(event["run_id"], ())):
            q.put_nowait(event)
