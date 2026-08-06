"""Fan out one mic frame stream to multiple bounded consumer queues."""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from .source import AudioSource

log = logging.getLogger(__name__)


class AudioBroadcaster:
    """Reads frames from an AudioSource and fans them out to subscriber queues.

    Every queue receives the same (already-copied) frame object. Subscribers
    that fall behind lose their oldest frame instead of blocking the live
    stream. A `None` sentinel marks the end of the stream — pushed when the
    source ends or the broadcaster stops.
    """

    def __init__(self, source: AudioSource, queue_size: int = 50) -> None:
        self._source = source
        self._queue_size = queue_size
        self._subscribers: list[asyncio.Queue[np.ndarray | None]] = []
        self._task: asyncio.Task | None = None

    def subscribe(self) -> asyncio.Queue[np.ndarray | None]:
        queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="audio-broadcast")

    async def _run(self) -> None:
        async for frame in self._source.frames():
            for queue in self._subscribers:
                self._put_drop_oldest(queue, frame)
        self._end_stream()

    @staticmethod
    def _put_drop_oldest(queue: asyncio.Queue, item: np.ndarray | None) -> None:
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        queue.put_nowait(item)

    def _end_stream(self) -> None:
        for queue in self._subscribers:
            self._put_drop_oldest(queue, None)

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # a subscriber blocked in get() (e.g. the recorder) must not hang
        self._end_stream()
