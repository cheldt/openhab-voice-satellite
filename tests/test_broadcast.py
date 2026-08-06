"""AudioBroadcaster fan-out, backpressure and end-of-stream sentinels."""

from __future__ import annotations

import asyncio

import numpy as np

from openhab_voice_satellite.audio.broadcast import AudioBroadcaster


class ScriptedSource:
    """Yields `n` numbered frames, then optionally stays open."""

    def __init__(self, n: int, stay_open: bool = False) -> None:
        self._n = n
        self._stay_open = stay_open
        self.closed = False

    async def frames(self):
        for i in range(self._n):
            await asyncio.sleep(0)
            yield np.full(4, i, dtype=np.int16)
        while self._stay_open and not self.closed:
            await asyncio.sleep(0.01)

    def close(self) -> None:
        self.closed = True


async def test_fans_out_to_all_subscribers():
    broadcaster = AudioBroadcaster(ScriptedSource(3))
    q1, q2 = broadcaster.subscribe(), broadcaster.subscribe()
    broadcaster.start()
    await asyncio.sleep(0.05)
    for q in (q1, q2):
        frames = [q.get_nowait() for _ in range(3)]
        assert [int(f[0]) for f in frames] == [0, 1, 2]
        assert q.get_nowait() is None  # source-driven end fans the sentinel


async def test_slow_subscriber_drops_oldest():
    broadcaster = AudioBroadcaster(ScriptedSource(10), queue_size=3)
    q = broadcaster.subscribe()
    broadcaster.start()
    await asyncio.sleep(0.05)
    # only the newest 3 frames survive, then the sentinel replaced nothing
    kept = []
    while not q.empty():
        item = q.get_nowait()
        if item is not None:
            kept.append(int(item[0]))
    assert kept == [8, 9]  # 3-slot queue: two newest frames + the None sentinel


async def test_stop_unblocks_waiting_subscriber():
    broadcaster = AudioBroadcaster(ScriptedSource(0, stay_open=True))
    q = broadcaster.subscribe()
    broadcaster.start()

    async def consume():
        return await q.get()  # blocks: no frames are coming

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    assert not consumer.done()
    await broadcaster.stop()
    # stop() must fan out the sentinel or the consumer hangs forever
    assert await asyncio.wait_for(consumer, timeout=1.0) is None


async def test_late_subscriber_misses_earlier_frames():
    broadcaster = AudioBroadcaster(ScriptedSource(2, stay_open=True))
    broadcaster.start()
    await asyncio.sleep(0.05)
    late = broadcaster.subscribe()
    assert late.empty()  # no replay of old frames
    await broadcaster.stop()
    assert late.get_nowait() is None
