import asyncio
import time

import numpy as np
import pytest

gi = pytest.importorskip("gi")

from openhab_voice_satellite.audio import gst_sink  # noqa: E402
from openhab_voice_satellite.audio.gst_sink import (  # noqa: E402
    SINK_RESIDUAL_S,
    PipewireSink,
)

FAKESINK = (
    "appsrc name=src format=time is-live=true max-bytes=0 block=false "
    "! audioconvert ! audioresample ! volume name=vol ! fakesink sync=%s"
)


def _pcm(seconds: float, rate: int = 16000) -> np.ndarray:
    return np.zeros(int(seconds * rate), dtype=np.int16)


@pytest.fixture
def make_sink(monkeypatch):
    """Sink factory over a fake pipeline; closes every created sink on teardown."""
    sinks: list[PipewireSink] = []

    def make(sync: str = "false", **kwargs) -> PipewireSink:
        monkeypatch.setattr(
            PipewireSink, "_describe", staticmethod(lambda target: FAKESINK % sync)
        )
        sink = PipewireSink(device=None, **kwargs)
        sinks.append(sink)
        return sink

    yield make
    for sink in sinks:
        sink.close()


class PacedQueue:
    """Emulates the appsrc queue draining at playout rate (bytes/s).

    fakesink consumes buffers instantly, so the real `current-level-bytes`
    is useless for testing the byte-accounted playout wait; this double
    simulates only the level observable.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, rate: int = 16000) -> None:
        self._loop = loop
        self._bps = rate * 2
        self._level = 0.0
        self._t = loop.time()
        self.flushes = 0

    def _decay(self) -> None:
        now = self._loop.time()
        self._level = max(0.0, self._level - (now - self._t) * self._bps)
        self._t = now

    def on_push(self, nbytes: int) -> None:
        self._decay()
        self._level += nbytes

    def level(self) -> int:
        self._decay()
        return int(self._level)

    def flush(self) -> None:
        self._decay()
        self._level = 0.0
        self.flushes += 1


@pytest.fixture
def paced_sink(make_sink, monkeypatch):
    """Sink whose appsrc level drains at playout rate; returns (sink, paced)."""

    def make(**kwargs) -> tuple[PipewireSink, PacedQueue]:
        sink = make_sink(**kwargs)
        paced = PacedQueue(asyncio.get_running_loop())
        orig_push, orig_flush = sink._push, sink._flush
        monkeypatch.setattr(
            sink, "_push", lambda b: (paced.on_push(len(b)), orig_push(b))[1]
        )
        monkeypatch.setattr(sink, "_queued_bytes", paced.level)
        monkeypatch.setattr(sink, "_flush", lambda: (paced.flush(), orig_flush())[1])
        return sink, paced

    return make


async def test_play_completes(make_sink):
    sink = make_sink()
    await asyncio.wait_for(sink.play(_pcm(0.5), 16000), timeout=5.0)


async def test_pipeline_survives_across_plays(make_sink):
    sink = make_sink()
    pipeline = sink._pipeline
    await asyncio.wait_for(sink.play(_pcm(0.2), 16000), timeout=5.0)
    await asyncio.wait_for(sink.play(_pcm(0.2), 22050), timeout=5.0)
    assert sink._pipeline is pipeline


async def test_stop_aborts_playback_quickly(paced_sink):
    sink, _ = paced_sink()
    task = asyncio.create_task(sink.play(_pcm(5.0), 16000))
    await asyncio.sleep(0.3)
    start = time.monotonic()
    sink.stop()
    await asyncio.wait_for(task, timeout=2.0)
    assert time.monotonic() - start < 1.0


async def test_new_play_replaces_queued_tail(paced_sink):
    sink, paced = paced_sink()
    long_play = asyncio.create_task(sink.play(_pcm(5.0), 16000))
    await asyncio.sleep(0.3)
    # a second play must not wait for the first one's 5s to elapse
    await asyncio.wait_for(sink.play(_pcm(0.2), 16000), timeout=2.0)
    assert paced.flushes >= 1  # the second play took the interrupt-flush path
    sink.stop()
    await asyncio.wait_for(long_play, timeout=2.0)


async def test_duck_applies_to_pipeline_and_persists(paced_sink):
    sink, _ = paced_sink()
    sink.duck(0.4)
    assert sink._volume.get_property("volume") == pytest.approx(0.4)
    task = asyncio.create_task(sink.play(_pcm(2.0), 16000))
    await asyncio.sleep(0.3)
    assert sink._volume.get_property("volume") == pytest.approx(0.4)
    sink.unduck()
    assert sink._volume.get_property("volume") == pytest.approx(1.0)
    sink.stop()
    await asyncio.wait_for(task, timeout=2.0)


async def test_wakeup_preamble_on_first_and_idle_plays_only(make_sink, monkeypatch):
    sink = make_sink(wakeup_preamble_ms=400)
    sink._keepalive_task.cancel()  # keep push counts deterministic
    pushed: list[int] = []
    orig_push = sink._push
    monkeypatch.setattr(sink, "_push", lambda b: (pushed.append(len(b)), orig_push(b))[1])
    pcm_bytes = len(_pcm(0.2).tobytes())
    preamble_bytes = 16000 * 400 // 1000 * 2

    await asyncio.wait_for(sink.play(_pcm(0.2), 16000), timeout=5.0)
    first_sound = sum(pushed)
    assert first_sound >= pcm_bytes + preamble_bytes  # first play gets preamble

    pushed.clear()
    await asyncio.wait_for(sink.play(_pcm(0.2), 16000), timeout=5.0)
    assert sum(pushed) == pcm_bytes  # immediate follow-up gets no preamble


async def test_rate_change_backlog_counted_at_old_rate(make_sink, monkeypatch):
    # audio queued at 16 kHz still plays out at 16 kHz after a retune to
    # 32 kHz; the drain estimate must not halve it (gst_sink backlog_s fix)
    sink = make_sink()
    sink._keepalive_task.cancel()
    backlog = _pcm(2.0, rate=16000)  # 2 s of queued audio at the old rate
    monkeypatch.setattr(sink, "_queued_bytes", lambda: len(backlog.tobytes()))

    loop = asyncio.get_running_loop()
    before = loop.time()
    task = asyncio.create_task(sink.play(_pcm(1.0, rate=32000), 32000))
    await asyncio.sleep(0)
    # _sound_until = now + backlog_s + len(pcm)/rate = ~2.0 + 1.0
    assert sink._sound_until - before == pytest.approx(3.0, abs=0.2)
    sink.stop()
    await asyncio.wait_for(task, timeout=2.0)


async def test_play_returns_at_playout_not_after_margin(paced_sink):
    # both bounds matter: an early return risks audible truncation, a late
    # one is the old worst-case sleep this change removes (was duration+0.9s)
    sink, _ = paced_sink()
    sink._keepalive_task.cancel()
    loop = asyncio.get_running_loop()
    start = loop.time()
    await asyncio.wait_for(sink.play(_pcm(0.8), 16000), timeout=5.0)
    elapsed = loop.time() - start
    assert 0.8 <= elapsed <= 0.8 + SINK_RESIDUAL_S + 0.15


async def test_back_to_back_plays_do_not_flush(paced_sink):
    # the TTS sentence loop plays sounds back to back; a play() returning
    # before physical playout would make the next one flush its tail
    sink, paced = paced_sink()
    sink._keepalive_task.cancel()
    await asyncio.wait_for(sink.play(_pcm(0.3), 16000), timeout=5.0)
    await asyncio.wait_for(sink.play(_pcm(0.3), 16000), timeout=5.0)
    assert paced.flushes == 0


async def test_keepalive_refill_does_not_delay_completion(paced_sink):
    # dither pushed behind the utterance raises pushed and level equally,
    # so it must not move the playout watermark (old code: >= 1.4s)
    sink, _ = paced_sink()
    loop = asyncio.get_running_loop()
    start = loop.time()
    await asyncio.wait_for(sink.play(_pcm(0.5), 16000), timeout=5.0)
    elapsed = loop.time() - start
    assert elapsed < 0.5 + gst_sink.KEEPALIVE_LOW_WATER_S + SINK_RESIDUAL_S + 0.2


async def test_playout_failsafe_deadline(make_sink, monkeypatch, caplog):
    # a level that never drains (dead device, no bus error) must not hang
    sink = make_sink()
    sink._keepalive_task.cancel()
    monkeypatch.setattr(sink, "_queued_bytes", lambda: 3200)
    monkeypatch.setattr(gst_sink, "PLAYOUT_FAILSAFE_EXTRA_S", 0.3)
    loop = asyncio.get_running_loop()
    start = loop.time()
    with caplog.at_level("WARNING"):
        await asyncio.wait_for(sink.play(_pcm(0.2), 16000), timeout=5.0)
    elapsed = loop.time() - start
    # backlog 0.1s + duration 0.2s + failsafe 0.3s
    assert 0.4 <= elapsed <= 1.0
    assert "failsafe" in caplog.text


async def test_stop_during_residual_returns_immediately(paced_sink):
    sink, paced = paced_sink()
    sink._keepalive_task.cancel()
    task = asyncio.create_task(sink.play(_pcm(0.2), 16000))
    while paced.level() > 0:
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.05)  # inside the SINK_RESIDUAL_S wait now
    start = time.monotonic()
    sink.stop()
    await asyncio.wait_for(task, timeout=1.0)
    assert time.monotonic() - start < 0.2


async def test_broken_pipeline_raises_on_play(make_sink, monkeypatch):
    monkeypatch.setattr(
        PipewireSink,
        "_describe",
        staticmethod(
            lambda target: "appsrc name=src format=time is-live=true ! volume name=vol "
            "! filesink location=/nonexistent-dir/out.raw"
        ),
    )
    sink = None
    try:
        sink = PipewireSink(device=None)
        with pytest.raises(RuntimeError, match="playback pipeline error"):
            await sink.play(_pcm(0.1), 16000)
    except RuntimeError as exc:
        # constructor may already detect the broken pipeline on some stacks
        assert "playback pipeline failed to start" in str(exc)
    finally:
        if sink is not None:
            sink.close()
