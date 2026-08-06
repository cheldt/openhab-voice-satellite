import asyncio
import time

import numpy as np
import pytest

gi = pytest.importorskip("gi")

from openhab_voice_satellite.audio.gst_sink import PipewireSink  # noqa: E402

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


async def test_play_completes(make_sink):
    sink = make_sink()
    await asyncio.wait_for(sink.play(_pcm(0.5), 16000), timeout=5.0)


async def test_pipeline_survives_across_plays(make_sink):
    sink = make_sink()
    pipeline = sink._pipeline
    await asyncio.wait_for(sink.play(_pcm(0.2), 16000), timeout=5.0)
    await asyncio.wait_for(sink.play(_pcm(0.2), 22050), timeout=5.0)
    assert sink._pipeline is pipeline


async def test_stop_aborts_playback_quickly(make_sink):
    sink = make_sink(sync="true")
    task = asyncio.create_task(sink.play(_pcm(5.0), 16000))
    await asyncio.sleep(0.3)
    start = time.monotonic()
    sink.stop()
    await asyncio.wait_for(task, timeout=2.0)
    assert time.monotonic() - start < 1.0


async def test_new_play_replaces_queued_tail(make_sink):
    sink = make_sink(sync="true")
    long_play = asyncio.create_task(sink.play(_pcm(5.0), 16000))
    await asyncio.sleep(0.3)
    # a second play must not wait for the first one's 5s to elapse
    await asyncio.wait_for(sink.play(_pcm(0.2), 16000), timeout=2.0)
    sink.stop()
    await asyncio.wait_for(long_play, timeout=2.0)


async def test_duck_applies_to_pipeline_and_persists(make_sink):
    sink = make_sink(sync="true")
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
