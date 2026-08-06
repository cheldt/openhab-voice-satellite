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


def _fake_describe(sync):
    return staticmethod(lambda target: FAKESINK % sync)


def _pcm(seconds: float, rate: int = 16000) -> np.ndarray:
    return np.zeros(int(seconds * rate), dtype=np.int16)


async def test_play_completes(monkeypatch):
    monkeypatch.setattr(PipewireSink, "_describe", _fake_describe("false"))
    sink = PipewireSink(device=None)
    await asyncio.wait_for(sink.play(_pcm(0.5), 16000), timeout=5.0)
    sink.close()


async def test_pipeline_survives_across_plays(monkeypatch):
    monkeypatch.setattr(PipewireSink, "_describe", _fake_describe("false"))
    sink = PipewireSink(device=None)
    pipeline = sink._pipeline
    await asyncio.wait_for(sink.play(_pcm(0.2), 16000), timeout=5.0)
    await asyncio.wait_for(sink.play(_pcm(0.2), 22050), timeout=5.0)
    assert sink._pipeline is pipeline
    sink.close()


async def test_stop_aborts_playback_quickly(monkeypatch):
    monkeypatch.setattr(PipewireSink, "_describe", _fake_describe("true"))
    sink = PipewireSink(device=None)
    task = asyncio.create_task(sink.play(_pcm(5.0), 16000))
    await asyncio.sleep(0.3)
    start = time.monotonic()
    sink.stop()
    await asyncio.wait_for(task, timeout=2.0)
    assert time.monotonic() - start < 1.0
    sink.close()


async def test_new_play_replaces_queued_tail(monkeypatch):
    monkeypatch.setattr(PipewireSink, "_describe", _fake_describe("true"))
    sink = PipewireSink(device=None)
    long_play = asyncio.create_task(sink.play(_pcm(5.0), 16000))
    await asyncio.sleep(0.3)
    # a second play must not wait for the first one's 5s to elapse
    await asyncio.wait_for(sink.play(_pcm(0.2), 16000), timeout=2.0)
    sink.stop()
    await asyncio.wait_for(long_play, timeout=2.0)
    sink.close()


async def test_duck_applies_to_pipeline_and_persists(monkeypatch):
    monkeypatch.setattr(PipewireSink, "_describe", _fake_describe("true"))
    sink = PipewireSink(device=None)
    sink.duck(0.4)
    assert sink._volume.get_property("volume") == pytest.approx(0.4)
    task = asyncio.create_task(sink.play(_pcm(2.0), 16000))
    await asyncio.sleep(0.3)
    assert sink._volume.get_property("volume") == pytest.approx(0.4)
    sink.unduck()
    assert sink._volume.get_property("volume") == pytest.approx(1.0)
    sink.stop()
    await asyncio.wait_for(task, timeout=2.0)
    sink.close()


async def test_wakeup_preamble_on_first_and_idle_plays_only(monkeypatch):
    monkeypatch.setattr(PipewireSink, "_describe", _fake_describe("false"))
    sink = PipewireSink(device=None, wakeup_preamble_ms=400)
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
    sink.close()


async def test_broken_pipeline_raises_on_play(monkeypatch):
    monkeypatch.setattr(
        PipewireSink,
        "_describe",
        staticmethod(
            lambda target: "appsrc name=src format=time is-live=true ! volume name=vol "
            "! filesink location=/nonexistent-dir/out.raw"
        ),
    )
    with pytest.raises(RuntimeError, match="playback"):
        sink = PipewireSink(device=None)
        await sink.play(_pcm(0.1), 16000)
