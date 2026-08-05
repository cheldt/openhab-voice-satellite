import asyncio
import time

import numpy as np
import pytest

gi = pytest.importorskip("gi")

from openhab_voice_satellite.audio.gst_sink import PipewireSink  # noqa: E402

FAKESINK = (
    "appsrc name=src format=time max-bytes=0 block=false "
    "! audioconvert ! audioresample ! volume name=vol ! fakesink sync=%s"
)


def _fake_describe(sync):
    return staticmethod(lambda target: FAKESINK % sync)


def _pcm(seconds: float, rate: int = 16000) -> np.ndarray:
    return np.zeros(int(seconds * rate), dtype=np.int16)


async def test_play_completes_on_eos(monkeypatch):
    monkeypatch.setattr(PipewireSink, "_describe", _fake_describe("false"))
    sink = PipewireSink(device=None, lead_in_ms=0)
    await asyncio.wait_for(sink.play(_pcm(1.0), 16000), timeout=5.0)


async def test_stop_aborts_playback_quickly(monkeypatch):
    monkeypatch.setattr(PipewireSink, "_describe", _fake_describe("true"))
    sink = PipewireSink(device=None, lead_in_ms=0)
    task = asyncio.create_task(sink.play(_pcm(5.0), 16000))
    await asyncio.sleep(0.3)
    start = time.monotonic()
    sink.stop()
    await asyncio.wait_for(task, timeout=2.0)
    assert time.monotonic() - start < 1.0


async def test_duck_applies_to_active_pipeline_and_persists(monkeypatch):
    monkeypatch.setattr(PipewireSink, "_describe", _fake_describe("true"))
    sink = PipewireSink(device=None, lead_in_ms=0)
    sink.duck(0.4)
    task = asyncio.create_task(sink.play(_pcm(2.0), 16000))
    await asyncio.sleep(0.3)
    assert sink._active is not None
    assert sink._active.volume.get_property("volume") == pytest.approx(0.4)
    sink.unduck()
    assert sink._active.volume.get_property("volume") == pytest.approx(1.0)
    sink.stop()
    await asyncio.wait_for(task, timeout=2.0)


async def test_unstartable_pipeline_raises(monkeypatch):
    monkeypatch.setattr(
        PipewireSink,
        "_describe",
        staticmethod(
            lambda target: "appsrc name=src format=time ! volume name=vol "
            "! filesink location=/nonexistent-dir/out.raw"
        ),
    )
    sink = PipewireSink(device=None, lead_in_ms=0)
    with pytest.raises(RuntimeError, match="playback"):
        await sink.play(_pcm(0.1), 16000)
