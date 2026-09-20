import asyncio

import numpy as np
import pytest


def test_the_end_of_stream_sentinel_survives_a_full_queue():
    """The sentinel is the one put that must never be lost.

    Every frame path drops the oldest entry under backpressure, so a bare
    put_nowait(None) behind a queue those paths just filled raised QueueFull
    into the loop's callback handler, where it was swallowed with nothing to
    retry it. The broadcaster then blocks in get() forever, no subscriber sees
    None, and app.py's CaptureClosedError exit never fires: the process lives
    on, permanently deaf. Needs no GStreamer graph — only the queue.
    """
    from openhab_voice_satellite.audio.gst_source import PipewireSource

    source = object.__new__(PipewireSource)
    source._queue = asyncio.Queue(maxsize=2)
    source._dropped = 0
    for _ in range(2):
        source._queue.put_nowait(np.zeros(1280, dtype=np.int16))
    assert source._queue.full()

    source._put_end()

    drained = [source._queue.get_nowait() for _ in range(source._queue.qsize())]
    assert drained[-1] is None  # delivered, at the cost of the oldest frame
    assert source._dropped == 1


gi = pytest.importorskip("gi")

from openhab_voice_satellite.audio.gst_common import s16_mono_caps  # noqa: E402
from openhab_voice_satellite.audio.gst_source import PipewireSource  # noqa: E402


def _testsrc_describe(target, sample_rate):
    # real pipeline, fake hardware: audiotestsrc instead of pipewiresrc
    return (
        f"audiotestsrc is-live=true samplesperbuffer=800 ! audioconvert ! audioresample "
        f"! {s16_mono_caps(sample_rate)} "
        f"! appsink name=sink emit-signals=true sync=false max-buffers=8 drop=true"
    )


async def test_yields_fixed_size_int16_frames(monkeypatch):
    monkeypatch.setattr(PipewireSource, "_describe", staticmethod(_testsrc_describe))
    source = PipewireSource(sample_rate=16000, frame_samples=1280, device=None)
    try:
        it = source.frames()
        for _ in range(3):
            frame = await asyncio.wait_for(anext(it), timeout=5.0)
            assert frame.dtype == np.int16
            assert len(frame) == 1280
    finally:
        source.close()


async def test_close_ends_frame_iteration(monkeypatch):
    monkeypatch.setattr(PipewireSource, "_describe", staticmethod(_testsrc_describe))
    source = PipewireSource(sample_rate=16000, frame_samples=1280, device=None)
    it = source.frames()
    await asyncio.wait_for(anext(it), timeout=5.0)
    source.close()

    async def drain():
        async for _ in it:
            pass

    await asyncio.wait_for(drain(), timeout=5.0)


async def test_unstartable_pipeline_raises(monkeypatch):
    def bad_describe(target, sample_rate):
        return (
            "filesrc location=/nonexistent/audio.raw ! "
            f"{s16_mono_caps(sample_rate)} ! appsink name=sink emit-signals=true"
        )

    monkeypatch.setattr(PipewireSource, "_describe", staticmethod(bad_describe))
    with pytest.raises(RuntimeError, match="capture"):
        PipewireSource(sample_rate=16000, frame_samples=1280, device=None)
