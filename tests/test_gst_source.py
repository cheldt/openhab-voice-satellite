import asyncio

import numpy as np
import pytest

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
