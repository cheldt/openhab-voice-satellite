import asyncio

import numpy as np
import pytest

# the accounting is pure python; only the pipeline tests below need gi
from openhab_voice_satellite.audio.gst_source import NS, CaptureStats, _Counters


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
    source._counters = _Counters(RATE)
    for _ in range(2):
        source._queue.put_nowait(np.zeros(1280, dtype=np.int16))
    assert source._queue.full()

    source._put_end()

    drained = [source._queue.get_nowait() for _ in range(source._queue.qsize())]
    assert drained[-1] is None  # delivered, at the cost of the oldest frame
    assert source._counters.dropped == 1


gi = pytest.importorskip("gi")

from openhab_voice_satellite.audio.gst_common import s16_mono_caps  # noqa: E402
from openhab_voice_satellite.audio.gst_source import PipewireSource  # noqa: E402

RATE = 16000


def _testsrc_describe(target, sample_rate):
    # real pipeline, fake hardware: audiotestsrc instead of pipewiresrc
    return (
        f"audiotestsrc is-live=true samplesperbuffer=800 ! audioconvert ! audioresample "
        f"! {s16_mono_caps(sample_rate)} "
        f"! appsink name=sink emit-signals=true sync=false max-buffers=0 drop=false"
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


async def test_capture_can_be_held_out_of_the_graph_until_start(monkeypatch):
    """A stream nobody services xruns its way out of PipeWire's scheduling."""
    monkeypatch.setattr(PipewireSource, "_describe", staticmethod(_testsrc_describe))
    source = PipewireSource(
        sample_rate=RATE, frame_samples=1280, device=None, autostart=False
    )
    try:
        await asyncio.sleep(0.3)
        assert source.stats().buffers == 0  # nothing captured before start()
        source.start()
        source.start()  # idempotent: a second call must not restart anything
        frame = await asyncio.wait_for(anext(source.frames()), timeout=5.0)
        assert len(frame) == 1280
    finally:
        source.close()


async def test_stats_count_what_the_graph_delivered(monkeypatch):
    monkeypatch.setattr(PipewireSource, "_describe", staticmethod(_testsrc_describe))
    source = PipewireSource(sample_rate=RATE, frame_samples=1280, device=None)
    try:
        it = source.frames()
        for _ in range(3):
            await asyncio.wait_for(anext(it), timeout=5.0)
        stats = source.stats()
        assert stats.buffers > 0
        assert stats.samples >= 3 * 1280  # at least what we already consumed
        assert stats.dropped == 0
    finally:
        source.close()


# -- accounting, without a pipeline ------------------------------------


def _push(counters, pts_s, n=800):
    counters.observe(int(pts_s * NS), n)


def test_contiguous_buffers_are_not_a_gap():
    counters = _Counters(RATE)
    for i in range(5):
        _push(counters, i * 800 / RATE)
    stats = counters.snapshot()
    assert (stats.buffers, stats.samples) == (5, 4000)
    assert stats.pts_gaps == 0


def test_a_skipped_buffer_is_counted_as_a_gap():
    """The one fact that separates 'the graph skipped it' from 'we lost it'."""
    counters = _Counters(RATE)
    _push(counters, 0.0)
    _push(counters, 0.05 + 800 / RATE)  # 50 ms of audio never arrived
    stats = counters.snapshot()
    assert stats.pts_gaps == 1
    assert stats.pts_gap_s == pytest.approx(0.05)
    assert stats.buffers == 2  # the gap is not a buffer we received


def test_timestamp_jitter_is_not_reported_as_loss():
    counters = _Counters(RATE)
    _push(counters, 0.0)
    _push(counters, 800 / RATE + 0.0005)  # 0.5 ms late, under the tolerance
    assert counters.snapshot().pts_gaps == 0


def test_an_untimed_buffer_cannot_manufacture_a_gap():
    counters = _Counters(RATE)
    _push(counters, 0.0)
    counters.observe(None, 800)  # GST_CLOCK_TIME_NONE in practice
    _push(counters, 99.0)  # a wild jump, but nothing to measure it against
    stats = counters.snapshot()
    assert stats.pts_gaps == 0
    assert stats.buffers == 3


def test_a_window_is_the_difference_of_two_snapshots():
    early = CaptureStats(buffers=10, samples=8000, pts_gaps=1, pts_gap_s=0.5, dropped=2)
    later = CaptureStats(buffers=25, samples=20000, pts_gaps=4, pts_gap_s=2.0, dropped=3)
    window = later.since(early)
    assert (window.buffers, window.samples) == (15, 12000)
    assert (window.pts_gaps, window.dropped) == (3, 1)
    assert window.pts_gap_s == pytest.approx(1.5)
    assert "15 buffers / 12000 samples" in window.describe()


async def test_unstartable_pipeline_raises(monkeypatch):
    def bad_describe(target, sample_rate):
        return (
            "filesrc location=/nonexistent/audio.raw ! "
            f"{s16_mono_caps(sample_rate)} ! appsink name=sink emit-signals=true"
        )

    monkeypatch.setattr(PipewireSource, "_describe", staticmethod(bad_describe))
    with pytest.raises(RuntimeError, match="capture"):
        PipewireSource(sample_rate=16000, frame_samples=1280, device=None)
