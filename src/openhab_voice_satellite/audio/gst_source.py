"""Microphone capture: GStreamer pipewiresrc -> asyncio queue of PCM frames."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import AsyncIterator

import numpy as np

from .chunker import FrameChunker
from .gst_common import capture_description, gst_init, install_sync_handler
from .gst_devices import resolve_node

log = logging.getLogger(__name__)

FIRST_FRAME_TIMEOUT_S = 5.0
DROP_WARN_INTERVAL_S = 5.0
NS = 1_000_000_000
# a gap smaller than this is jitter in the graph's timestamps, not lost audio
PTS_GAP_TOLERANCE_NS = NS // 1000  # 1 ms

_CLOCK_TIME_NONE = 2**64 - 1  # GST_CLOCK_TIME_NONE


@dataclass(frozen=True)
class CaptureStats:
    """What the graph handed us, as opposed to what a consumer received.

    Counted at the appsink, so a shortfall here is PipeWire under-feeding this
    client and a shortfall further downstream is ours. Cumulative since the
    source opened; callers difference two snapshots over a window.
    """

    buffers: int = 0
    samples: int = 0
    pts_gaps: int = 0  # timestamp discontinuities, i.e. audio the graph skipped
    pts_gap_s: float = 0.0
    dropped: int = 0  # frames evicted from the source queue (loop too slow)

    def since(self, earlier: "CaptureStats") -> "CaptureStats":
        return CaptureStats(
            buffers=self.buffers - earlier.buffers,
            samples=self.samples - earlier.samples,
            pts_gaps=self.pts_gaps - earlier.pts_gaps,
            pts_gap_s=self.pts_gap_s - earlier.pts_gap_s,
            dropped=self.dropped - earlier.dropped,
        )

    def describe(self) -> str:
        return (
            f"graph: {self.buffers} buffers / {self.samples} samples, "
            f"{self.pts_gaps} PTS gaps totalling {self.pts_gap_s:.1f}s; "
            f"source queue drops {self.dropped}"
        )


class _Counters:
    """Mutable half of CaptureStats; buffer fields are gst-thread-only."""

    def __init__(self, sample_rate: int) -> None:
        self._sample_rate = sample_rate
        self.buffers = 0
        self.samples = 0
        self.pts_gaps = 0
        self.pts_gap_ns = 0
        self.dropped = 0  # written on the event loop, unlike the rest
        self._next_pts: int | None = None

    def observe(self, pts: int | None, n_samples: int) -> None:
        """Account one delivered buffer and the gap in front of it."""
        self.buffers += 1
        self.samples += n_samples
        if pts is None or pts == _CLOCK_TIME_NONE:
            self._next_pts = None  # untimed buffer: no gap can be attributed
            return
        if self._next_pts is not None and pts - self._next_pts > PTS_GAP_TOLERANCE_NS:
            self.pts_gaps += 1
            self.pts_gap_ns += pts - self._next_pts
        self._next_pts = pts + n_samples * NS // self._sample_rate

    def snapshot(self) -> CaptureStats:
        return CaptureStats(
            buffers=self.buffers,
            samples=self.samples,
            pts_gaps=self.pts_gaps,
            pts_gap_s=self.pts_gap_ns / NS,
            dropped=self.dropped,
        )


class PipewireSource:
    """16 kHz mono int16 capture in frames of `frame_samples`."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_samples: int = 1280,
        device: str | None = None,
        queue_size: int = 50,
    ) -> None:
        Gst = gst_init()
        self._Gst = Gst
        target = resolve_node(device, "input")
        self.target = target  # resolved node name; None = default source
        self._queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=queue_size)
        self._loop = asyncio.get_running_loop()
        self._chunker = FrameChunker(frame_samples)
        self._got_frame = False
        self._caps_logged = False
        self._counters = _Counters(sample_rate)
        self._last_drop_warn = float("-inf")
        self._pipeline = Gst.parse_launch(self._describe(target, sample_rate))
        self._pipeline.get_by_name("sink").connect("new-sample", self._on_sample)
        install_sync_handler(Gst, self._pipeline.get_bus(), self._on_error, self._on_eos)
        if self._pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            self._pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("capture pipeline failed to start (is PipeWire running?)")
        self._loop.call_later(FIRST_FRAME_TIMEOUT_S, self._warn_if_stalled, target)
        log.info("audio input open: node=%s rate=%d frame=%d",
                 target or "default", sample_rate, frame_samples)

    @staticmethod
    def _describe(target: str | None, sample_rate: int) -> str:
        return capture_description(target, sample_rate)

    def _on_sample(self, appsink):
        # runs on the GStreamer streaming thread
        sample = appsink.emit("pull-sample")
        if sample is None:
            return self._Gst.FlowReturn.OK
        if not self._caps_logged:
            self._caps_logged = True
            caps = sample.get_caps()
            log.info("capture caps negotiated: %s", caps.to_string() if caps else "?")
        buffer = sample.get_buffer()
        ok, mapinfo = buffer.map(self._Gst.MapFlags.READ)
        if not ok:
            log.warning("capture buffer could not be mapped — audio lost")
            return self._Gst.FlowReturn.OK
        try:
            pcm = np.frombuffer(mapinfo.data, dtype=np.int16)
            # account for what the graph delivered before the chunker reshapes
            # it: a shortfall here is upstream, one further down is ours
            self._counters.observe(buffer.pts, len(pcm))
            frames = self._chunker.push(pcm)
        finally:
            buffer.unmap(mapinfo)
        if frames:
            try:
                self._loop.call_soon_threadsafe(self._put_frames, frames)
            except RuntimeError:
                pass  # loop already closed during shutdown
        return self._Gst.FlowReturn.OK

    def _put_frames(self, frames: list[np.ndarray]) -> None:
        # runs on the event loop (via call_soon_threadsafe), so logging is safe
        self._got_frame = True
        for frame in frames:
            if self._queue.full():
                # drop oldest so live audio keeps flowing under backpressure;
                # only happens when the event loop stalled for ~4s
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                else:
                    self._counters.dropped += 1
                    now = self._loop.time()
                    if now - self._last_drop_warn >= DROP_WARN_INTERVAL_S:
                        self._last_drop_warn = now
                        log.warning(
                            "mic capture queue overflow: %d frames dropped so far",
                            self._counters.dropped,
                        )
            self._queue.put_nowait(frame)

    def _warn_if_stalled(self, target: str | None) -> None:
        if not self._got_frame:
            log.warning(
                "no audio from %s after %.0fs — is the node linkable? (check wpctl status)",
                target or "default source", FIRST_FRAME_TIMEOUT_S,
            )

    def _on_error(self, err, debug) -> None:
        log.error("capture pipeline error: %s (%s)", err.message, debug)
        self._end_stream()

    def _on_eos(self) -> None:
        log.warning("capture pipeline reached end of stream")
        self._end_stream()

    def _end_stream(self) -> None:
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
        except RuntimeError:
            pass

    def stats(self) -> CaptureStats:
        """Cumulative capture accounting; difference two for a window."""
        return self._counters.snapshot()

    async def frames(self) -> AsyncIterator[np.ndarray]:
        while True:
            frame = await self._queue.get()
            if frame is None:
                return
            yield frame

    def close(self) -> None:
        self._pipeline.set_state(self._Gst.State.NULL)
        self._end_stream()
