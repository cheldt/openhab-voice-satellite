"""Microphone capture: GStreamer pipewiresrc -> asyncio queue of PCM frames."""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

import numpy as np

from .chunker import FrameChunker
from .gst_common import capture_description, gst_init, install_sync_handler
from .gst_devices import resolve_node

log = logging.getLogger(__name__)

FIRST_FRAME_TIMEOUT_S = 5.0


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
            return self._Gst.FlowReturn.OK
        try:
            frames = self._chunker.push(np.frombuffer(mapinfo.data, dtype=np.int16))
        finally:
            buffer.unmap(mapinfo)
        if frames:
            try:
                self._loop.call_soon_threadsafe(self._put_frames, frames)
            except RuntimeError:
                pass  # loop already closed during shutdown
        return self._Gst.FlowReturn.OK

    def _put_frames(self, frames: list[np.ndarray]) -> None:
        self._got_frame = True
        for frame in frames:
            if self._queue.full():
                # drop oldest so live audio keeps flowing under backpressure
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
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

    async def frames(self) -> AsyncIterator[np.ndarray]:
        while True:
            frame = await self._queue.get()
            if frame is None:
                return
            yield frame

    def close(self) -> None:
        self._pipeline.set_state(self._Gst.State.NULL)
        self._end_stream()
