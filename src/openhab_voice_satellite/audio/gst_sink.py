"""Cancellable PCM playback via GStreamer appsrc -> pipewiresink, with ducking."""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .gst_common import gst_init, install_sync_handler, s16_mono_caps
from .gst_devices import resolve_node

log = logging.getLogger(__name__)

PUSH_CHUNK_MS = 100


def _set_threadsafe(loop: asyncio.AbstractEventLoop, event: asyncio.Event) -> None:
    try:
        loop.call_soon_threadsafe(event.set)
    except RuntimeError:
        pass  # loop already closed during shutdown


@dataclass
class _ActivePlayback:
    pipeline: Any
    volume: Any
    done: asyncio.Event
    loop: asyncio.AbstractEventLoop
    errors: list[str] = field(default_factory=list)


class PipewireSink:
    """Plays mono int16 buffers through a per-utterance appsrc pipeline.

    `duck()` scales output via the volume element (barge-in confirmation phase).
    `stop()` aborts playback immediately from any thread.
    """

    def __init__(self, device: str | None = None, lead_in_ms: int = 300) -> None:
        self._Gst = gst_init()
        self._target = resolve_node(device, "output")
        self._lead_in_ms = lead_in_ms
        self._gain = 1.0
        self._lock = threading.Lock()
        self._active: _ActivePlayback | None = None

    @staticmethod
    def _describe(target: str | None) -> str:
        # max-bytes=0: the default ~200 kB queue would throttle long utterances
        t = f'target-object="{target}" ' if target else ""
        return (
            "appsrc name=src format=time max-bytes=0 block=false "
            "! audioconvert ! audioresample ! volume name=vol "
            f"! pipewiresink client-name=openhab-voice-satellite {t}"
        )

    def duck(self, factor: float) -> None:
        self._gain = factor
        with self._lock:
            if self._active is not None:
                self._active.volume.set_property("volume", factor)

    def unduck(self) -> None:
        self.duck(1.0)

    def stop(self) -> None:
        with self._lock:
            active = self._active
        if active is not None:
            # NULL suppresses the EOS message, so release the waiter explicitly
            active.pipeline.set_state(self._Gst.State.NULL)
            _set_threadsafe(active.loop, active.done)

    async def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        Gst = self._Gst
        if self._lead_in_ms > 0:
            # USB DACs mute the first ~hundreds of ms after stream start
            # (anti-pop); lead-in silence keeps speech out of that window
            pad = np.zeros(sample_rate * self._lead_in_ms // 1000, dtype=pcm.dtype)
            pcm = np.concatenate([pad, pcm])

        pipeline = Gst.parse_launch(self._describe(self._target))
        src = pipeline.get_by_name("src")
        src.set_property("caps", Gst.Caps.from_string(s16_mono_caps(sample_rate)))
        volume = pipeline.get_by_name("vol")
        volume.set_property("volume", self._gain)

        active = _ActivePlayback(
            pipeline=pipeline,
            volume=volume,
            done=asyncio.Event(),
            loop=asyncio.get_running_loop(),
        )

        def on_error(err, debug) -> None:
            active.errors.append(f"{err.message} ({debug})")
            _set_threadsafe(active.loop, active.done)

        def on_eos() -> None:
            _set_threadsafe(active.loop, active.done)

        install_sync_handler(Gst, pipeline.get_bus(), on_error, on_eos)

        chunk_samples = sample_rate * PUSH_CHUNK_MS // 1000
        for start in range(0, len(pcm), chunk_samples):
            chunk = pcm[start:start + chunk_samples]
            buffer = Gst.Buffer.new_wrapped(chunk.tobytes())
            buffer.pts = start * Gst.SECOND // sample_rate
            buffer.duration = len(chunk) * Gst.SECOND // sample_rate
            src.emit("push-buffer", buffer)
        src.emit("end-of-stream")

        with self._lock:
            self._active = active
        try:
            if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("playback pipeline failed to start (is PipeWire running?)")
            await active.done.wait()
            if active.errors:
                raise RuntimeError(f"playback pipeline error: {active.errors[0]}")
        finally:
            pipeline.set_state(Gst.State.NULL)
            with self._lock:
                self._active = None
