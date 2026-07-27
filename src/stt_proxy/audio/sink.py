"""Cancellable PCM playback with volume ducking."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Protocol

import numpy as np
import sounddevice as sd

from .source import find_device

log = logging.getLogger(__name__)


class AudioSink(Protocol):
    async def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        """Play mono int16 PCM to completion; must be cancellable."""
        ...

    def stop(self) -> None: ...

    def duck(self, factor: float) -> None: ...

    def unduck(self) -> None: ...


class SounddeviceSink:
    """Plays mono int16 buffers through an OutputStream opened per utterance.

    `duck()` scales sample amplitude on the fly (barge-in confirmation phase).
    `stop()` aborts playback immediately from any thread.
    """

    def __init__(self, device: str | None = None) -> None:
        self._device = find_device(device, "output")
        self._gain = 1.0
        self._stop_flag = threading.Event()
        self._active_stream: sd.OutputStream | None = None

    def duck(self, factor: float) -> None:
        self._gain = factor

    def unduck(self) -> None:
        self._gain = 1.0

    def stop(self) -> None:
        self._stop_flag.set()
        stream = self._active_stream
        if stream is not None:
            try:
                stream.abort()
            except Exception:
                pass

    async def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        self._stop_flag.clear()
        done = asyncio.Event()
        loop = asyncio.get_running_loop()
        pos = 0

        def callback(outdata: np.ndarray, frames: int, time_info, status) -> None:
            nonlocal pos
            if status:
                log.warning("output stream status: %s", status)
            chunk = pcm[pos:pos + frames]
            pos += frames
            if self._gain != 1.0:
                chunk = (chunk.astype(np.float32) * self._gain).astype(np.int16)
            outdata[: len(chunk), 0] = chunk
            outdata[len(chunk):, 0] = 0
            if pos >= len(pcm) or self._stop_flag.is_set():
                raise sd.CallbackStop

        def finished() -> None:
            loop.call_soon_threadsafe(done.set)

        stream = sd.OutputStream(
            samplerate=sample_rate,
            device=self._device,
            channels=1,
            dtype="int16",
            callback=callback,
            finished_callback=finished,
        )
        self._active_stream = stream
        try:
            with stream:
                await done.wait()
        finally:
            self._active_stream = None
