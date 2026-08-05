"""Microphone capture: PortAudio callback thread -> asyncio queue of PCM frames."""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Protocol

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)


class AudioSource(Protocol):
    async def frames(self) -> AsyncIterator[np.ndarray]:
        """Yield mono int16 frames of fixed length until the source closes."""
        ...

    def close(self) -> None: ...


def find_device(name_substring: str | None, kind: str) -> int | None:
    """Resolve a device index by case-insensitive name substring. None = default."""
    if name_substring is None:
        return None
    for idx, dev in enumerate(sd.query_devices()):
        channels = dev["max_input_channels"] if kind == "input" else dev["max_output_channels"]
        if channels > 0 and name_substring.lower() in dev["name"].lower():
            return idx
    raise ValueError(f"no {kind} device matching {name_substring!r}")


class SounddeviceSource:
    """16 kHz mono int16 capture in frames of `frame_samples`."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_samples: int = 1280,
        device: str | None = None,
        queue_size: int = 50,
    ) -> None:
        self._sample_rate = sample_rate
        self._frame_samples = frame_samples
        self._device = find_device(device, "input")
        self._queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=queue_size)
        self._loop = asyncio.get_running_loop()
        self._stream = sd.InputStream(
            samplerate=sample_rate,
            blocksize=frame_samples,
            device=self._device,
            channels=1,
            dtype="int16",
            callback=self._callback,
        )
        self._stream.start()
        log.info("audio input open: device=%s rate=%d frame=%d",
                 self._device if self._device is not None else "default",
                 sample_rate, frame_samples)

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            log.warning("input stream status: %s", status)
        frame = indata[:, 0].copy()

        def _put() -> None:
            if self._queue.full():
                # drop oldest so live audio keeps flowing under backpressure
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            self._queue.put_nowait(frame)

        self._loop.call_soon_threadsafe(_put)

    async def frames(self) -> AsyncIterator[np.ndarray]:
        while True:
            frame = await self._queue.get()
            if frame is None:
                return
            yield frame

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()
        self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
