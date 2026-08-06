"""Shared open/close lifecycle for the capture+playback pair (app and probe)."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncIterator

if TYPE_CHECKING:
    from ..config import AudioConfig
    from .gst_sink import PipewireSink
    from .gst_source import PipewireSource

log = logging.getLogger(__name__)


@asynccontextmanager
async def audio_io(audio: AudioConfig) -> AsyncIterator[tuple[PipewireSource, PipewireSink]]:
    """Open the capture source and playback sink; close both on exit.

    Imports stay lazy so `--check`/`--probe-mic` error paths can report a
    missing GStreamer stack instead of failing at import time.
    """
    from .gst_sink import PipewireSink
    from .gst_source import PipewireSource

    source = PipewireSource(
        sample_rate=audio.sample_rate,
        frame_samples=audio.frame_samples,
        device=audio.input_device,
    )
    try:
        # sink construction can raise after the source is already PLAYING
        sink = PipewireSink(
            device=audio.output_device,
            wakeup_preamble_ms=audio.wakeup_preamble_ms,
            wakeup_preamble_idle_s=audio.wakeup_preamble_idle_s,
        )
    except BaseException:
        source.close()
        raise
    try:
        yield source, sink
    finally:
        sink.close()
        source.close()


async def verify_links(input_target: str | None, output_target: str | None) -> None:
    """Check (after WirePlumber settles) that both streams actually linked."""
    from .gst_devices import verify_stream_links

    await asyncio.sleep(3.0)
    await verify_stream_links("openhab-voice-satellite", input_target, output_target)
