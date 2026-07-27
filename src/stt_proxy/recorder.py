"""LISTENING stage: collect PCM from a frame queue until VAD endpoint or timeout."""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from .config import VadConfig
from .vad import SpeechEndpointer

log = logging.getLogger(__name__)


class NoSpeechError(Exception):
    """User said nothing within the no-speech timeout."""


async def record_utterance(
    frames: asyncio.Queue[np.ndarray | None],
    endpointer: SpeechEndpointer,
    config: VadConfig,
) -> np.ndarray:
    """Drain `frames` until the endpointer signals end of utterance.

    Returns the full utterance as int16 PCM. Raises NoSpeechError when no
    speech starts within `no_speech_timeout_s`, and returns whatever was
    collected when `max_utterance_s` is hit.
    """
    endpointer.reset()
    collected: list[np.ndarray] = []
    while True:
        frame = await frames.get()
        if frame is None:
            raise NoSpeechError("audio source closed")
        collected.append(frame)
        endpointer.update(frame)
        if not endpointer.speech_started and endpointer.elapsed_s >= config.no_speech_timeout_s:
            raise NoSpeechError(f"no speech within {config.no_speech_timeout_s}s")
        if endpointer.endpoint_reached:
            break
        if endpointer.elapsed_s >= config.max_utterance_s:
            log.warning("max utterance length reached (%.1fs), cutting off", config.max_utterance_s)
            break
    return np.concatenate(collected)
