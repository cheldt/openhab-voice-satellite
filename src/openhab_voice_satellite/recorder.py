"""LISTENING stage: collect PCM from a frame queue until VAD endpoint or timeout."""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from .config import SAMPLE_RATE, VadConfig
from .vad import SpeechEndpointer

log = logging.getLogger(__name__)

# Wall-clock backstop for a mic that stops delivering frames without EOS:
# frames normally arrive every 80 ms, and app.MIC_STALL_WARN_S logs at the
# same 10 s mark. The endpointer's own timeouts count received samples and
# can never fire on a silent queue.
MIC_STALL_TIMEOUT_S = 10.0


class NoSpeechError(Exception):
    """User said nothing within the no-speech timeout."""


async def _next_frame(
    frames: asyncio.Queue[np.ndarray | None], budget: float
) -> np.ndarray | None:
    """One frame within `budget` seconds; TimeoutError/QueueEmpty on none."""
    if budget <= 0:
        # window exhausted: consume what already arrived, never wait
        return frames.get_nowait()
    # asyncio.timeout, not wait_for: 3.11's wait_for swallows an external
    # cancel that races a completed get() (gh-86296), which would eat a
    # barge-in during LISTENING
    async with asyncio.timeout(budget):
        return await frames.get()


async def record_utterance(
    frames: asyncio.Queue[np.ndarray | None],
    endpointer: SpeechEndpointer,
    config: VadConfig,
    *,
    no_speech_timeout_s: float | None = None,
) -> np.ndarray:
    """Drain `frames` until the endpointer signals end of utterance.

    Returns the full utterance as int16 PCM. Raises NoSpeechError when no
    speech starts within the no-speech timeout (`no_speech_timeout_s`
    overrides `config.no_speech_timeout_s`), and returns whatever was
    collected when `max_utterance_s` is hit.
    """
    timeout_s = config.no_speech_timeout_s if no_speech_timeout_s is None else no_speech_timeout_s
    endpointer.reset()
    collected: list[np.ndarray] = []
    loop = asyncio.get_running_loop()
    # overall bound also catches a trickling mic whose rare frames keep
    # resetting the per-get cap while the sample clock barely advances
    deadline = loop.time() + timeout_s + config.max_utterance_s + MIC_STALL_TIMEOUT_S
    while True:
        budget = min(MIC_STALL_TIMEOUT_S, deadline - loop.time())
        try:
            frame = await _next_frame(frames, budget)
        except (TimeoutError, asyncio.QueueEmpty):
            reason = (
                f"mic stalled: no frames within {MIC_STALL_TIMEOUT_S:.0f}s"
                if budget > 0 else "listening window exhausted"
            )
            if endpointer.speech_started and collected:
                log.warning(
                    "%s mid-utterance, keeping %.1fs already collected",
                    reason, endpointer.elapsed_s,
                )
                break
            raise NoSpeechError(reason) from None
        if frame is None:
            raise NoSpeechError("audio source closed")
        collected.append(frame)
        endpointer.update(frame)
        if not endpointer.speech_started and endpointer.elapsed_s >= timeout_s:
            raise NoSpeechError(f"no speech within {timeout_s}s")
        if endpointer.endpoint_reached:
            break
        if endpointer.elapsed_s >= config.max_utterance_s:
            log.warning("max utterance length reached (%.1fs), cutting off", config.max_utterance_s)
            break
    # dropped frames were spliced out invisibly; the queue is fresh per
    # utterance (subscribed per round in pipeline.py), so the counter is ours
    dropped = getattr(frames, "dropped", 0)
    if dropped:
        lost_s = dropped * len(collected[0]) / SAMPLE_RATE
        log.warning(
            "utterance had %d dropped mic frames (~%.1fs lost, audio spliced)",
            dropped, lost_s,
        )
    return np.concatenate(collected)
