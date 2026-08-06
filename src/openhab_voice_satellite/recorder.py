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
        remaining = deadline - loop.time()
        try:
            frame = await asyncio.wait_for(
                frames.get(), timeout=max(0.0, min(MIC_STALL_TIMEOUT_S, remaining))
            )
        except asyncio.TimeoutError:
            if endpointer.speech_started and collected:
                log.warning(
                    "mic stalled mid-utterance, keeping %.1fs already collected",
                    endpointer.elapsed_s,
                )
                break
            raise NoSpeechError(
                f"mic stalled: no frames within {MIC_STALL_TIMEOUT_S:.0f}s"
            ) from None
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
