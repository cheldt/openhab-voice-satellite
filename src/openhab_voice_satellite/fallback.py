"""Local-fallback wrappers shared by all cloud STT/TTS providers."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable

import aiohttp
import numpy as np

from .stt import Transcript

log = logging.getLogger(__name__)


class CloudEngineError(Exception):
    """HTTP error or malformed response from a cloud STT/TTS API."""


class PartialSpeechError(CloudEngineError):
    """Cloud TTS failed after some audio already played.

    Carries the unspoken remainder so the fallback engine can pick up where
    playback stopped instead of repeating the whole utterance.
    """

    def __init__(self, message: str, remaining: str) -> None:
        super().__init__(message)
        self.remaining = remaining


# Failures that mean "cloud unusable right now" — everything else propagates.
# TimeoutError must be consumed here: the pipeline treats a leaked TimeoutError
# as an openHAB timeout. CancelledError (barge-in) is not an Exception and
# passes through both wrappers untouched.
_FALLBACK_ERRORS = (CloudEngineError, aiohttp.ClientError, TimeoutError, json.JSONDecodeError)


class FallbackTranscriber:
    def __init__(self, primary, fallback, label: str = "cloud") -> None:
        self._primary = primary
        self._fallback = fallback
        self._label = label

    async def transcribe(self, pcm: np.ndarray) -> Transcript:
        try:
            return await self._primary.transcribe(pcm)
        except _FALLBACK_ERRORS as exc:
            log.warning("%s STT failed (%s), falling back to local", self._label, exc)
            return await self._fallback.transcribe(pcm)


class LazySpeaker:
    """Defers construction of a local speaker until first use.

    Used when a cloud TTS engine is primary: the local models then only load
    on the first fallback instead of costing RAM and startup time up front.
    Construction runs in the executor so the multi-second model load never
    blocks the event loop (and with it the wakeword monitor).
    """

    def __init__(self, factory: Callable[[], object]) -> None:
        self._factory = factory
        self._speaker: object | None = None
        self._lock = asyncio.Lock()

    async def speak(self, text: str, language: str) -> None:
        async with self._lock:
            if self._speaker is None:
                log.info("loading local TTS fallback on first use")
                loop = asyncio.get_running_loop()
                self._speaker = await loop.run_in_executor(None, self._factory)
        await self._speaker.speak(text, language)


class FallbackSpeaker:
    def __init__(self, primary, fallback, label: str = "cloud") -> None:
        self._primary = primary
        self._fallback = fallback
        self._label = label

    async def speak(self, text: str, language: str) -> None:
        try:
            # A plain _FALLBACK_ERRORS failure happens before any audio played
            # and the local engine re-speaks the whole text without repetition.
            # Chunking speakers raise PartialSpeechError once audio has played,
            # so only the unspoken remainder is handed to the fallback.
            await self._primary.speak(text, language)
        except PartialSpeechError as exc:
            log.warning(
                "%s TTS failed mid-utterance (%s), local speaks the rest",
                self._label, exc,
            )
            await self._fallback.speak(exc.remaining, language)
        except _FALLBACK_ERRORS as exc:
            log.warning("%s TTS failed (%s), falling back to local", self._label, exc)
            await self._fallback.speak(text, language)
