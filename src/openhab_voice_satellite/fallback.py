"""Local-fallback wrappers shared by all cloud STT/TTS providers."""

from __future__ import annotations

import json
import logging

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
