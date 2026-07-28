"""faster-whisper transcription with de/en language restriction."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import numpy as np

from .config import SttConfig

log = logging.getLogger(__name__)


@dataclass
class Transcript:
    text: str
    language: str


class Transcriber:
    def __init__(self, config: SttConfig, default_language: str) -> None:
        from faster_whisper import WhisperModel

        self._config = config
        self._default_language = default_language
        self._model = WhisperModel(
            config.model,
            device="cpu",
            compute_type=config.compute_type,
            cpu_threads=config.cpu_threads,
        )
        log.info("whisper model %s loaded (%s)", config.model, config.compute_type)

    def _transcribe_sync(self, pcm: np.ndarray) -> Transcript:
        audio = pcm.astype(np.float32) / 32768.0
        # single configured language: skip whisper's language-detection pass
        languages = self._config.languages
        fixed_language = languages[0] if len(languages) == 1 else None
        segments, info = self._model.transcribe(
            audio,
            language=fixed_language,
            beam_size=self._config.beam_size,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        language = info.language
        if language not in self._config.languages:
            # restrict auto-detect to the configured languages
            probs = info.all_language_probs or []
            allowed = [(lang, p) for lang, p in probs if lang in self._config.languages]
            language = max(allowed, key=lambda lp: lp[1])[0] if allowed else self._default_language
            log.info("detected language %s not configured, using %s", info.language, language)
        return Transcript(text=text, language=language)

    async def transcribe(self, pcm: np.ndarray) -> Transcript:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, pcm)
