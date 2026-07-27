"""Piper TTS: per-language voices, sentence-split streaming synthesis."""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

import numpy as np

from .audio.sink import AudioSink
from .config import TtsConfig

log = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?:;])\s+")


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT.split(text.strip()) if s]


class Speaker:
    def __init__(self, config: TtsConfig, sink: AudioSink, base_dir: Path | None = None) -> None:
        from piper import PiperVoice

        self._sink = sink
        self._config = config
        base = base_dir or Path.cwd()
        self._voices = {}
        for lang, model_path in config.voices.items():
            path = Path(model_path) if Path(model_path).is_absolute() else base / model_path
            self._voices[lang] = PiperVoice.load(str(path))
            log.info("piper voice loaded: %s -> %s", lang, path.name)

    def _synthesize_sync(self, voice, sentence: str) -> tuple[np.ndarray, int]:
        chunks = list(voice.synthesize(sentence))
        pcm = np.concatenate(
            [np.frombuffer(c.audio_int16_bytes, dtype=np.int16) for c in chunks]
        )
        return pcm, chunks[0].sample_rate

    async def speak(self, text: str, language: str) -> None:
        """Speak `text`, overlapping synthesis of sentence N+1 with playback of N."""
        voice = self._voices.get(language) or self._voices[self._config.default_language]
        loop = asyncio.get_running_loop()
        sentences = split_sentences(text)
        if not sentences:
            return

        synth = loop.run_in_executor(None, self._synthesize_sync, voice, sentences[0])
        for i, _ in enumerate(sentences):
            pcm, rate = await synth
            if i + 1 < len(sentences):
                synth = loop.run_in_executor(
                    None, self._synthesize_sync, voice, sentences[i + 1]
                )
            await self._sink.play(pcm, rate)
