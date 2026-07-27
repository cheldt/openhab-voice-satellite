"""Kokoro TTS: per-language models, sentence-split streaming synthesis."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path

import numpy as np

from .audio.sink import AudioSink
from .config import TtsConfig, TtsVoiceConfig

log = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?:;])\s+")


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT.split(text.strip()) if s]


def to_int16(samples: np.ndarray) -> np.ndarray:
    """float32 [-1, 1] -> int16 PCM."""
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)


class Speaker:
    def __init__(self, config: TtsConfig, sink: AudioSink, base_dir: Path | None = None) -> None:
        from kokoro_onnx import Kokoro

        self._sink = sink
        self._config = config
        base = base_dir or Path.cwd()

        def resolve(p: str) -> Path:
            return Path(p) if Path(p).is_absolute() else base / p

        self._engines: dict[str, tuple[Kokoro, TtsVoiceConfig]] = {}
        cache: dict[tuple[Path, Path], Kokoro] = {}
        for lang, vc in config.voices.items():
            key = (resolve(vc.model), resolve(vc.voices))
            if key not in cache:
                cache[key] = Kokoro(str(key[0]), str(key[1]))
            self._engines[lang] = (cache[key], vc)
            log.info("kokoro voice loaded: %s -> %s (%s)", lang, vc.voice, key[0].name)

    def _synthesize_sync(
        self, engine, vc: TtsVoiceConfig, sentence: str
    ) -> tuple[np.ndarray, int]:
        start = time.monotonic()
        samples, sample_rate = engine.create(
            sentence, voice=vc.voice, speed=vc.speed, lang=vc.lang
        )
        elapsed = time.monotonic() - start
        duration = len(samples) / sample_rate if sample_rate else 0.0
        log.debug(
            "synthesized %.1fs audio in %.1fs (rtf %.2f): %r",
            duration, elapsed, elapsed / duration if duration else 0.0, sentence,
        )
        return to_int16(samples), sample_rate

    async def speak(self, text: str, language: str) -> None:
        """Speak `text`, overlapping synthesis of sentence N+1 with playback of N."""
        engine, vc = self._engines.get(language) or self._engines[self._config.default_language]
        loop = asyncio.get_running_loop()
        sentences = split_sentences(text)
        if not sentences:
            return

        synth = loop.run_in_executor(None, self._synthesize_sync, engine, vc, sentences[0])
        for i, _ in enumerate(sentences):
            pcm, rate = await synth
            if i + 1 < len(sentences):
                synth = loop.run_in_executor(
                    None, self._synthesize_sync, engine, vc, sentences[i + 1]
                )
            if len(pcm):
                await self._sink.play(pcm, rate)
