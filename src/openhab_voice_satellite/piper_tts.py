"""Piper TTS: per-language voices, sentence-split streaming synthesis."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from .audio.sink import AudioSink
from .config import PiperConfig, TtsConfig
from .tts import stream_synthesis, tts_chunks

log = logging.getLogger(__name__)


class PiperSpeaker:
    def __init__(
        self,
        config: PiperConfig,
        tts_config: TtsConfig,
        sink: AudioSink,
    ) -> None:
        from piper import PiperVoice

        self._sink = sink
        self._default_language = tts_config.default_language
        self._voices = {}
        for lang, model_path in config.voices.items():
            path = Path(model_path)
            self._voices[lang] = PiperVoice.load(str(path))
            log.info("piper voice loaded: %s -> %s", lang, path.name)

    def _synthesize_sync(self, voice, sentence: str) -> tuple[np.ndarray, int]:
        start = time.monotonic()
        chunks = list(voice.synthesize(sentence))
        if not chunks:
            return np.empty(0, dtype=np.int16), 0
        pcm = np.concatenate(
            [np.frombuffer(c.audio_int16_bytes, dtype=np.int16) for c in chunks]
        )
        sample_rate = chunks[0].sample_rate
        elapsed = time.monotonic() - start
        duration = len(pcm) / sample_rate if sample_rate else 0.0
        log.debug(
            "synthesized %.1fs audio in %.1fs (rtf %.2f): %r",
            duration, elapsed, elapsed / duration if duration else 0.0, sentence,
        )
        return pcm, sample_rate

    async def speak(self, text: str, language: str) -> None:
        """Speak `text`, overlapping synthesis of chunk N+1 with playback of N.

        `tts_chunks`, not `split_sentences`: the case TTS_CHUNK_CHARS was
        built for is a "list all items" answer that is one giant
        comma-separated sentence, and splitting on punctuation alone hands
        that to piper as a single synthesize() call. `_synthesize_sync`
        materializes the whole thing before a byte can play, so pipelining
        degenerates to nothing and the satellite sits silent in SPEAKING for
        as long as the synthesis takes. The same applies on the fallback path,
        where a PartialSpeechError remainder is a space-joined comma list with
        no sentence punctuation left in it at all.
        """
        voice = self._voices.get(language) or self._voices[self._default_language]
        chunks = tts_chunks(text)
        if not chunks:
            return
        await stream_synthesis(
            chunks, lambda s: self._synthesize_sync(voice, s), self._sink
        )
