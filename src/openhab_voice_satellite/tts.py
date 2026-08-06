"""Kokoro TTS: per-language models, sentence-split streaming synthesis."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .audio.sink import AudioSink
from .config import KokoroConfig, KokoroVoiceConfig, TtsConfig, resolve_path

log = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?:;])\s+")


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT.split(text.strip()) if s]


async def stream_synthesis(
    sentences: list[str],
    synth: Callable[[str], tuple[np.ndarray, int]],
    sink: AudioSink,
) -> None:
    """Play sentence N while synthesizing N+1 in the executor."""
    loop = asyncio.get_running_loop()
    pending = loop.run_in_executor(None, synth, sentences[0])
    for i, _ in enumerate(sentences):
        pcm, rate = await pending
        if i + 1 < len(sentences):
            pending = loop.run_in_executor(None, synth, sentences[i + 1])
        if len(pcm):
            await sink.play(pcm, rate)


def to_int16(samples: np.ndarray) -> np.ndarray:
    """float32 [-1, 1] -> int16 PCM."""
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)


def make_onnx_session(model_path: str, threads: int):
    """ONNX session with a bounded, non-spinning thread pool.

    ORT defaults to one intra-op thread per core with spin-waiting workers,
    which burns idle CPU long after the last inference.
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(
        model_path, sess_options=opts, providers=["CPUExecutionProvider"]
    )


class Speaker:
    def __init__(
        self,
        config: KokoroConfig,
        tts_config: TtsConfig,
        sink: AudioSink,
        base_dir: Path | None = None,
    ) -> None:
        from kokoro_onnx import Kokoro

        self._sink = sink
        self._default_language = tts_config.default_language
        base = base_dir or Path.cwd()

        self._engines: dict[str, tuple[Kokoro, KokoroVoiceConfig]] = {}
        cache: dict[tuple[Path, Path], Kokoro] = {}
        for lang, vc in config.voices.items():
            key = (resolve_path(vc.model, base), resolve_path(vc.voices, base))
            if key not in cache:
                cache[key] = Kokoro.from_session(
                    make_onnx_session(str(key[0]), config.threads), str(key[1])
                )
            self._engines[lang] = (cache[key], vc)
            log.info("kokoro voice loaded: %s -> %s (%s)", lang, vc.voice, key[0].name)

    def _synthesize_sync(
        self, engine, vc: KokoroVoiceConfig, sentence: str
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
        engine, vc = self._engines.get(language) or self._engines[self._default_language]
        sentences = split_sentences(text)
        if not sentences:
            return
        await stream_synthesis(
            sentences, lambda s: self._synthesize_sync(engine, vc, s), self._sink
        )
