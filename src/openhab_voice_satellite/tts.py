"""Shared TTS helpers: sentence splitting and pipelined synthesis playback."""

from __future__ import annotations

import asyncio
import re
from typing import Callable

import numpy as np

from .audio.sink import AudioSink

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
