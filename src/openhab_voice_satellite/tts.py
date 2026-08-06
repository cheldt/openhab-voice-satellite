"""Shared TTS helpers: sentence splitting, chunking, pipelined playback."""

from __future__ import annotations

import asyncio
import re
from typing import Awaitable, Callable

import numpy as np

from .audio.sink import AudioSink
from .fallback import FALLBACK_ERRORS, PartialSpeechError

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?:;])\s+")

# Cloud TTS synthesizes a whole request before responding, so long texts
# (e.g. "list all items" answers) blow the timeout. Chunks this size return
# in a few seconds and are pipelined with playback.
TTS_CHUNK_CHARS = 400


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT.split(text.strip()) if s]


def tts_chunks(text: str) -> list[str]:
    """Sentences, with oversized ones split further at commas/spaces."""
    chunks = []
    for sentence in split_sentences(text):
        while len(sentence) > TTS_CHUNK_CHARS:
            cut = sentence.rfind(", ", 0, TTS_CHUNK_CHARS)
            if cut == -1:
                cut = sentence.rfind(" ", 0, TTS_CHUNK_CHARS)
            if cut == -1:
                cut = TTS_CHUNK_CHARS
            head, sentence = sentence[:cut].rstrip(","), sentence[cut:].lstrip(", ")
            if head:
                chunks.append(head)
        if sentence:
            chunks.append(sentence)
    return chunks


async def play_pipelined(
    chunks: list[str],
    fetch: Callable[[str], Awaitable[tuple[np.ndarray, int]]],
    sink: AudioSink,
) -> None:
    """Play chunk N while fetching/synthesizing chunk N+1.

    A fetch that fails with FALLBACK_ERRORS after audio already played raises
    PartialSpeechError carrying the unspoken remainder, so the fallback engine
    picks up where playback stopped instead of repeating the whole utterance.
    """
    pending: asyncio.Task | None = asyncio.create_task(fetch(chunks[0]))
    played = False
    try:
        for i in range(len(chunks)):
            current, pending = pending, None
            try:
                pcm, rate = await current
            except FALLBACK_ERRORS as exc:
                if played:
                    # some audio already out — hand only the rest to the fallback
                    raise PartialSpeechError(
                        str(exc), remaining=" ".join(chunks[i:])
                    ) from exc
                raise
            if i + 1 < len(chunks):
                pending = asyncio.create_task(fetch(chunks[i + 1]))
            if len(pcm):
                await sink.play(pcm, rate)
                played = True
    finally:
        if pending is not None:
            pending.cancel()


async def stream_synthesis(
    sentences: list[str],
    synth: Callable[[str], tuple[np.ndarray, int]],
    sink: AudioSink,
) -> None:
    """Play sentence N while synthesizing N+1 in the executor."""
    loop = asyncio.get_running_loop()

    async def fetch(sentence: str) -> tuple[np.ndarray, int]:
        return await loop.run_in_executor(None, synth, sentence)

    await play_pipelined(sentences, fetch, sink)
