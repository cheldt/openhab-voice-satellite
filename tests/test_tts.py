from __future__ import annotations

import numpy as np
import pytest

from openhab_voice_satellite.fallback import CloudEngineError, PartialSpeechError
from openhab_voice_satellite.tts import (
    TTS_CHUNK_CHARS,
    play_pipelined,
    split_sentences,
    stream_synthesis,
    tts_chunks,
)

from .fakes import BufferAudioSink


def test_split_basic():
    assert split_sentences("Hallo. Wie geht es dir? Gut!") == [
        "Hallo.", "Wie geht es dir?", "Gut!",
    ]


def test_split_single_sentence():
    assert split_sentences("Das Licht ist an") == ["Das Licht ist an"]


def test_split_empty():
    assert split_sentences("   ") == []


def test_split_keeps_abbrev_number_text_together():
    # no split without whitespace after the punctuation
    assert split_sentences("Es ist 21.30 Uhr.") == ["Es ist 21.30 Uhr."]


def test_tts_chunks_splits_long_comma_list():
    # a "list all items" answer: one giant comma sentence
    text = ", ".join(f"item number {i} living room lamp" for i in range(80))
    chunks = tts_chunks(text)
    assert len(chunks) > 1
    assert all(len(c) <= TTS_CHUNK_CHARS for c in chunks)
    # nothing lost apart from separators
    assert "".join(chunks).replace(" ", "").replace(",", "") == text.replace(
        " ", ""
    ).replace(",", "")


async def test_play_pipelined_failure_before_audio_propagates():
    async def fetch(chunk: str):
        raise CloudEngineError("boom")

    with pytest.raises(CloudEngineError):
        await play_pipelined(["One.", "Two."], fetch, BufferAudioSink())


async def test_play_pipelined_failure_after_audio_carries_remainder():
    async def fetch(chunk: str):
        if chunk != "One.":
            raise CloudEngineError("boom")
        return np.ones(4, dtype=np.int16), 24000

    sink = BufferAudioSink()
    with pytest.raises(PartialSpeechError) as excinfo:
        await play_pipelined(["One.", "Two.", "Three."], fetch, sink)
    assert excinfo.value.remaining == "Two. Three."
    assert len(sink.played) == 1


async def test_stream_synthesis_plays_all_sentences_in_order():
    sink = BufferAudioSink()
    synthesized: list[str] = []

    def synth(sentence: str):
        synthesized.append(sentence)
        return np.full(4, len(sentence), dtype=np.int16), 24000

    await stream_synthesis(["One.", "Two two.", "Three."], synth, sink)
    assert synthesized == ["One.", "Two two.", "Three."]
    assert [int(pcm[0]) for pcm, _ in sink.played] == [4, 8, 6]
    assert all(rate == 24000 for _, rate in sink.played)


async def test_stream_synthesis_skips_empty_pcm():
    sink = BufferAudioSink()

    def synth(sentence: str):
        if sentence == "skip":
            return np.empty(0, dtype=np.int16), 0
        return np.ones(4, dtype=np.int16), 24000

    await stream_synthesis(["ok", "skip", "ok"], synth, sink)
    assert len(sink.played) == 2
