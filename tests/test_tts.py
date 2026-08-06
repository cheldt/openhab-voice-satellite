from __future__ import annotations

import numpy as np

from openhab_voice_satellite.tts import split_sentences, stream_synthesis

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
