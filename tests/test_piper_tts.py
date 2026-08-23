"""PiperSpeaker tests with a stubbed `piper` module (no model loading)."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from openhab_voice_satellite.config import PiperConfig, TtsConfig
from openhab_voice_satellite.piper_tts import PiperSpeaker

from .fakes import BufferAudioSink


class _Chunk:
    def __init__(self, pcm: np.ndarray, sample_rate: int) -> None:
        self.audio_int16_bytes = pcm.tobytes()
        self.sample_rate = sample_rate


class StubPiperVoice:
    """Yields one deterministic chunk per sentence; records what was spoken."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.sentences: list[str] = []
        self.sample_rate = 16000 if "de_DE" in path else 22050
        self.empty = False

    @classmethod
    def load(cls, path: str) -> StubPiperVoice:
        return cls(path)

    def synthesize(self, sentence: str):
        self.sentences.append(sentence)
        if self.empty:
            return
        yield _Chunk(np.full(4, len(sentence), dtype=np.int16), self.sample_rate)


@pytest.fixture
def speaker(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "piper", types.SimpleNamespace(PiperVoice=StubPiperVoice)
    )
    sink = BufferAudioSink()
    speaker = PiperSpeaker(
        PiperConfig(
            voices={"de": "models/piper/de_DE-x.onnx", "en": "models/piper/en_GB-x.onnx"}
        ),
        TtsConfig(default_language="de"),
        sink,
    )
    return speaker, sink


async def test_speak_per_sentence(speaker):
    spk, sink = speaker
    await spk.speak("Hello there. How are you?", "en")
    assert [s for v in spk._voices.values() for s in v.sentences] == [
        "Hello there.", "How are you?",
    ]
    assert len(sink.played) == 2
    for pcm, rate in sink.played:
        assert pcm.dtype == np.int16
        assert rate == 22050  # en stub rate


async def test_unknown_language_falls_back_to_default(speaker):
    spk, sink = speaker
    await spk.speak("Bonjour.", "fr")
    assert spk._voices["de"].sentences == ["Bonjour."]
    assert sink.played[0][1] == 16000  # de stub rate


async def test_empty_text_is_noop(speaker):
    spk, sink = speaker
    await spk.speak("   ", "de")
    assert sink.played == []


async def test_empty_synthesis_not_played(speaker):
    spk, sink = speaker
    spk._voices["de"].empty = True
    await spk.speak("Hallo.", "de")
    assert sink.played == []


async def test_an_unpunctuated_long_answer_is_chunked(speaker):
    """The case TTS_CHUNK_CHARS exists for reaches piper too.

    A "list all items" answer is one giant comma-separated sentence, so
    splitting on punctuation alone handed it to piper as a single
    synthesize() call — the whole thing materialized before a byte could
    play, pipelining gone, the satellite silent in SPEAKING for the duration.
    """
    from openhab_voice_satellite.tts import TTS_CHUNK_CHARS

    items = ", ".join(f"Lampe {i}" for i in range(120))  # no sentence end
    assert len(items) > 2 * TTS_CHUNK_CHARS
    spk, sink = speaker
    await spk.speak(items, "de")

    spoken = spk._voices["de"].sentences
    assert len(spoken) > 2                              # actually pipelined
    assert all(len(c) <= TTS_CHUNK_CHARS for c in spoken)
    assert len(sink.played) == len(spoken)


async def test_a_punctuated_answer_still_splits_per_sentence(speaker):
    # chunking must not change the ordinary case: short sentences pass through
    spk, sink = speaker
    await spk.speak("Eins. Zwei. Drei.", "de")
    assert spk._voices["de"].sentences == ["Eins.", "Zwei.", "Drei."]
