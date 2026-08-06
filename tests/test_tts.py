from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import openhab_voice_satellite.tts as tts_module
from openhab_voice_satellite.config import KokoroConfig, KokoroVoiceConfig, TtsConfig
from openhab_voice_satellite.tts import Speaker, split_sentences, stream_synthesis, to_int16

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


def test_to_int16_scales_and_clips():
    samples = np.array([0.0, 0.5, 1.0, -1.0, 2.0, -2.0], dtype=np.float32)
    pcm = to_int16(samples)
    assert pcm.dtype == np.int16
    assert pcm[0] == 0
    assert pcm[1] == 16383
    assert pcm[2] == 32767
    assert pcm[3] == -32767
    assert pcm[4] == 32767  # clipped
    assert pcm[5] == -32767  # clipped


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


# --- Speaker (kokoro) with stubbed engine ----------------------------------


class StubKokoro:
    """Records created sentences; returns a short float ramp per sentence."""

    instances: list[StubKokoro] = []

    def __init__(self) -> None:
        self.created: list[tuple[str, str, float, str]] = []
        StubKokoro.instances.append(self)

    @classmethod
    def from_session(cls, session, voices_path: str) -> StubKokoro:
        return cls()

    def create(self, sentence, voice, speed, lang):
        self.created.append((sentence, voice, speed, lang))
        return np.linspace(0.0, 0.5, 8, dtype=np.float32), 24000


@pytest.fixture
def kokoro_speaker(monkeypatch, tmp_path):
    module = types.ModuleType("kokoro_onnx")
    module.Kokoro = StubKokoro
    monkeypatch.setitem(sys.modules, "kokoro_onnx", module)
    monkeypatch.setattr(tts_module, "make_onnx_session", lambda path, threads: object())
    StubKokoro.instances = []

    def make(voices: dict[str, KokoroVoiceConfig]) -> tuple[Speaker, BufferAudioSink]:
        sink = BufferAudioSink()
        speaker = Speaker(
            KokoroConfig(voices=voices), TtsConfig(), sink, base_dir=tmp_path
        )
        return speaker, sink

    return make


def _voice(model: str, voice: str, lang: str) -> KokoroVoiceConfig:
    return KokoroVoiceConfig(model=model, voices=f"{model}.bin", voice=voice, lang=lang)


async def test_speaker_speaks_per_sentence_with_language_voice(kokoro_speaker):
    speaker, sink = kokoro_speaker(
        {"de": _voice("de.onnx", "martin", "de"), "en": _voice("en.onnx", "bf_emma", "en-gb")}
    )
    await speaker.speak("Erster Satz. Zweiter Satz.", "de")
    engine = speaker._engines["de"][0]
    assert [c[0] for c in engine.created] == ["Erster Satz.", "Zweiter Satz."]
    assert all(c[1] == "martin" for c in engine.created)
    assert len(sink.played) == 2
    assert all(rate == 24000 for _, rate in sink.played)
    assert all(pcm.dtype == np.int16 for pcm, _ in sink.played)


async def test_speaker_unknown_language_falls_back_to_default(kokoro_speaker):
    speaker, sink = kokoro_speaker(
        {"de": _voice("de.onnx", "martin", "de"), "en": _voice("en.onnx", "bf_emma", "en-gb")}
    )
    await speaker.speak("Bonjour.", "fr")  # tts default_language is "de"
    engine = speaker._engines["de"][0]
    assert engine.created[0][1] == "martin"


async def test_speaker_shares_engine_for_same_model(kokoro_speaker):
    # two languages pointing at one model file must share one session
    speaker, _ = kokoro_speaker(
        {"de": _voice("same.onnx", "martin", "de"), "en": _voice("same.onnx", "emma", "en-gb")}
    )
    assert len(StubKokoro.instances) == 1


async def test_speaker_empty_text_is_noop(kokoro_speaker):
    speaker, sink = kokoro_speaker({"de": _voice("de.onnx", "martin", "de")})
    await speaker.speak("   ", "de")
    assert sink.played == []
