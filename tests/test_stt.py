"""Transcriber: the cpu_threads advisory and the language-remap contract."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from openhab_voice_satellite.config import SttConfig
from openhab_voice_satellite.stt import Transcript


@pytest.fixture
def transcriber_factory(monkeypatch):
    calls: dict = {}

    class StubWhisperModel:
        def __init__(self, model, device, compute_type, cpu_threads):
            calls["cpu_threads"] = cpu_threads

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = StubWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)

    def make(**config_kwargs) -> dict:
        from openhab_voice_satellite.stt import Transcriber

        Transcriber(SttConfig(**config_kwargs), default_language="de")
        return calls

    return make


def test_cpu_threads_saturation_warns_but_never_coerces(
    transcriber_factory, monkeypatch, caplog
):
    monkeypatch.setattr("openhab_voice_satellite.stt.os.cpu_count", lambda: 4)
    with caplog.at_level("WARNING"):
        calls = transcriber_factory(cpu_threads=4)
    assert "cpu_threads" in caplog.text
    assert calls["cpu_threads"] == 4  # explicit config stays honored


def test_cpu_threads_below_core_count_is_silent(
    transcriber_factory, monkeypatch, caplog
):
    monkeypatch.setattr("openhab_voice_satellite.stt.os.cpu_count", lambda: 4)
    with caplog.at_level("WARNING"):
        transcriber_factory(cpu_threads=3)
    assert "cpu_threads" not in caplog.text


# --- _transcribe_sync: fixed language, remap, segment join -----------------


class _Segment:
    def __init__(self, text: str) -> None:
        self.text = text


class _Info:
    def __init__(self, language: str, all_language_probs=None) -> None:
        self.language = language
        self.all_language_probs = all_language_probs


@pytest.fixture
def transcribe(monkeypatch):
    """Build a Transcriber over a stubbed model with scriptable results."""
    calls: dict = {}

    class StubWhisperModel:
        segments: list = []
        info: _Info = _Info("de")

        def __init__(self, model, device, compute_type, cpu_threads):
            pass

        def transcribe(self, audio, language, beam_size):
            calls["language"] = language
            calls["beam_size"] = beam_size
            return list(type(self).segments), type(self).info

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = StubWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    monkeypatch.setattr("openhab_voice_satellite.stt.os.cpu_count", lambda: 8)

    def run(segments, info, **config_kwargs) -> tuple[Transcript, dict]:
        from openhab_voice_satellite.stt import Transcriber

        StubWhisperModel.segments = segments
        StubWhisperModel.info = info
        transcriber = Transcriber(SttConfig(**config_kwargs), default_language="de")
        return transcriber._transcribe_sync(np.zeros(1600, dtype=np.int16)), calls

    return run


def test_a_single_configured_language_skips_detection(transcribe):
    # whisper's detection pass costs a forward run; with one language there is
    # nothing for it to decide
    result, calls = transcribe(
        [_Segment(" licht an ")], _Info("en"), languages=["en"]
    )
    assert calls["language"] == "en"
    assert result == Transcript(text="licht an", language="en")


def test_two_languages_leave_detection_to_whisper(transcribe):
    _, calls = transcribe([_Segment("hi")], _Info("de"), languages=["de", "en"])
    assert calls["language"] is None


def test_an_out_of_set_detection_is_remapped_to_the_best_allowed(transcribe):
    """The remap is finer than the cloud engines' clamp-to-default.

    A misdetected utterance locks the TTS voice for the whole dialog round
    (the pipeline locks language on round 0), so picking the best *allowed*
    candidate rather than the default is worth having — and worth testing.
    """
    result, _ = transcribe(
        [_Segment("licht an")],
        _Info("nl", all_language_probs=[("nl", 0.6), ("en", 0.1), ("de", 0.3)]),
        languages=["de", "en"],
    )
    assert result.language == "de"  # 0.3 beats en's 0.1


def test_an_out_of_set_detection_without_probs_falls_back_to_the_default(
    transcribe,
):
    result, _ = transcribe(
        [_Segment("hi")], _Info("nl", all_language_probs=None),
        languages=["de", "en"],
    )
    assert result.language == "de"  # default_language


def test_probs_holding_no_allowed_language_fall_back_to_the_default(transcribe):
    result, _ = transcribe(
        [_Segment("hi")],
        _Info("nl", all_language_probs=[("nl", 0.9), ("fr", 0.1)]),
        languages=["de", "en"],
    )
    assert result.language == "de"


def test_an_in_set_detection_is_kept(transcribe):
    result, _ = transcribe(
        [_Segment("hi")],
        _Info("en", all_language_probs=[("de", 0.9), ("en", 0.1)]),
        languages=["de", "en"],
    )
    assert result.language == "en"  # detection wins; no remap runs


def test_segments_are_stripped_and_joined(transcribe):
    result, _ = transcribe(
        [_Segment("  schalte das  "), _Segment(" licht an ")], _Info("de")
    )
    assert result.text == "schalte das licht an"


def test_no_segments_is_an_empty_transcript(transcribe):
    # the pipeline treats an empty transcript as "back to idle", so this is a
    # real path, not a degenerate one
    result, _ = transcribe([], _Info("de"))
    assert result.text == ""
