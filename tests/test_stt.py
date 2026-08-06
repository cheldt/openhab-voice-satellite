"""Transcriber construction: the cpu_threads saturation advisory."""

from __future__ import annotations

import sys
import types

import pytest

from openhab_voice_satellite.config import SttConfig


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
