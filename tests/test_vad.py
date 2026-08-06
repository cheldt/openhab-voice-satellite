"""SpeechEndpointer chunking and silence accounting with scripted probabilities."""

from __future__ import annotations

import numpy as np
import pytest

from openhab_voice_satellite.config import VadConfig
from openhab_voice_satellite.vad import VAD_CHUNK, SpeechEndpointer


@pytest.fixture
def endpointer_factory(monkeypatch):
    def make(probabilities: list[float], **config_kwargs) -> SpeechEndpointer:
        ep = SpeechEndpointer(VadConfig(**config_kwargs))
        script = list(probabilities)
        calls: list[int] = []

        def fake_probability(chunk: np.ndarray) -> float:
            calls.append(len(chunk))
            return script.pop(0) if script else 0.0

        monkeypatch.setattr(ep, "probability", fake_probability)
        ep.probability_calls = calls  # type: ignore[attr-defined]
        return ep

    return make


def _frame(samples: int) -> np.ndarray:
    return np.zeros(samples, dtype=np.int16)


def test_residual_rechunking(endpointer_factory):
    # frames are not VAD_CHUNK-aligned; the residual must carry over
    ep = endpointer_factory([0.0] * 10)
    ep.update(_frame(800))  # 800 -> 1 chunk of 512, residual 288
    assert ep.probability_calls == [VAD_CHUNK]
    ep.update(_frame(800))  # 288+800=1088 -> 2 chunks, residual 64
    assert ep.probability_calls == [VAD_CHUNK] * 3
    ep.update(_frame(448))  # 64+448=512 -> exactly 1 chunk, residual 0
    assert ep.probability_calls == [VAD_CHUNK] * 4


def test_speech_starts_and_endpoint_after_silence(endpointer_factory):
    # 1 speech chunk, then silence until silence_ms is reached
    ep = endpointer_factory([0.9, 0.0, 0.0, 0.0, 0.0], threshold=0.5, silence_ms=64)
    assert ep.update(_frame(VAD_CHUNK)) is True
    assert ep.speech_started
    assert not ep.endpoint_reached
    ep.update(_frame(VAD_CHUNK))  # 32 ms silence
    assert not ep.endpoint_reached
    ep.update(_frame(VAD_CHUNK))  # 64 ms silence
    assert ep.endpoint_reached


def test_no_endpoint_without_speech(endpointer_factory):
    ep = endpointer_factory([0.0] * 20, silence_ms=32)
    for _ in range(10):
        ep.update(_frame(VAD_CHUNK))
    assert not ep.endpoint_reached  # silence alone never endpoints


def test_speech_resets_silence_accounting(endpointer_factory):
    ep = endpointer_factory([0.9, 0.0, 0.9, 0.0, 0.0], threshold=0.5, silence_ms=64)
    for _ in range(4):
        ep.update(_frame(VAD_CHUNK))
    # the mid-stream speech chunk reset the silence run: only 32 ms so far
    assert not ep.endpoint_reached
    ep.update(_frame(VAD_CHUNK))
    assert ep.endpoint_reached


def test_elapsed_tracks_total_samples(endpointer_factory):
    ep = endpointer_factory([0.0] * 4)
    ep.update(_frame(1280))
    ep.update(_frame(1280))
    assert ep.elapsed_s == pytest.approx(2 * 1280 / 16000)


def test_reset_clears_state(endpointer_factory):
    ep = endpointer_factory([0.9, 0.0], threshold=0.5)
    ep.update(_frame(800))
    ep.reset()
    assert not ep.speech_started
    assert ep.elapsed_s == 0.0
