"""WakewordDetector edge-trigger/re-arm hysteresis with a stubbed openwakeword."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from openhab_voice_satellite.config import WakewordConfig

FRAME = np.zeros(1280, dtype=np.int16)


class StubModel:
    """Scripted scores: each predict() pops the next score per model."""

    scripts: dict[str, list[float]] = {}

    def __init__(self, wakeword_models, inference_framework, ncpu=1):
        self.models = {name: object() for name in wakeword_models}
        self.prediction_buffer: dict[str, list[float]] = {m: [0.0] for m in self.models}
        self.reset_calls = 0

    def predict(self, frame):
        out = {}
        for name in self.models:
            script = self.scripts.get(name, [])
            score = script.pop(0) if script else 0.0
            self.prediction_buffer[name].append(score)
            out[name] = score
        return out

    def reset(self):
        self.reset_calls += 1


@pytest.fixture
def detector_factory(monkeypatch):
    module = types.ModuleType("openwakeword.model")
    module.Model = StubModel
    package = types.ModuleType("openwakeword")
    package.model = module
    monkeypatch.setitem(sys.modules, "openwakeword", package)
    monkeypatch.setitem(sys.modules, "openwakeword.model", module)

    def make(scripts: dict[str, list[float]], **config_kwargs):
        from openhab_voice_satellite.wakeword import WakewordDetector

        StubModel.scripts = {k: list(v) for k, v in scripts.items()}
        return WakewordDetector(WakewordConfig(**config_kwargs))

    return make


def test_edge_trigger_fires_once_then_rearms(detector_factory):
    # threshold 0.5, re-arm below 0.25
    detector = detector_factory({"wake": [0.9, 0.9, 0.3, 0.2, 0.9]}, model="wake")
    results = [detector.process(FRAME) for _ in range(5)]
    # fires once, stays disarmed while high AND while above threshold/2,
    # re-arms below 0.25, fires again
    assert results == ["wake", None, None, None, "wake"]


def test_speaking_raises_threshold(detector_factory):
    detector = detector_factory(
        {"wake": [0.6, 0.6]}, model="wake", threshold=0.5, threshold_speaking=0.7
    )
    assert detector.process(FRAME, speaking=True) is None  # 0.6 < 0.7
    assert detector.process(FRAME, speaking=False) == "wake"  # 0.6 >= 0.5


def test_stop_model_wins_over_wake(detector_factory):
    detector = detector_factory(
        {"wake": [0.9], "stop": [0.9]}, model="wake", stop_model="stop"
    )
    assert detector.process(FRAME) == "stop"


def test_reset_rearms(detector_factory):
    detector = detector_factory({"wake": [0.9, 0.9]}, model="wake")
    assert detector.process(FRAME) == "wake"
    detector.reset()
    assert detector.process(FRAME) == "wake"  # armed again without a low dip


def test_score_reads_prediction_buffer(detector_factory):
    detector = detector_factory({"wake": [0.42]}, model="wake")
    detector.process(FRAME)
    assert detector.score("wake") == pytest.approx(0.42)
