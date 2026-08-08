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

    outputs: dict[str, int] = {}

    def __init__(self, wakeword_models, inference_framework, ncpu=1):
        self.models = {name: object() for name in wakeword_models}
        self.model_outputs = {m: self.outputs.get(m, 1) for m in self.models}
        self.prediction_buffer: dict[str, list[float]] = {m: [0.0] for m in self.models}
        self.custom_verifier_models: dict[str, object] = {}
        self.custom_verifier_threshold = 0.1
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

    yield make
    StubModel.outputs = {}


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


def test_score_returns_the_latest_raw_score(detector_factory):
    detector = detector_factory({"wake": [0.42]}, model="wake")
    detector.process(FRAME)
    assert detector.score("wake") == pytest.approx(0.42)


def test_detector_survives_unpatchable_model(detector_factory, caplog):
    # StubModel has no .preprocessor: the ring-buffer patch must fail open
    # (warning + stock behavior), never break detector construction
    with caplog.at_level("WARNING"):
        detector = detector_factory({"wake": [0.9]}, model="wake")
    assert "preprocessor patch skipped" in caplog.text
    assert detector.process(FRAME) == "wake"
    assert detector.tail(1.0) is None  # no ring to read a pre-roll from


def test_stop_threshold_speaking_defaults_to_stop_threshold(detector_factory):
    detector = detector_factory(
        {"wake": [0.0, 0.0], "stop": [0.45, 0.45]},
        model="wake", stop_model="stop", stop_threshold=0.4,
    )
    assert detector.process(FRAME, speaking=True) == "stop"  # unchanged default


def test_stop_threshold_speaking_guards_the_stop_model(detector_factory):
    detector = detector_factory(
        {"wake": [0.0, 0.0], "stop": [0.45, 0.45]},
        model="wake", stop_model="stop",
        stop_threshold=0.4, stop_threshold_speaking=0.6,
    )
    assert detector.process(FRAME, speaking=True) is None  # 0.45 < 0.6
    assert detector.process(FRAME, speaking=False) == "stop"  # 0.45 >= 0.4


def test_patience_requires_consecutive_frames(detector_factory):
    detector = detector_factory({"wake": [0.9, 0.3, 0.9, 0.9]}, model="wake", patience=2)
    results = [detector.process(FRAME) for _ in range(4)]
    # the lone 0.9 spike is rejected; the sustained pair fires on its 2nd frame
    assert results == [None, None, None, "wake"]


def test_patience_one_keeps_single_frame_triggering(detector_factory):
    detector = detector_factory({"wake": [0.9]}, model="wake", patience=1)
    assert detector.process(FRAME) == "wake"


def test_patience_history_advances_even_when_stop_fires_first(detector_factory):
    """A stop hit short-circuits process(); the wake window must not gap."""
    detector = detector_factory(
        {"wake": [0.9, 0.9], "stop": [0.9, 0.0]},
        model="wake", stop_model="stop", patience=2,
    )
    assert detector.process(FRAME) == "stop"
    assert detector.process(FRAME) == "wake"  # both wake frames counted


def test_score_stays_raw_while_patience_suppresses(detector_factory):
    """_DuckController reads score(); patience must not hide a climbing score."""
    detector = detector_factory({"wake": [0.9]}, model="wake", patience=3)
    assert detector.process(FRAME) is None
    assert detector.score("wake") == pytest.approx(0.9)


def test_duplicate_model_basenames_fail_fast(detector_factory):
    with pytest.raises(ValueError, match="share a basename"):
        detector_factory({"wake": [0.9]}, model="wake", stop_model="wake")


def test_multi_output_model_rejected(detector_factory):
    StubModel.outputs = {"wake": 3}
    with pytest.raises(ValueError, match="multiple outputs"):
        detector_factory({"wake": [0.9]}, model="wake")
