"""openWakeWord specifics: the model-name mapping, verifiers, patch fail-open."""

from __future__ import annotations

import numpy as np
import pytest

from .wakeword_stubs import StubModel, make_detector, reset_stub_state

FRAME = np.zeros(1280, dtype=np.int16)


@pytest.fixture
def detector_factory(monkeypatch):
    reset_stub_state()

    def make(scripts: dict[str, list[float]], **config_kwargs):
        return make_detector("openwakeword", monkeypatch, scripts, **config_kwargs)

    yield make
    reset_stub_state()


def test_detector_survives_unpatchable_model(detector_factory, caplog):
    # StubModel has no .preprocessor: the ring-buffer patch must fail open
    # (warning + stock behavior), never break detector construction
    with caplog.at_level("WARNING"):
        detector = detector_factory({"wake": [0.9]}, model="wake")
    assert "preprocessor patch skipped" in caplog.text
    assert detector.process(FRAME) == "wake"
    assert detector.tail(1.0) is None  # no ring to read a pre-roll from


def test_duplicate_model_basenames_fail_fast(detector_factory):
    with pytest.raises(ValueError, match="share a basename"):
        detector_factory({"wake": [0.9]}, model="wake", stop_model="wake")


def test_multi_output_model_rejected(detector_factory):
    StubModel.outputs = {"wake": 3}
    with pytest.raises(ValueError, match="multiple outputs"):
        detector_factory({"wake": [0.9]}, model="wake")


def test_canonical_keys_map_onto_model_names(detector_factory):
    """Config names the models; the base class only ever sees wake/stop."""
    detector = detector_factory(
        {"jarvis": [0.9], "halt": [0.1]}, model="jarvis", stop_model="halt"
    )
    detector.process(FRAME)
    assert detector.score("wake") == pytest.approx(0.9)
    assert detector.score("stop") == pytest.approx(0.1)


def test_reset_resets_the_underlying_model(detector_factory):
    detector = detector_factory({"wake": [0.9]}, model="wake")
    detector.reset()
    assert detector._model.reset_calls == 1
