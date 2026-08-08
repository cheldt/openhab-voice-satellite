"""Decision logic shared by every engine: edge trigger, patience, thresholds.

Both engines run the identical assertions. That is the point of the split —
an engine contributes scores, never a decision, so swapping one must not
change when the app sees a detection.
"""

from __future__ import annotations

import numpy as np
import pytest

from .wakeword_stubs import ENGINES, make_detector, reset_stub_state

FRAME = np.zeros(1280, dtype=np.int16)


@pytest.fixture(params=ENGINES)
def detector_factory(request, monkeypatch):
    reset_stub_state()

    def make(scripts: dict[str, list[float]], **config_kwargs):
        return make_detector(request.param, monkeypatch, scripts, **config_kwargs)

    yield make
    reset_stub_state()


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


def test_score_falls_back_to_wake_without_a_stop_model(detector_factory):
    # app.py reads score("stop") unconditionally; no stop model must not KeyError
    detector = detector_factory({"wake": [0.42]}, model="wake")
    detector.process(FRAME)
    assert detector.score("stop") == pytest.approx(0.42)


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


def test_detection_survives_playback(detector_factory):
    """Barge-in: speaking only raises the bar, it never blocks a detection.

    violawake's own detect() rejects unconditionally while audio is playing;
    routing around that gate is the reason this adapter uses process().
    """
    detector = detector_factory(
        {"wake": [0.9], "stop": [0.9]},
        model="wake", stop_model="stop", threshold_speaking=0.7,
    )
    assert detector.process(FRAME, speaking=True) == "stop"
    detector = detector_factory({"wake": [0.9]}, model="wake", threshold_speaking=0.7)
    assert detector.process(FRAME, speaking=True) == "wake"
