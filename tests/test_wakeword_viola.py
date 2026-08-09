"""ViolaWake specifics: one engine per model, the tail ring, the opt-in extras."""

from __future__ import annotations

import numpy as np
import pytest

from .wakeword_stubs import (
    StubNoiseProfiler,
    StubPowerManager,
    StubWakeDetector,
    make_detector,
    reset_stub_state,
)

FRAME = np.zeros(1280, dtype=np.int16)


@pytest.fixture
def detector_factory(monkeypatch):
    reset_stub_state()

    def make(scripts: dict[str, list[float]], **config_kwargs):
        return make_detector("violawake", monkeypatch, scripts, **config_kwargs)

    yield make
    reset_stub_state()


def test_one_engine_per_configured_model(detector_factory):
    detector_factory({"wake": [0.0]}, model="wake")
    assert [d.model for d in StubWakeDetector.instances] == ["wake"]
    reset_stub_state()


def test_engines_are_built_inside_the_single_threaded_block(monkeypatch):
    """Wherever violawake builds its sessions, they must be bound."""
    from contextlib import contextmanager

    from openhab_voice_satellite import wakeword_viola

    reset_stub_state()
    seen = []

    @contextmanager
    def recording():
        seen.append(len(StubWakeDetector.instances))
        yield
        seen.append(len(StubWakeDetector.instances))

    monkeypatch.setattr(wakeword_viola, "single_threaded_sessions", recording)
    make_detector("violawake", monkeypatch, {"wake": [0.0]}, model="wake")
    assert seen == [0, 1]  # constructed inside the block, not before or after


def test_stop_model_adds_a_second_engine(detector_factory):
    detector_factory({"wake": [0.0], "stop": [0.0]}, model="wake", stop_model="stop")
    # each carries its own backbone; this is the doubled per-frame cost
    assert sorted(d.model for d in StubWakeDetector.instances) == ["stop", "wake"]


def test_reset_resets_every_engine_and_clears_the_ring(detector_factory):
    detector = detector_factory(
        {"wake": [0.0], "stop": [0.0]}, model="wake", stop_model="stop"
    )
    detector.process(np.full(1280, 500, dtype=np.int16))
    detector.reset()
    assert all(d.resets == 1 for d in StubWakeDetector.instances)
    assert not detector.tail(1.0).any()  # ring emptied with the engines


def test_tail_returns_the_audio_that_preceded_the_detection(detector_factory):
    detector = detector_factory({"wake": [0.9]}, model="wake")
    frame = np.arange(1280, dtype=np.int16)
    assert detector.process(frame) == "wake"
    tail = detector.tail(1280 / 16000)
    assert np.array_equal(tail, frame)


def test_frames_reach_the_engine_unmodified(detector_factory):
    # mic frames are shared across subscribers; mutating one corrupts the others
    detector = detector_factory({"wake": [0.0]}, model="wake")
    frame = np.arange(1280, dtype=np.int16)
    detector.process(frame)
    assert np.array_equal(StubWakeDetector.instances[0].frames[0], np.arange(1280))


def test_no_extras_by_default(detector_factory):
    detector_factory({"wake": [0.0]}, model="wake")
    assert StubNoiseProfiler.instances == []
    assert StubPowerManager.instances == []


def test_adaptive_threshold_replaces_the_configured_one(detector_factory):
    StubNoiseProfiler.adapted = [0.9, 0.3]
    detector = detector_factory(
        {"wake": [0.5, 0.5]},
        model="wake",
        threshold=0.4,
        viola={"adaptive": {"enabled": True}},
    )
    # 0.5 clears the configured 0.4 but not the profiler's 0.9
    assert detector.process(FRAME) is None
    # ...and fires once the profiler drops the bar to 0.3
    assert detector.process(FRAME) == "wake"


def test_adaptive_threshold_keeps_the_speaking_margin(detector_factory):
    StubNoiseProfiler.adapted = [0.5, 0.5]
    detector = detector_factory(
        {"wake": [0.6, 0.6]},
        model="wake",
        threshold=0.4,
        threshold_speaking=0.7,  # +0.3 margin, so the adapted bar becomes 0.8
        viola={"adaptive": {"enabled": True}},
    )
    assert detector.process(FRAME, speaking=True) is None  # 0.6 < 0.5 + 0.3
    assert detector.process(FRAME, speaking=False) == "wake"  # 0.6 >= 0.5


def test_profiler_frame_rate_follows_the_configured_frame_size(detector_factory):
    """Upstream defaults to 50 fps (its 20 ms frames); ours are 80 ms."""
    detector_factory(
        {"wake": [0.0]}, model="wake", frame_ms=80,
        viola={"adaptive": {"enabled": True}},
    )
    assert StubNoiseProfiler.instances[0].kwargs["frames_per_second"] == pytest.approx(12.5)


def test_adaptive_only_moves_the_wake_bar(detector_factory):
    StubNoiseProfiler.adapted = [0.9]
    detector = detector_factory(
        {"wake": [0.0], "stop": [0.5]},
        model="wake", stop_model="stop", stop_threshold=0.4,
        viola={"adaptive": {"enabled": True}},
    )
    assert detector.process(FRAME) == "stop"  # 0.5 >= its own static 0.4


def test_skipped_frame_does_not_advance_the_patience_window(detector_factory):
    StubPowerManager.decisions = [True, False, True]
    detector = detector_factory(
        {"wake": [0.9, 0.9]},
        model="wake", patience=2,
        viola={"power": {"enabled": True}},
    )
    assert detector.process(FRAME) is None  # 1 of 2
    assert detector.process(FRAME) is None  # skipped: no score, no progress
    assert detector.process(FRAME) == "wake"  # 2 of 2, window unbroken


def test_skipped_frame_still_reaches_the_tail_ring(detector_factory):
    StubPowerManager.decisions = [False]
    detector = detector_factory(
        {"wake": [0.0]}, model="wake", viola={"power": {"enabled": True}}
    )
    frame = np.arange(1280, dtype=np.int16)
    assert detector.process(frame) is None
    # the mic still heard it, so a later dump must contain it
    assert np.array_equal(detector.tail(1280 / 16000), frame)


def test_skipped_frame_is_never_scored(detector_factory):
    StubPowerManager.decisions = [False]
    detector = detector_factory(
        {"wake": [0.9]}, model="wake", viola={"power": {"enabled": True}}
    )
    detector.process(FRAME)
    assert StubWakeDetector.instances[0].frames == []
