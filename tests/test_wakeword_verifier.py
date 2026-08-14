"""Stage-2 verifier: wake deferral, accept/reject, and what it must not touch.

The verifier itself is stubbed at the `_build_verifier`/`_verify` seam — its
scoring quality is the violawakeword repo's problem (TRAINING.md); these tests
pin the decision plumbing: a WAKE is withheld for the configured delay, then
either released or swallowed by the verifier's verdict, while STOP and the
score histories behave as if the verifier did not exist.
"""

from __future__ import annotations

import numpy as np
import pytest

from openhab_voice_satellite import wakeword_viola

from .wakeword_stubs import make_detector, reset_stub_state

FRAME = np.zeros(1280, dtype=np.int16)

VERIFIER_CONFIG = {"verifier": {"model": "v.onnx", "mel_basis": "m.npy",
                                "threshold": 0.5, "delay_ms": 300}}


@pytest.fixture
def verified_factory(monkeypatch):
    """Detector factory with the verifier seam stubbed to scripted scores."""
    reset_stub_state()
    scores: list[float] = []

    monkeypatch.setattr(
        wakeword_viola.ViolaWakeDetector,
        "_build_verifier",
        staticmethod(
            lambda config: ("session", "frontend") if config.viola.verifier.model else None
        ),
    )
    monkeypatch.setattr(
        wakeword_viola.ViolaWakeDetector,
        "_verify",
        lambda self: scores.pop(0),
    )

    def make(scripts, verifier_scores, **config_kwargs):
        scores.clear()
        scores.extend(verifier_scores)
        # frame_ms=80 and delay_ms=300 -> ceil(300/80) = 4 frames of deferral,
        # the trigger frame included
        return make_detector("violawake", monkeypatch, scripts, **config_kwargs)

    yield make
    reset_stub_state()


def test_wake_is_deferred_then_released_on_accept(verified_factory):
    detector = verified_factory(
        {"wake": [0.9, 0.0, 0.0, 0.0]}, [0.8],
        model="wake", viola=VERIFIER_CONFIG,
    )
    assert detector.process(FRAME) is None  # stage-1 fired; deferred
    assert detector.process(FRAME) is None
    assert detector.process(FRAME) is None
    assert detector.process(FRAME) == "wake"  # verifier 0.8 >= 0.5


def test_wake_is_swallowed_on_reject(verified_factory):
    detector = verified_factory(
        {"wake": [0.9, 0.0, 0.0, 0.0, 0.0]}, [0.2],
        model="wake", viola=VERIFIER_CONFIG,
    )
    assert all(detector.process(FRAME) is None for _ in range(5))
    assert detector.last_verifier_score == pytest.approx(0.2)


def test_stop_fires_during_a_pending_verification(verified_factory):
    # stopping playback late defeats its purpose; STOP is never deferred
    detector = verified_factory(
        {"wake": [0.9, 0.0], "stop": [0.0, 0.9]}, [0.8],
        model="wake", stop_model="stop", viola=VERIFIER_CONFIG,
    )
    assert detector.process(FRAME) is None  # wake deferred
    assert detector.process(FRAME) == "stop"


def test_second_trigger_keeps_the_first_countdown(verified_factory):
    # restarting the countdown would push the capture window past the phrase
    detector = verified_factory(
        {"wake": [0.9, 0.1, 0.9, 0.0]}, [0.8],
        model="wake", viola=VERIFIER_CONFIG,
    )
    assert detector.process(FRAME) is None  # trigger, countdown 4
    assert detector.process(FRAME) is None  # 0.1 re-arms the edge
    assert detector.process(FRAME) is None  # re-trigger must not restart it
    assert detector.process(FRAME) == "wake"  # original schedule holds


def test_reset_clears_a_pending_verification(verified_factory):
    detector = verified_factory(
        {"wake": [0.9, 0.0, 0.0, 0.0]}, [0.8],
        model="wake", viola=VERIFIER_CONFIG,
    )
    assert detector.process(FRAME) is None
    detector.reset()
    # the pending wake died with the reset; nothing fires afterwards
    assert all(detector.process(FRAME) is None for _ in range(4))
    assert detector.last_verifier_score is None


def test_no_verifier_means_no_deferral(verified_factory):
    detector = verified_factory({"wake": [0.9]}, [], model="wake")
    assert detector.process(FRAME) == "wake"


def test_verifier_is_built_outside_the_single_threaded_block(monkeypatch):
    """The verifier passes explicit single-thread options itself; building it
    inside the block would let its scipy import trip the block's OS-thread
    watchdog with BLAS pool threads that are not ORT's."""
    from contextlib import contextmanager

    reset_stub_state()
    calls: list[str] = []

    @contextmanager
    def recording():
        calls.append("enter")
        yield
        calls.append("exit")

    monkeypatch.setattr(wakeword_viola, "single_threaded_sessions", recording)
    monkeypatch.setattr(
        wakeword_viola.ViolaWakeDetector,
        "_build_verifier",
        staticmethod(lambda config: calls.append("verifier") or None),
    )
    make_detector("violawake", monkeypatch, {"wake": [0.0]}, model="wake",
                  viola=VERIFIER_CONFIG)
    assert calls == ["enter", "exit", "verifier"]
