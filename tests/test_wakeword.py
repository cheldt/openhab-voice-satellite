"""Decision logic shared by every engine: edge trigger, patience, thresholds.

Both engines run the identical assertions. That is the point of the split —
an engine contributes scores, never a decision, so swapping one must not
change when the app sees a detection.
"""

from __future__ import annotations

import numpy as np
import pytest

from openhab_voice_satellite.config import WakewordConfig
from openhab_voice_satellite.wakeword import BaseWakewordDetector

from .wakeword_stubs import ENGINES, make_detector, prime, reset_stub_state

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


def test_patience_does_not_re_judge_playback_frames_at_the_idle_bar(detector_factory):
    """Each frame is judged against the bar that applied when it arrived.

    The patience window is a window of verdicts, not of scores. Judging the
    stored scores against the current frame's bar meant that on the first
    frame after playback ended, scores that had only cleared the raised
    speaking bar were re-read at the lower idle one — so a detection could be
    built half out of a frame the speaking threshold existed to reject, which
    is echo pressure at every playback boundary.
    """
    detector = detector_factory(
        {"wake": [0.45, 0.45]},
        model="wake", threshold=0.35, threshold_speaking=0.55, patience=2,
    )
    assert detector.process(FRAME, speaking=True) is None  # 0.45 < 0.55
    # the sink drained; this frame clears the idle bar but the previous one
    # never cleared the bar that applied to it, so patience is not satisfied
    assert detector.process(FRAME, speaking=False) is None


def test_patience_counts_frames_that_cleared_their_own_bar(detector_factory):
    # the same shape with both frames idle does fire — the guard above is
    # about the moving bar, not about patience itself
    detector = detector_factory(
        {"wake": [0.45, 0.45]},
        model="wake", threshold=0.35, threshold_speaking=0.55, patience=2,
    )
    assert detector.process(FRAME, speaking=False) is None
    assert detector.process(FRAME, speaking=False) == "wake"


def test_stop_model_wins_over_wake(detector_factory):
    detector = detector_factory(
        {"wake": [0.9], "stop": [0.9]}, model="wake", stop_model="stop"
    )
    assert detector.process(FRAME) == "stop"


def test_reset_rearms(detector_factory):
    detector = detector_factory({"wake": [0.9, 0.9]}, model="wake")
    assert detector.process(FRAME) == "wake"
    detector.reset()
    # reset() also drops the audio a window-scoring engine needs, and that
    # re-priming is the engine's business, not the trigger's; what this pins
    # is that the trigger is armed again without having dipped low first
    prime(detector)
    assert detector.process(FRAME) == "wake"


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

    An engine's own detect() may reject unconditionally while audio is
    playing; routing around that is why these adapters take raw scores and
    leave every threshold decision to BaseWakewordDetector.
    """
    detector = detector_factory(
        {"wake": [0.9], "stop": [0.9]},
        model="wake", stop_model="stop", threshold_speaking=0.7,
    )
    assert detector.process(FRAME, speaking=True) == "stop"
    detector = detector_factory({"wake": [0.9]}, model="wake", threshold_speaking=0.7)
    assert detector.process(FRAME, speaking=True) == "wake"


# -- the skip contract, against the base class directly -----------------


class _SkippingDetector(BaseWakewordDetector):
    """Scores only the frames it is told to, to exercise `_scores` -> None."""

    def __init__(self, config, frame_ms, scores):
        super().__init__(config, frame_ms)
        self._scripted = list(scores)

    def _scores(self, frame):
        score = self._scripted.pop(0)
        return None if score is None else {"wake": score}

    def _engine_reset(self):
        pass


def _skipping(scores, **kwargs):
    return _SkippingDetector(WakewordConfig(model="wake", **kwargs), 80, scores)


def test_a_skipped_frame_yields_no_detection():
    detector = _skipping([None, 0.9])
    assert detector.process(FRAME) is None
    assert detector.scored_last_frame is False
    assert detector.process(FRAME) == "wake"
    assert detector.scored_last_frame is True


def test_a_skipped_frame_leaves_the_history_untouched():
    """A skip must not read as the phrase ending, nor stand in for a frame.

    With patience 2, the skip between two high scores would break the window
    if it appended anything at all — a zero ends the phrase, a repeat would
    have let the first score alone satisfy it.
    """
    detector = _skipping([0.9, None, 0.9], patience=2, threshold=0.5)
    assert [detector.process(FRAME) for _ in range(3)] == [None, None, "wake"]


def test_a_skipped_frame_does_not_move_the_reported_score():
    detector = _skipping([0.42, None])
    detector.process(FRAME)
    detector.process(FRAME)
    assert detector.score("wake") == pytest.approx(0.42)
