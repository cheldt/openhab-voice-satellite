"""livekit engine specifics: the trailing window, the hop, and startup checks.

The decision logic this engine feeds is covered once for every engine in
test_wakeword.py. What is left here is what makes livekit different from
openwakeword: it scores a window rather than a frame, so it is muted until one
exists, it skips frames rather than scoring every one, and it names its own
classifier keys instead of inheriting them from filenames.
"""

from __future__ import annotations

import numpy as np
import pytest

from openhab_voice_satellite.config import LivekitConfig, WakewordConfig
from openhab_voice_satellite.wakeword import STOP, WAKE

from .wakeword_stubs import (
    LIVEKIT_MODEL,
    LIVEKIT_STOP_MODEL,
    StubLivekitModel,
    install_livekit,
    reset_stub_state,
)

FRAME = np.zeros(1280, dtype=np.int16)
# frames of audio before a 2 s window exists, at 80 ms a frame
PRIMING_FRAMES = 25


@pytest.fixture(autouse=True)
def _clean_stubs():
    reset_stub_state()
    yield
    reset_stub_state()


def make(monkeypatch, scripts=None, hop_frames=1, **config_kwargs):
    """A livekit detector with an empty ring, as the app would build it."""
    install_livekit(monkeypatch)
    from openhab_voice_satellite.wakeword_livekit import LivekitDetector

    config_kwargs.setdefault("model", LIVEKIT_MODEL)
    detector = LivekitDetector(
        WakewordConfig(
            engine="livekit",
            livekit=LivekitConfig(hop_frames=hop_frames),
            **config_kwargs,
        ),
        80,
    )
    # after construction, so the startup probe does not eat the script
    StubLivekitModel.scripts = {k: list(v) for k, v in (scripts or {}).items()}
    return detector, StubLivekitModel.instances[-1]


# -- the window --------------------------------------------------------


def test_muted_until_the_window_fills(monkeypatch):
    detector, model = make(monkeypatch)
    model.windows.clear()
    for _ in range(PRIMING_FRAMES - 1):
        assert detector.process(FRAME) is None
        assert detector.scored_last_frame is False
    assert model.windows == []  # not one evaluation yet
    detector.process(FRAME)
    assert detector.scored_last_frame is True
    assert len(model.windows) == 1


def test_window_is_two_seconds_ending_at_the_current_frame(monkeypatch):
    from openhab_voice_satellite.wakeword_livekit import WINDOW_SAMPLES

    detector, model = make(monkeypatch)
    model.windows.clear()
    fed = []
    for i in range(PRIMING_FRAMES):
        frame = np.full(1280, i + 1, dtype=np.int16)
        fed.append(frame)
        detector.process(frame)

    window = model.windows[-1]
    assert window.dtype == np.int16
    assert len(window) == WINDOW_SAMPLES
    # the newest samples are this frame's, not a frame behind
    assert np.array_equal(window, np.concatenate(fed)[-WINDOW_SAMPLES:])


def test_reset_remutes_until_the_window_refills(monkeypatch):
    detector, model = make(monkeypatch)
    for _ in range(PRIMING_FRAMES):
        detector.process(FRAME)
    assert detector.scored_last_frame is True

    detector.reset()
    model.windows.clear()
    for _ in range(PRIMING_FRAMES - 1):
        assert detector.process(FRAME) is None
        assert detector.scored_last_frame is False
    assert model.windows == []
    detector.process(FRAME)
    assert detector.scored_last_frame is True


# -- the hop -----------------------------------------------------------


def prime(detector):
    """Fill the window without scoring, so the next process() is evaluation 1.

    Warming it with real frames instead would burn an evaluation (and a script
    entry) on the frame the ring happens to fill, putting every count below
    one frame out of step with what the test is describing.
    """
    from openhab_voice_satellite.wakeword_livekit import WINDOW_SAMPLES

    detector._ring.extend(np.zeros(WINDOW_SAMPLES, dtype=np.int16))


def test_hop_scores_every_nth_frame_and_reports_the_skip(monkeypatch):
    detector, model = make(monkeypatch, hop_frames=3)
    prime(detector)
    model.windows.clear()
    scored = []
    for _ in range(9):
        detector.process(FRAME)
        scored.append(detector.scored_last_frame)
    assert scored == [True, False, False] * 3
    assert len(model.windows) == 3


def test_patience_counts_evaluations_not_frames(monkeypatch):
    detector, _ = make(
        monkeypatch, {"wake": [0.9, 0.9]}, hop_frames=3, patience=2, threshold=0.5
    )
    prime(detector)
    # two evaluations are needed, and they are three frames apart, so the
    # detection lands on the fourth frame rather than the second
    results = [detector.process(FRAME) for _ in range(4)]
    assert results == [None, None, None, WAKE]


def test_a_skipped_frame_cannot_complete_the_patience_window(monkeypatch):
    """The regression that holding the last score across the hop would cause.

    With patience 2 and hop 3, repeating one high score over the skipped
    frames would satisfy the window off a single evaluation. Skipping leaves
    the history untouched, so the second evaluation's low score is what the
    window actually sees, and nothing fires.
    """
    detector, _ = make(
        monkeypatch, {"wake": [0.9, 0.1]}, hop_frames=3, patience=2, threshold=0.5
    )
    prime(detector)
    assert [detector.process(FRAME) for _ in range(6)] == [None] * 6


def test_a_skipped_frame_leaves_the_score_untouched(monkeypatch):
    detector, _ = make(monkeypatch, {"wake": [0.42]}, hop_frames=3)
    prime(detector)
    detector.process(FRAME)
    assert detector.score(WAKE) == pytest.approx(0.42)
    for _ in range(2):  # skipped frames must not push a stale or zero score
        detector.process(FRAME)
        assert detector.score(WAKE) == pytest.approx(0.42)


# -- keys and startup validation ---------------------------------------


def test_keys_are_named_at_load_not_derived_from_filenames(monkeypatch):
    _, model = make(monkeypatch, stop_model=LIVEKIT_STOP_MODEL)
    assert model.classifiers == [WAKE, STOP]


def test_models_sharing_a_basename_do_not_collapse(monkeypatch):
    """openwakeword raises on this; livekit cannot hit it at all.

    Its keys come from the name we pass to load_model, not from the file, so
    two models with the same basename stay distinguishable.
    """
    _, model = make(
        monkeypatch, model="models/a/x.onnx", stop_model="models/b/x.onnx"
    )
    assert model.classifiers == [WAKE, STOP]


def test_the_zero_sentinel_is_rejected_at_construction(monkeypatch):
    StubLivekitModel.idle_score = 0.0
    with pytest.raises(ValueError, match="32000 samples"):
        make(monkeypatch)


def test_a_score_that_is_not_a_probability_is_rejected(monkeypatch):
    StubLivekitModel.idle_score = 3.0
    with pytest.raises(ValueError, match="not a probability"):
        make(monkeypatch)


def test_a_missing_key_fails_at_construction(monkeypatch):
    install_livekit(monkeypatch)
    monkeypatch.setattr(
        StubLivekitModel, "predict", lambda self, audio: {WAKE: 0.01}
    )
    from openhab_voice_satellite.wakeword_livekit import LivekitDetector

    with pytest.raises(ValueError, match="did not score"):
        LivekitDetector(
            WakewordConfig(
                engine="livekit", model=LIVEKIT_MODEL, stop_model=LIVEKIT_STOP_MODEL
            ),
            80,
        )


# -- ORT session binding -----------------------------------------------


def test_sessions_are_bound_during_construction_and_restored_after(monkeypatch):
    import onnxruntime as ort

    original = ort.InferenceSession
    _, model = make(monkeypatch)
    # the wrapper owned the constructor while livekit was building its models
    assert model.ort_during_init is not original
    # and handed it back, so faster-whisper and piper are unaffected
    assert ort.InferenceSession is original
