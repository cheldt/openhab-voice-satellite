"""Stage-2 verifier: wake deferral, accept/reject, and what it must not touch.

Run against every engine, because the second stage is engine-neutral: the same
mel-PCEN verifier rejects ~99 % of stage-1 triggers whichever architecture
produced them, measured on the same 5.48 h of speech.

The verifier itself is stubbed at the `_build_verifier`/`_verify` seam — its
scoring quality is the training pipeline's problem; these tests pin the
decision plumbing: a WAKE is withheld for the configured delay, then either
released or swallowed by the verifier's verdict, while STOP and the score
histories behave as if the verifier did not exist.
"""

from __future__ import annotations

import numpy as np
import pytest

from openhab_voice_satellite import wakeword

from .wakeword_stubs import ENGINES, make_detector, reset_stub_state

FRAME = np.zeros(1280, dtype=np.int16)

STAGE2 = {"model": "v.onnx", "mel_basis": "m.npy",
          "threshold": 0.5, "delay_ms": 300}


@pytest.fixture(params=ENGINES)
def verified_factory(request, monkeypatch):
    """Detector factory with the verifier seam stubbed to scripted scores."""
    reset_stub_state()
    scores: list[float] = []

    monkeypatch.setattr(
        wakeword.BaseWakewordDetector,
        "_build_verifier",
        staticmethod(
            lambda config: ("session", "frontend") if config.stage2.model else None
        ),
    )
    monkeypatch.setattr(
        wakeword.BaseWakewordDetector,
        "_verify",
        lambda self: scores.pop(0),
    )

    def make(scripts, verifier_scores, **config_kwargs):
        scores.clear()
        scores.extend(verifier_scores)
        # frame_ms=80 and delay_ms=300 -> ceil(300/80) = 4 frames of audio
        # gathered *past* the trigger frame, so the verdict lands on the 5th
        # process() call. The trigger frame is not one of the four: it is the
        # frame the phrase was still unfinished on, which is the whole reason
        # the delay exists, and the offline calibration measured its threshold
        # on a window ending 4 frames after the trigger.
        return make_detector(request.param, monkeypatch, scripts, **config_kwargs)

    yield make
    reset_stub_state()


def test_wake_is_deferred_then_released_on_accept(verified_factory):
    detector = verified_factory(
        {"wake": [0.9, 0.0, 0.0, 0.0, 0.0]}, [0.8],
        model="wake", stage2=STAGE2,
    )
    assert detector.process(FRAME) is None  # stage-1 fired; deferred
    for _ in range(3):  # frames 1-3 of the four post-trigger frames
        assert detector.process(FRAME) is None
    assert detector.process(FRAME) == "wake"  # 4th frame: verifier 0.8 >= 0.5


def test_the_verdict_lands_a_full_delay_past_the_trigger(verified_factory):
    """The trigger frame must not consume one of the delay frames.

    delay_ms/frame_ms frames of *further* audio is what every config comment
    promises and what ultiwake's replay scored the deployed threshold on;
    verifying a frame early truncates the phrase tail the delay exists to
    capture.
    """
    detector = verified_factory(
        {"wake": [0.9] + [0.0] * 8}, [0.8],
        model="wake", stage2=STAGE2,  # delay_ms 300, frame_ms 80 -> 4 frames
    )
    for _ in range(4):  # trigger frame + 3 post-trigger frames: still pending
        assert detector.process(FRAME) is None
    assert detector.process(FRAME) == "wake"


def test_a_delay_under_one_frame_still_gathers_one_frame(verified_factory):
    # delay_ms <= frame_ms used to verify on the trigger frame itself, scoring
    # exactly the truncated phrase the delay is there to avoid
    detector = verified_factory(
        {"wake": [0.9, 0.0, 0.0]}, [0.8],
        model="wake", stage2=dict(STAGE2, delay_ms=80),
    )
    assert detector.process(FRAME) is None  # trigger frame: nothing scored yet
    assert detector.process(FRAME) == "wake"  # one frame of settling audio


def test_wake_is_swallowed_on_reject(verified_factory):
    detector = verified_factory(
        {"wake": [0.9, 0.0, 0.0, 0.0, 0.0, 0.0]}, [0.2],
        model="wake", stage2=STAGE2,
    )
    assert all(detector.process(FRAME) is None for _ in range(6))
    assert detector.last_verifier_score == pytest.approx(0.2)


def test_stop_fires_during_a_pending_verification(verified_factory):
    # stopping playback late defeats its purpose; STOP is never deferred
    detector = verified_factory(
        {"wake": [0.9, 0.0], "stop": [0.0, 0.9]}, [0.8],
        model="wake", stop_model="stop", stage2=STAGE2,
    )
    assert detector.process(FRAME) is None  # wake deferred
    assert detector.process(FRAME) == "stop"


def test_second_trigger_keeps_the_first_countdown(verified_factory):
    # restarting the countdown would push the capture window past the phrase
    detector = verified_factory(
        {"wake": [0.9, 0.1, 0.9, 0.0, 0.0]}, [0.8],
        model="wake", stage2=STAGE2,
    )
    assert detector.process(FRAME) is None  # trigger, 4 frames to go
    assert detector.process(FRAME) is None  # 0.1 re-arms the edge
    assert detector.process(FRAME) is None  # re-trigger must not restart it
    assert detector.process(FRAME) is None
    assert detector.process(FRAME) == "wake"  # original schedule holds


def test_a_trigger_on_a_rejections_verdict_frame_is_not_swallowed(verified_factory):
    """A wake that lands where a rejection expires must get its own verdict.

    Rejections are common live — once a minute in a quiet room, several times
    a second while someone is talking — so a real wakeword whose stage-1
    crossing happens to fall on one used to be folded into the dying countdown
    and thrown away with it — and its edge trigger was already disarmed, so the
    rest of the phrase could not re-fire either. The word was simply lost.
    """
    # frame 4 is both the rejected candidate's verdict frame and a fresh
    # stage-1 crossing (0.05 re-armed the edge on frame 3)
    detector = verified_factory(
        {"wake": [0.9, 0.0, 0.0, 0.05, 0.95, 0.0, 0.0, 0.0, 0.0]},
        [0.2, 0.8],  # first candidate rejected, second accepted
        model="wake", stage2=STAGE2,
    )
    for _ in range(4):
        assert detector.process(FRAME) is None
    assert detector.process(FRAME) is None  # the rejection lands here
    # the peak spans the candidate's whole window, verdict frame included, so
    # the colliding crossing is inside it — a diagnostic score, not a decision
    assert detector.last_rejection == pytest.approx(0.95)
    for _ in range(3):  # the new candidate's own four frames
        assert detector.process(FRAME) is None
    assert detector.process(FRAME) == "wake"
    assert detector.last_trigger_score == pytest.approx(0.95)


def test_a_trigger_on_an_accepts_verdict_frame_is_not_verified_twice(verified_factory):
    # the accept just reported belongs to this phrase; re-verifying its tail
    # would emit a second wake a few frames later
    detector = verified_factory(
        {"wake": [0.9, 0.0, 0.0, 0.05, 0.95, 0.0, 0.0, 0.0, 0.0]},
        [0.8],  # one verifier score: a second call would IndexError
        model="wake", stage2=STAGE2,
    )
    for _ in range(4):
        assert detector.process(FRAME) is None
    assert detector.process(FRAME) == "wake"
    assert all(detector.process(FRAME) is None for _ in range(4))


def test_reset_clears_a_pending_verification(verified_factory):
    detector = verified_factory(
        {"wake": [0.9, 0.0, 0.0, 0.0]}, [0.8],
        model="wake", stage2=STAGE2,
    )
    assert detector.process(FRAME) is None
    detector.reset()
    # the pending wake died with the reset; nothing fires afterwards
    assert all(detector.process(FRAME) is None for _ in range(4))
    assert detector.last_verifier_score is None


def test_no_verifier_means_no_deferral(verified_factory):
    detector = verified_factory({"wake": [0.9]}, [], model="wake")
    assert detector.process(FRAME) == "wake"


def test_stop_wins_when_the_verdict_lands_on_the_same_frame(verified_factory):
    # a verifier accept must not displace a STOP decided on the verdict frame:
    # the app would treat it as a wake barge-in and restart LISTENING
    detector = verified_factory(
        {"wake": [0.9, 0.0, 0.0, 0.0, 0.0], "stop": [0.0, 0.0, 0.0, 0.0, 0.9]},
        [0.9],
        model="wake", stop_model="stop", stage2=STAGE2,
    )
    assert detector.process(FRAME) is None  # wake deferred, 4 frames to go
    for _ in range(3):
        assert detector.process(FRAME) is None
    assert detector.process(FRAME) == "stop"  # collision: stop, not wake


def test_the_trigger_score_is_the_candidates_peak(verified_factory):
    # stage 1 keeps climbing past the trigger frame and has decayed again by
    # the verdict frame; the reported score must be the peak, not either end
    detector = verified_factory(
        {"wake": [0.9, 0.97, 0.4, 0.1, 0.05]}, [0.8],
        model="wake", stage2=STAGE2,
    )
    for _ in range(4):
        assert detector.process(FRAME) is None
    assert detector.process(FRAME) == "wake"
    assert detector.last_trigger_score == pytest.approx(0.97)
    assert detector.last_verifier_score == pytest.approx(0.8)
    assert detector.last_rejection is None


def test_a_rejection_reports_the_peak_for_one_frame(verified_factory):
    # the near-miss dump gate reads last_rejection on the verdict frame; by
    # then the live stage-1 score has decayed below any useful floor
    detector = verified_factory(
        {"wake": [0.9, 0.97, 0.4, 0.1, 0.0, 0.0]}, [0.2],
        model="wake", stage2=STAGE2,
    )
    for _ in range(5):
        assert detector.process(FRAME) is None
    assert detector.last_rejection == pytest.approx(0.97)
    assert detector.last_trigger_score == pytest.approx(0.97)
    detector.process(FRAME)  # any later frame clears the one-frame flag
    assert detector.last_rejection is None


def test_no_verifier_still_reports_the_trigger_score(verified_factory):
    detector = verified_factory({"wake": [0.9]}, [], model="wake")
    assert detector.process(FRAME) == "wake"
    assert detector.last_trigger_score == pytest.approx(0.9)
    assert detector.last_verifier_score is None


def _stub_ort(monkeypatch, inputs, outputs):
    """Install a fake onnxruntime whose session reports these io signatures."""
    import sys
    import types

    class FakeOptions:
        def __init__(self):
            self.intra_op_num_threads = 0
            self.inter_op_num_threads = 0
            self.execution_mode = None

        def add_session_config_entry(self, key, value):
            pass

    class FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        def get_inputs(self):
            return [types.SimpleNamespace(name=n, shape=s) for n, s in inputs]

        def get_outputs(self):
            return [types.SimpleNamespace(name=n, shape=s) for n, s in outputs]

    monkeypatch.setitem(sys.modules, "onnxruntime", types.SimpleNamespace(
        SessionOptions=FakeOptions,
        ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL=0),
        InferenceSession=FakeSession,
    ))


GOOD_INPUT = [("features", ["batch", 40, 151])]


def _build(monkeypatch, inputs, outputs, mel_basis="m.npy"):
    _stub_ort(monkeypatch, inputs, outputs)
    from openhab_voice_satellite.config import WakewordConfig
    from openhab_voice_satellite.wakeword import BaseWakewordDetector

    monkeypatch.setattr(
        "openhab_voice_satellite.verifier_mel.MelPcenFrontend",
        lambda path: "frontend",
    )
    return BaseWakewordDetector._build_verifier(
        WakewordConfig(model="wake", stage2=dict(STAGE2, mel_basis=mel_basis))
    )


def test_build_verifier_rejects_a_wrong_input_contract(monkeypatch):
    # --check feeds silence and never reaches _verify(); the input contract
    # must therefore fail at load, not at the first live wake
    with pytest.raises(ValueError, match="features"):
        _build(monkeypatch, [("input", ["batch", 40, 151])], [("y", ["batch", 1])])


@pytest.mark.parametrize("outputs, why", [
    ([("probs", ["batch", 2])], "a two-class softmax head"),
    ([("logits", ["batch", 1]), ("embedding", ["batch", 96])], "an extra output"),
])
def test_build_verifier_rejects_a_wrong_output_contract(monkeypatch, outputs, why):
    """`_verify` reads flatten()[0], so a wide head is silently wrong.

    Nothing crashes with one: the load gate and --check both pass and every
    wake is then scored against the wrong tensor element — the negative class,
    or a raw logit — so the satellite either goes deaf or accepts everything.
    """
    with pytest.raises(ValueError, match="single score"):
        _build(monkeypatch, GOOD_INPUT, outputs)


@pytest.mark.parametrize("shape", [["batch", 1], ["batch"]])
def test_build_verifier_accepts_a_single_score_head(monkeypatch, shape):
    session, frontend = _build(monkeypatch, GOOD_INPUT, [("score", shape)])
    assert session is not None and frontend == "frontend"
