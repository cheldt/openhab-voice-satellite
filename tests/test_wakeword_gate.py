"""The VAD gate, across every engine that supports it.

Three of these are the invariants violawake's PowerManager used to carry, moved
here because the gate replaced it and because they were only ever asserted for
one engine. The fourth group is what the PowerManager never had: a pre-roll,
without which the gate scores the wake phrase on spliced audio.

Silero is driven the same way tests/test_vad.py drives it — the real package,
with the instance's `probability` replaced — because the residual re-chunking
between an 80 ms frame and Silero's 512-sample window is part of what is being
tested.
"""

from __future__ import annotations

import numpy as np
import pytest

from .wakeword_stubs import (
    StubWakeDetector,
    StubWakeforgeRunner,
    make_detector,
    reset_stub_state,
)

FRAME = np.zeros(1280, dtype=np.int16)

# the gate is rejected on openwakeword at config load (its 2040 ms of context
# makes every opening a burst longer than a frame period); that rejection has
# its own test in test_config.py
GATED_ENGINES = ("violawake", "wakeforge")


def _engine_frames(engine):
    """Frames the engine actually scored, whichever stub is behind it."""
    instances = (
        StubWakeDetector.instances if engine == "violawake"
        else StubWakeforgeRunner.instances
    )
    return instances[0].frames


@pytest.fixture(params=GATED_ENGINES)
def gated(request, monkeypatch):
    """A detector with the gate on, plus a switch for what Silero hears."""
    reset_stub_state()
    engine = request.param
    speech = [False]

    def make(scripts, gate=None, **config_kwargs):
        detector = make_detector(
            engine, monkeypatch, scripts, model="wake",
            vad_gate={"enabled": True, **(gate or {})}, **config_kwargs,
        )
        monkeypatch.setattr(
            detector._gate._vad, "probability", lambda chunk: 1.0 if speech[0] else 0.0
        )
        return detector

    yield engine, speech, make
    reset_stub_state()


def test_gate_is_not_built_when_disabled(monkeypatch):
    reset_stub_state()
    detector = make_detector("violawake", monkeypatch, {"wake": [0.0]}, model="wake")
    assert detector._gate is None
    assert detector.scored_last_frame
    reset_stub_state()


def test_a_gated_frame_is_never_scored(gated):
    engine, speech, make = gated
    detector = make({"wake": [0.9]}, {"hangover_ms": 0})
    assert detector.process(FRAME) is None
    assert _engine_frames(engine) == []
    assert not detector.scored_last_frame


def test_a_gated_frame_still_reaches_the_tail_ring(gated):
    _, speech, make = gated
    detector = make({"wake": [0.0]}, {"hangover_ms": 0})
    frame = np.arange(1280, dtype=np.int16)
    assert detector.process(frame) is None
    # the mic still heard it, so a wake-audio dump has to contain it
    assert np.array_equal(detector.tail(1280 / 16000), frame)


def test_a_gated_frame_does_not_advance_the_patience_window(gated):
    _, speech, make = gated
    detector = make({"wake": [0.9, 0.9]}, {"hangover_ms": 0, "preroll_ms": 0}, patience=2)
    speech[0] = True
    assert detector.process(FRAME) is None   # 1 of 2
    speech[0] = False
    assert detector.process(FRAME) is None   # gated: no score, no progress
    speech[0] = True
    assert detector.process(FRAME) == "wake"  # 2 of 2, window unbroken


def test_the_hangover_keeps_scoring_after_speech_stops(gated):
    _, speech, make = gated
    detector = make({"wake": [0.0] * 6}, {"hangover_ms": 240, "preroll_ms": 0})
    speech[0] = True
    detector.process(FRAME)
    speech[0] = False
    # 240 ms of budget against 80 ms frames: two more frames, then shut
    assert detector.process(FRAME) is None and detector.scored_last_frame
    assert detector.process(FRAME) is None and detector.scored_last_frame
    detector.process(FRAME)
    assert not detector.scored_last_frame


def test_the_preroll_is_replayed_in_order_when_the_gate_opens(gated):
    engine, speech, make = gated
    detector = make({"wake": [0.0] * 8}, {"hangover_ms": 0, "preroll_ms": 160})
    buffered = [np.full(1280, i + 1, dtype=np.int16) for i in range(2)]
    for frame in buffered:
        detector.process(frame)
    assert _engine_frames(engine) == []
    speech[0] = True
    live = np.full(1280, 9, dtype=np.int16)
    detector.process(live)
    # the engine's streaming state is primed with the audio that preceded the
    # onset, in order, before the frame that gets to decide anything
    seen = _engine_frames(engine)
    assert [f[0] for f in seen] == [1, 2, 9]


def test_preroll_frames_do_not_reach_the_trigger(gated):
    _, speech, make = gated
    # every score is above threshold, so an observed pre-roll would satisfy a
    # patience window of 3 from a single process() call
    detector = make({"wake": [0.9] * 8}, {"hangover_ms": 0, "preroll_ms": 240}, patience=3)
    for _ in range(3):
        detector.process(FRAME)
    speech[0] = True
    assert detector.process(FRAME) is None, "pre-roll was observed by the trigger"
    assert detector.process(FRAME) is None
    assert detector.process(FRAME) == "wake"


def test_the_gate_never_closes_while_speaking(gated):
    engine, speech, make = gated
    detector = make({"wake": [0.0] * 4}, {"hangover_ms": 0})
    # silence as far as Silero is concerned, but our own output is audible:
    # barge-in and the stop word cannot pay onset-clipping latency
    detector.process(FRAME, speaking=True)
    assert detector.scored_last_frame
    assert len(_engine_frames(engine)) == 1


def test_the_gate_is_open_after_reset(gated):
    _, speech, make = gated
    # reset() runs at every detection and barge-in and empties the pre-roll,
    # so shutting here would leave the next opening with nothing to replay
    detector = make({"wake": [0.0] * 4})
    detector.process(FRAME)
    detector.reset()
    detector.process(FRAME)
    assert detector.scored_last_frame


def test_the_gate_fails_open_when_silero_raises(gated):
    engine, speech, make = gated
    detector = make({"wake": [0.9, 0.9]}, {"hangover_ms": 0})

    def boom(chunk):
        raise RuntimeError("silero exploded")

    detector._gate._vad.probability = boom
    # an exception here must cost CPU, not detections, and must not repeat
    assert detector.process(FRAME) == "wake"
    assert detector.process(FRAME) is None  # edge trigger, not the gate
    assert len(_engine_frames(engine)) == 2


def test_frames_are_rechunked_to_silero_s_window(gated):
    _, speech, make = gated
    detector = make({"wake": [0.0] * 4}, {"hangover_ms": 0})
    sizes = []
    detector._gate._vad.probability = lambda chunk: sizes.append(len(chunk)) or 0.0
    # pysilero raises on anything but exactly 512 samples, and an exception in
    # the monitor loop takes the whole detector down
    detector.process(FRAME)
    assert sizes == [512, 512]
