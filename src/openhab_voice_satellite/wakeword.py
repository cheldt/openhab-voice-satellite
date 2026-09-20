"""Wakeword detection: the engine-neutral contract, decision logic and factory.

Engines supply scores; the decision stays here. That split is deliberate — the
thresholds, the edge trigger and the patience window are what the rest of the
app is tuned against, and barge-in depends on a detection still being possible
while our own output is audible. An engine's own gating — cooldowns, RMS floors,
a hard reject during playback — would quietly change all three, so we take raw
scores from every engine and decide identically.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Protocol

import numpy as np

from .config import SAMPLE_RATE, Config, WakewordConfig
from .wakeword_buffer import Int16Ring

log = logging.getLogger(__name__)

RECENT_SCORES = 16  # per-model score history; must exceed the patience cap

# raw audio kept for tail() and for the engines that score a trailing window
# rather than a frame, matching openwakeword's own buffer length so wake-audio
# dumps look the same whichever engine produced the detection
TAIL_SECONDS = 10

WAKE = "wake"
STOP = "stop"


class WakewordProtocol(Protocol):
    """What the app needs from a detector, whichever engine backs it."""

    # score of the candidate behind the last WAKE the detector returned
    last_trigger_score: float | None
    # False when the engine declined to score the frame just passed to
    # `process` — see `_scores`. Always True for engines that score every
    # frame, so callers can read it with a `getattr(..., True)` default.
    scored_last_frame: bool

    def process(self, frame: np.ndarray, speaking: bool = False) -> str | None: ...

    def score(self, key: str = WAKE) -> float: ...

    def tail(self, seconds: float) -> np.ndarray: ...

    def reset(self) -> None: ...


class EdgeTrigger:
    """One model's score history and armed state.

    Edge-triggered: once the score crosses `threshold` the trigger disarms and
    only re-arms once the score falls below half of it, so one spoken wakeword
    yields one event even though scores stay high for several frames. A
    `patience` above 1 additionally requires the crossing to hold for that many
    consecutive observations, which rejects single-frame transients.

    Observing a score and deciding on it are separate calls because a detector
    with a stop model advances *every* model's history per frame but stops
    checking once one of them fires — a gap in the history would corrupt the
    patience window. `feed` is the shorthand for the single-model case.

    The patience window is a window of *verdicts*, not of scores: `observe`
    takes the bar that applied on that frame and records whether the score met
    it. The bar moves per frame — `threshold_speaking` while our own output is
    audible — so re-testing the stored scores against the current frame's bar
    would let scores that only cleared the raised bar be re-judged at the idle
    one on the first frame after playback, which is exactly the echo the raise
    exists to reject.

    This lives apart from the detectors so an offline evaluator can replay
    recorded scores through the rule the app actually runs, rather than a copy
    of it that drifts.
    """

    def __init__(self) -> None:
        self.scores: deque[float] = deque(maxlen=RECENT_SCORES)
        # whether each score met the threshold of its own frame
        self.passes: deque[bool] = deque(maxlen=RECENT_SCORES)
        self.armed = True

    def observe(self, score: float, threshold: float) -> None:
        """Advance the history by one observation without deciding anything."""
        score = float(score)
        self.scores.append(score)
        self.passes.append(score >= threshold)

    def fired(self, threshold: float, patience: int) -> bool:
        """Whether the newest observed score completes a detection.

        `threshold` is read for the re-arm hysteresis only; whether each frame
        cleared its own bar was already decided in `observe`.
        """
        if self.armed:
            window = list(self.passes)[-patience:]
            if len(window) == patience and all(window):
                self.armed = False
                return True
            return False
        if self.scores and self.scores[-1] < threshold / 2:
            self.armed = True
        return False

    def feed(self, score: float, threshold: float, patience: int) -> bool:
        """observe + fired, for callers tracking a single model."""
        self.observe(score, threshold)
        return self.fired(threshold, patience)

    @property
    def last(self) -> float:
        return float(self.scores[-1]) if self.scores else 0.0

    def reset(self) -> None:
        self.scores.clear()
        self.passes.clear()
        self.armed = True


class BaseWakewordDetector:
    """Turns model scores into at most one event per spoken phrase.

    The decision itself lives in `EdgeTrigger`, one per configured model.

    Subclasses implement `_scores` and `_engine_reset`.

    The raw-audio ring behind `tail()` lives here rather than in the engines,
    and `process` fills it before anything else runs. Engine-owned rings could
    not make that promise: openwakeword's is filled inside `predict()` and
    sized for its own melspectrogram, while this one also has to serve wake
    dumps and the engines that score a trailing window instead of a frame.
    """

    def __init__(self, config: WakewordConfig, frame_ms: int) -> None:
        # frame_ms is part of the engine contract rather than something the
        # base reads: an engine sizes its own cadence against it
        self._config = config
        keys = [WAKE] + ([STOP] if config.stop_model else [])
        self._triggers: dict[str, EdgeTrigger] = {key: EdgeTrigger() for key in keys}
        self._ring = Int16Ring(SAMPLE_RATE * TAIL_SECONDS)
        self.last_trigger_score: float | None = None
        self.scored_last_frame = True

    # -- engine hooks ---------------------------------------------------

    def _scores(self, frame: np.ndarray) -> dict[str, float] | None:
        """Score `frame` for every configured key, or None if it was skipped.

        A skipped frame leaves the score history untouched rather than
        appending a stale or zero score. Both would corrupt the decision: a
        repeat lets one high evaluation satisfy a multi-frame patience window
        and keeps the trigger from re-arming, and a zero reads as the phrase
        ending mid-window. Engines that cannot afford to score every frame
        (see `wakeword_livekit`) return None on the frames they skip, which
        makes `patience` count engine evaluations rather than mic frames.
        """
        raise NotImplementedError

    def _engine_reset(self) -> None:
        raise NotImplementedError

    def tail(self, seconds: float) -> np.ndarray:
        """Newest `seconds` of raw mic audio.

        The window predates the detection that prompted the call, which is
        what makes it useful; reset() clears it, so read before resetting.
        Copied because `Int16Ring.tail` can hand back a live view.
        """
        return self._ring.tail(int(seconds * SAMPLE_RATE)).copy()

    # -- decision -------------------------------------------------------

    def _threshold(self, key: str, speaking: bool) -> float:
        """The bar for `key` on this frame.

        `speaking=True` raises it (echo mitigation while our own output is
        audible) — raised, not disabled, because barge-in and the stop word
        both have to keep working through playback.
        """
        config = self._config
        if key == STOP:
            return (
                config.effective_stop_threshold_speaking
                if speaking
                else config.stop_threshold
            )
        return config.threshold_speaking if speaking else config.threshold

    def _check(self, key: str, threshold: float, patience: int) -> bool:
        return self._triggers[key].fired(threshold, patience)

    def process(self, frame: np.ndarray, speaking: bool = False) -> str | None:
        """Return 'wake' or 'stop' on detection, else None."""
        # the ring first, unconditionally: tail() must show what the mic heard
        # even on a frame the engine declines to score, and the window-scoring
        # engines read their input straight back out of it
        self._ring.extend(frame)
        scores = self._scores(frame)
        self.scored_last_frame = scores is not None
        if scores is None:
            return None
        result = self._decide(scores, speaking)
        if result == WAKE:
            self.last_trigger_score = self.score(WAKE)
        return result

    def _decide(self, scores: dict[str, float], speaking: bool) -> str | None:
        """Advance every trigger with this frame's scores and read the verdict."""
        # every model's history advances on every scored frame, even when an
        # earlier model already fired — a gap would corrupt the patience window
        for key, trigger in self._triggers.items():
            trigger.observe(scores[key], self._threshold(key, speaking))

        config = self._config
        if STOP in self._triggers:
            if self._check(STOP, self._threshold(STOP, speaking), config.stop_patience):
                return STOP
        if self._check(WAKE, self._threshold(WAKE, speaking), config.patience):
            return WAKE
        return None

    def score(self, key: str = WAKE) -> float:
        trigger = self._triggers.get(key) or self._triggers[WAKE]
        return trigger.last

    def reset(self) -> None:
        self._engine_reset()
        self._ring.clear()
        self.last_trigger_score = None
        self.scored_last_frame = True
        for trigger in self._triggers.values():
            trigger.reset()


def build_detector(config: Config) -> WakewordProtocol:
    """The detector for this config's engine selection.

    Imported lazily per engine so an uninstalled stack reports as one failed
    --check step rather than killing the process at import time.
    """
    frame_ms = config.audio.frame_ms
    if config.wakeword.engine == "livekit":
        from .wakeword_livekit import LivekitDetector

        return LivekitDetector(config.wakeword, frame_ms)
    from .wakeword_oww import OpenWakewordDetector

    return OpenWakewordDetector(config.wakeword, frame_ms)
