"""Wakeword detection: the engine-neutral contract, decision logic and factory.

Engines supply scores; the decision stays here. That split is deliberate — the
thresholds, the edge trigger and the patience window are what the rest of the
app is tuned against, and barge-in depends on a detection still being possible
while our own output is audible. An engine's own gating (violawake ships a
cooldown, an RMS floor and a hard reject during playback) would quietly change
all three, so we take raw scores from every engine and decide identically.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Protocol

import numpy as np

from .config import SAMPLE_RATE, Config, WakewordConfig, WakewordVadGateConfig
from .wakeword_buffer import Int16Ring

log = logging.getLogger(__name__)

RECENT_SCORES = 16  # per-model score history; must exceed the patience cap

# raw audio kept for tail(), matching openwakeword's own buffer length so
# wake-audio dumps look the same whichever engine produced the detection
TAIL_SECONDS = 10

WAKE = "wake"
STOP = "stop"


class WakewordProtocol(Protocol):
    """What the app needs from a detector, whichever engine backs it."""

    def process(self, frame: np.ndarray, speaking: bool = False) -> str | None: ...

    def score(self, key: str = WAKE) -> float: ...

    def tail(self, seconds: float) -> np.ndarray | None: ...

    def reset(self) -> None: ...


class EdgeTrigger:
    """One model's score history and armed state.

    Edge-triggered: once the score crosses `threshold` the trigger disarms and
    only re-arms once the score falls below half of it, so one spoken wakeword
    yields one event even though scores stay high for several frames. A
    `patience` above 1 additionally requires the crossing to hold for that many
    consecutive frames, which rejects single-frame transients.

    Observing a score and deciding on it are separate calls because a detector
    with a stop model advances *every* model's history per frame but stops
    checking once one of them fires — a gap in the history would corrupt the
    patience window. `feed` is the shorthand for the single-model case.

    This lives apart from the detectors so the offline evaluator can replay
    recorded scores through the rule the app actually runs, rather than a copy
    of it that drifts.
    """

    def __init__(self) -> None:
        self.scores: deque[float] = deque(maxlen=RECENT_SCORES)
        self.armed = True

    def observe(self, score: float) -> None:
        """Advance the history by one frame without deciding anything."""
        self.scores.append(float(score))

    def fired(self, threshold: float, patience: int) -> bool:
        """Whether the newest observed score completes a detection."""
        if self.armed:
            window = list(self.scores)[-patience:]
            if len(window) == patience and all(s >= threshold for s in window):
                self.armed = False
                return True
            return False
        if self.scores and self.scores[-1] < threshold / 2:
            self.armed = True
        return False

    def feed(self, score: float, threshold: float, patience: int) -> bool:
        """observe + fired, for callers tracking a single model."""
        self.observe(score)
        return self.fired(threshold, patience)

    @property
    def last(self) -> float:
        return float(self.scores[-1]) if self.scores else 0.0

    def reset(self) -> None:
        self.scores.clear()
        self.armed = True


class VadGate:
    """Silero deciding which frames are worth scoring.

    Not a duty cycle: the engines behind this splice rather than pause when a
    frame goes missing (see ENGINE_CONTEXT_MS), so the gate only ever closes
    after sustained silence, and it replays a full context of buffered audio
    when it opens. Those replayed frames prime the engine; they never reach a
    trigger, so one `process()` call still produces exactly one decision.

    Its Silero instance is its own. The app's `SpeechEndpointer` is reset once
    per utterance by the recorder and latches `speech_started`; sharing it
    would have a per-utterance state machine and an always-on one writing to
    the same recurrent state.
    """

    def __init__(self, config: WakewordVadGateConfig, preroll_ms: int) -> None:
        from .config import VadConfig
        from .vad import SpeechEndpointer

        # only `threshold` means anything to update(); the rest of VadConfig is
        # endpointing, which is a different job with a different cost of error
        self._vad = SpeechEndpointer(VadConfig(threshold=config.threshold))
        self._config = config
        self._preroll_samples = SAMPLE_RATE * preroll_ms // 1000
        self._hangover_samples = SAMPLE_RATE * config.hangover_ms // 1000
        self._live = True
        self._preroll: deque[np.ndarray] = deque()
        self._buffered = 0
        self._budget = self._hangover_samples

    def admit(self, frame: np.ndarray, speaking: bool) -> tuple[np.ndarray, ...]:
        """The frames to score: none, this one, or a pre-roll then this one."""
        if not self._live:
            return (frame,)
        bypass = speaking and self._config.bypass_while_speaking
        if bypass or self._speech(frame):
            # a speech frame is always scored; the hangover governs only how
            # much of the trailing silence keeps being scored after it
            self._budget = self._hangover_samples
            admit = True
        else:
            self._budget = max(0, self._budget - len(frame))
            admit = self._budget > 0
        if admit:
            admitted = (*self._preroll, frame)
            self._preroll.clear()
            self._buffered = 0
            return admitted
        self._preroll.append(frame)
        self._buffered += len(frame)
        while self._preroll and self._buffered - len(self._preroll[0]) >= self._preroll_samples:
            self._buffered -= len(self._preroll.popleft())
        return ()

    def _speech(self, frame: np.ndarray) -> bool:
        """Whether Silero heard speech, failing open and permanently on error."""
        try:
            return self._vad.update(frame)
        except Exception:
            log.exception("wakeword VAD gate failed; scoring every frame from here")
            self._live = False
            return True

    def reset(self) -> None:
        # open, not shut: reset() runs at every detection and barge-in and
        # empties the pre-roll, so the next opening would have nothing to
        # replay. Feed the engine continuously while its context refills.
        self._vad.reset()
        self._preroll.clear()
        self._buffered = 0
        self._budget = self._hangover_samples


class BaseWakewordDetector:
    """Turns per-frame model scores into at most one event per spoken phrase.

    The decision itself lives in `EdgeTrigger`, one per configured model.

    Subclasses implement `_scores` and `_engine_reset`.

    The raw-audio ring behind `tail()` lives here rather than in the engines,
    and `process` fills it before anything else runs. Engine-owned rings could
    not make that promise: openwakeword's is filled inside `predict()`, so any
    frame an engine declines to score would leave a hole in the audio a
    detection dump is supposed to show.
    """

    def __init__(self, config: WakewordConfig) -> None:
        self._config = config
        keys = [WAKE] + ([STOP] if config.stop_model else [])
        self._triggers: dict[str, EdgeTrigger] = {key: EdgeTrigger() for key in keys}
        self._ring = Int16Ring(SAMPLE_RATE * TAIL_SECONDS)
        self._gate = self._build_gate(config)
        # every consumer of score() needs to know whether it is looking at this
        # frame's score or the last scored frame's — see app.py's monitor loop
        self.scored_last_frame = True

    @staticmethod
    def _build_gate(config: WakewordConfig) -> VadGate | None:
        if not config.vad_gate.enabled:
            return None
        try:
            gate = VadGate(config.vad_gate, config.gate_preroll_ms)
        except Exception:
            # an unimportable pysilero or a failed session must cost CPU, not
            # detections: same posture as the ORT and preprocessor patches
            log.exception("wakeword VAD gate unavailable; scoring every frame")
            return None
        log.info(
            "wakeword VAD gate on: threshold %.2f, %d ms hangover, %d ms pre-roll",
            config.vad_gate.threshold, config.vad_gate.hangover_ms,
            config.gate_preroll_ms,
        )
        return gate

    # -- engine hooks ---------------------------------------------------

    def _scores(self, frame: np.ndarray) -> dict[str, float] | None:
        """Score `frame` for every configured key, or None if it was skipped.

        A skipped frame leaves the score history untouched rather than
        appending a zero, so a deliberate gap (frame skipping) cannot be
        mistaken for the phrase ending mid-patience-window.
        """
        raise NotImplementedError

    def _engine_reset(self) -> None:
        raise NotImplementedError

    def tail(self, seconds: float) -> np.ndarray | None:
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
        # even on a frame the engine declines to score
        self._ring.extend(frame)
        admitted = self._gate.admit(frame, speaking) if self._gate else (frame,)
        if not admitted:
            self.scored_last_frame = False
            return None
        # everything but the last is pre-roll: it primes the engine's streaming
        # state and is deliberately not observed, so one call stays one decision
        for warm in admitted[:-1]:
            self._scores(warm)
        scores = self._scores(admitted[-1])
        self.scored_last_frame = scores is not None
        if scores is None:
            return None
        return self._decide(scores, speaking)

    def _decide(self, scores: dict[str, float], speaking: bool) -> str | None:
        """Advance every trigger with this frame's scores and read the verdict."""
        # every model's history advances on every frame, even when an earlier
        # model already fired — a gap would corrupt the patience window
        for key, trigger in self._triggers.items():
            trigger.observe(scores[key])

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
        if self._gate is not None:
            self._gate.reset()
        self.scored_last_frame = True
        for trigger in self._triggers.values():
            trigger.reset()


def build_detector(config: Config) -> WakewordProtocol:
    """The detector for this config's engine selection.

    Imported lazily per engine so an uninstalled stack reports as one failed
    --check step rather than killing the process at import time.
    """
    if config.wakeword.engine == "violawake":
        from .wakeword_viola import ViolaWakeDetector

        return ViolaWakeDetector(config.wakeword, config.audio.frame_ms)
    if config.wakeword.engine == "wakeforge":
        from .wakeword_wakeforge import WakeforgeDetector

        return WakeforgeDetector(config.wakeword)
    from .wakeword_oww import OpenWakewordDetector

    return OpenWakewordDetector(config.wakeword)
