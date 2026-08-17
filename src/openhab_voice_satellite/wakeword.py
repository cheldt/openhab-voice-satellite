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

# raw audio kept for tail(), matching openwakeword's own buffer length so
# wake-audio dumps look the same whichever engine produced the detection
TAIL_SECONDS = 10

# the stage-2 verifier scores exactly the 1.5 s window its training used
VERIFIER_CLIP_SAMPLES = int(1.5 * SAMPLE_RATE)

WAKE = "wake"
STOP = "stop"


def _session_options(ort):
    """One thread, no spinning: the options every ORT session here gets.

    ORT's default sizes the intra-op pool to the core count and those workers
    spin-wait between runs. At an 80 ms inference cadence they never park, so
    the pool burns whole cores while the satellite sits idle.
    """
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # belt and braces: even a one-thread pool spins between runs by default
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return options


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


class BaseWakewordDetector:
    """Turns per-frame model scores into at most one event per spoken phrase.

    The decision itself lives in `EdgeTrigger`, one per configured model.

    Subclasses implement `_scores` and `_engine_reset`.

    The raw-audio ring behind `tail()` lives here rather than in the engines,
    and `process` fills it before anything else runs. Engine-owned rings could
    not make that promise: openwakeword's is filled inside `predict()` and
    sized for its own melspectrogram, while this one has to serve both a
    detection dump and the stage-2 verifier's 1.5 s window.
    """

    def __init__(self, config: WakewordConfig, frame_ms: int) -> None:
        self._config = config
        keys = [WAKE] + ([STOP] if config.stop_model else [])
        self._triggers: dict[str, EdgeTrigger] = {key: EdgeTrigger() for key in keys}
        self._ring = Int16Ring(SAMPLE_RATE * TAIL_SECONDS)
        self._verifier = self._build_verifier(config)
        self._verify_delay_frames = max(1, -(-config.stage2.delay_ms // frame_ms))
        self._verify_countdown: int | None = None
        self.last_verifier_score: float | None = None

    @staticmethod
    def _build_verifier(config: WakewordConfig):
        """The stage-2 verifier (session, frontend), or None if unconfigured.

        The verifier ONNX must never go through an engine's own loader: those
        route architectures by input shape and can silently misread the
        (batch, 40, 151) verifier as a model of their own kind. A plain ORT
        session is the contract, with the same explicit single-thread options
        every other session here gets.
        """
        stage2 = config.stage2
        if not stage2.model:
            return None
        import onnxruntime as ort

        from .verifier_mel import MelPcenFrontend

        session = ort.InferenceSession(
            stage2.model,
            sess_options=_session_options(ort),
            providers=["CPUExecutionProvider"],
        )
        log.info(
            "wakeword stage-2 verifier loaded: %s (threshold %.2f, delay %d ms)",
            stage2.model, stage2.threshold, stage2.delay_ms,
        )
        return session, MelPcenFrontend(stage2.mel_basis)

    # -- engine hooks ---------------------------------------------------

    def _scores(self, frame: np.ndarray) -> dict[str, float]:
        """Score `frame` for every configured key."""
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
        # the ring first: tail() must show what the mic heard, and the verifier
        # reads its 1.5 s window straight out of it
        self._ring.extend(frame)
        result = self._decide(self._scores(frame), speaking)
        if self._verifier is None:
            return result
        return self._stage2(result)

    def _stage2(self, result: str | None) -> str | None:
        """Hold each WAKE until the verifier has a finished phrase to score.

        Stage 1 crosses its threshold before the phrase ends, so the verifier
        waits `delay_ms` of further audio and then scores the last 1.5 s from
        the ring. STOP is never deferred — stopping playback late defeats the
        purpose, and the stop model has no verifier anyway.
        """
        if result == WAKE:
            # a second stage-1 trigger while one is pending keeps the first
            # countdown — restarting it would push the window past the phrase
            if self._verify_countdown is None:
                self._verify_countdown = self._verify_delay_frames
            result = None
        if self._verify_countdown is not None:
            self._verify_countdown -= 1
            if self._verify_countdown <= 0:
                self._verify_countdown = None
                score = self._verify()
                self.last_verifier_score = score
                if score >= self._config.stage2.threshold:
                    return WAKE
                log.info("wake candidate rejected by verifier (score %.3f)", score)
        return result

    def _verify(self) -> float:
        session, frontend = self._verifier
        pcm = self._ring.tail(VERIFIER_CLIP_SAMPLES)
        if len(pcm) < VERIFIER_CLIP_SAMPLES:
            pcm = np.pad(pcm, (VERIFIER_CLIP_SAMPLES - len(pcm), 0))
        features = frontend(pcm.astype(np.float32) / 32768.0)[np.newaxis]
        return float(session.run(None, {"features": features})[0].flatten()[0])

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
        self._verify_countdown = None
        self.last_verifier_score = None
        for trigger in self._triggers.values():
            trigger.reset()


def build_detector(config: Config) -> WakewordProtocol:
    """The detector for this config's engine selection.

    Imported lazily so an uninstalled openwakeword reports as one failed
    --check step rather than killing the process at import time.
    """
    from .wakeword_oww import OpenWakewordDetector

    return OpenWakewordDetector(config.wakeword, config.audio.frame_ms)
