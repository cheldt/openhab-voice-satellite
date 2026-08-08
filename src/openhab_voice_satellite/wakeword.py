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

from .config import Config, WakewordConfig

log = logging.getLogger(__name__)

RECENT_SCORES = 16  # per-model score history; must exceed the patience cap

WAKE = "wake"
STOP = "stop"


class WakewordProtocol(Protocol):
    """What the app needs from a detector, whichever engine backs it."""

    def process(self, frame: np.ndarray, speaking: bool = False) -> str | None: ...

    def score(self, key: str = WAKE) -> float: ...

    def tail(self, seconds: float) -> np.ndarray | None: ...

    def reset(self) -> None: ...


class BaseWakewordDetector:
    """Turns per-frame model scores into at most one event per spoken phrase.

    Detection is edge-triggered: once a model crosses its threshold it must
    fall below half the threshold before it can fire again, so one spoken
    wakeword yields one event even though scores stay high for several frames.
    With `patience` above 1 the crossing must also hold for that many
    consecutive frames, which rejects single-frame transients.

    Subclasses implement `_scores` and `_engine_reset`; `tail` is optional.
    """

    def __init__(self, config: WakewordConfig) -> None:
        self._config = config
        keys = [WAKE] + ([STOP] if config.stop_model else [])
        self._armed: dict[str, bool] = {key: True for key in keys}
        self._recent: dict[str, deque] = {
            key: deque(maxlen=RECENT_SCORES) for key in keys
        }

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
        """Newest `seconds` of raw mic audio, or None if the engine has none.

        The window predates the detection that prompted the call, which is
        what makes it useful; reset() clears it, so read before resetting.
        """
        return None

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
        recent = self._recent[key]
        score = recent[-1]
        if self._armed[key]:
            window = list(recent)[-patience:]
            if len(window) == patience and all(s >= threshold for s in window):
                self._armed[key] = False
                return True
            return False
        if score < threshold / 2:
            self._armed[key] = True
        return False

    def process(self, frame: np.ndarray, speaking: bool = False) -> str | None:
        """Return 'wake' or 'stop' on detection, else None."""
        scores = self._scores(frame)
        if scores is None:
            return None
        # every model's history advances on every frame, even when an earlier
        # model already fired — a gap would corrupt the patience window
        for key, history in self._recent.items():
            history.append(float(scores[key]))

        config = self._config
        if STOP in self._recent:
            if self._check(STOP, self._threshold(STOP, speaking), config.stop_patience):
                return STOP
        if self._check(WAKE, self._threshold(WAKE, speaking), config.patience):
            return WAKE
        return None

    def score(self, key: str = WAKE) -> float:
        history = self._recent.get(key) or self._recent[WAKE]
        return float(history[-1]) if history else 0.0

    def reset(self) -> None:
        self._engine_reset()
        for key in self._armed:
            self._armed[key] = True
            self._recent[key].clear()


def build_detector(config: Config) -> WakewordProtocol:
    """The detector for this config's engine selection.

    Imported lazily per engine so an uninstalled stack reports as one failed
    --check step rather than killing the process at import time.
    """
    if config.wakeword.engine == "violawake":
        from .wakeword_viola import ViolaWakeDetector

        return ViolaWakeDetector(config.wakeword, config.audio.frame_ms)
    from .wakeword_oww import OpenWakewordDetector

    return OpenWakewordDetector(config.wakeword)
