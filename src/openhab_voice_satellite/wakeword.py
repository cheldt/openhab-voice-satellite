"""openWakeWord wrapper (ONNX backend): wake model plus optional stop model."""

from __future__ import annotations

import logging
import pickle
from collections import deque
from pathlib import Path

import numpy as np

from .config import SAMPLE_RATE, WakewordConfig
from .wakeword_buffer import patch_preprocessor

log = logging.getLogger(__name__)

RECENT_SCORES = 16  # per-model score history; must exceed the patience cap


class WakewordDetector:
    """Feeds 80 ms int16 frames to openWakeWord and reports detections.

    Detection is edge-triggered: once a model crosses its threshold it must
    fall below half the threshold before it can fire again, so one spoken
    wakeword yields one event even though scores stay high for several frames.
    With `patience` above 1 the crossing must also hold for that many
    consecutive frames, which rejects single-frame transients.
    """

    def __init__(self, config: WakewordConfig) -> None:
        from openwakeword.model import Model

        self._config = config
        models = [config.model]
        self._stop_key: str | None = None
        if config.stop_model:
            models.append(config.stop_model)
        try:
            # ncpu=1: single-threaded ONNX sessions. The default lets ORT spawn
            # a spin-waiting pool per core, and the 80 ms inference cadence
            # never lets those workers park -> steady multi-core burn at idle.
            self._model = Model(wakeword_models=models, inference_framework="onnx", ncpu=1)
        except TypeError:
            log.warning("openwakeword lacks ncpu kwarg; upgrade to >=0.6.0 to bound CPU")
            self._model = Model(wakeword_models=models, inference_framework="onnx")
        self._ring = patch_preprocessor(self._model)
        # openwakeword builds .models by zipping paths with derived names in
        # order (model.py:143), so position maps back to our list — but only
        # while the names stay distinct and single-output
        keys = list(self._model.models.keys())
        self._check_keys(keys, models)
        self._wake_key = keys[0]
        if config.stop_model:
            self._stop_key = keys[1]
        self._armed: dict[str, bool] = {key: True for key in keys}
        self._recent: dict[str, deque] = {
            key: deque(maxlen=RECENT_SCORES) for key in keys
        }
        self._load_verifiers()
        log.info("wakeword models loaded: %s", keys)

    def _check_keys(self, keys: list[str], models: list[str]) -> None:
        """Fail at startup on the two ways the positional mapping breaks."""
        if len(keys) != len(models):
            raise ValueError(
                f"openwakeword collapsed {models} into {keys}: the model files share "
                "a basename, so they cannot be told apart — rename one"
            )
        multi = [k for k in keys if self._model.model_outputs.get(k, 1) != 1]
        if multi:
            raise ValueError(
                f"wakeword models {multi} have multiple outputs; openwakeword then "
                "keys predictions by class label instead of model name, which this "
                "wrapper does not support"
            )

    def _load_verifiers(self) -> None:
        """Attach optional per-speaker verifier models, keyed by model name.

        Loaded here rather than passed to Model(...) for two reasons: upstream
        rejects a verifier supplied for anything but the first model
        (model.py:187-194 compares cumulative counts inside the load loop), and
        unpickling is arbitrary code execution, so it belongs somewhere visible.
        """
        pairs = (
            (self._wake_key, self._config.verifier_model),
            (self._stop_key, self._config.stop_verifier_model),
        )
        for key, path in pairs:
            if key is None or not path:
                continue
            with open(path, "rb") as handle:
                self._model.custom_verifier_models[key] = pickle.load(handle)
            log.info("verifier model attached to %s: %s", key, Path(path).name)
        if self._model.custom_verifier_models:
            # the verifier REPLACES the base score (model.py:327), so every
            # threshold below is now read against a logistic probability
            self._model.custom_verifier_threshold = self._config.verifier_threshold
            log.warning(
                "verifier models active: thresholds now apply to verifier "
                "probabilities, not base wakeword scores — re-check them"
            )

    def reset(self) -> None:
        self._model.reset()
        for key in self._armed:
            self._armed[key] = True
            self._recent[key].clear()

    def tail(self, seconds: float) -> np.ndarray | None:
        """Newest `seconds` of raw mic audio, or None if the patch failed open.

        The window predates the detection that prompted the call, which is
        what makes it useful; reset() clears it, so read before resetting.
        """
        if self._ring is None:
            return None
        return self._ring.tail(int(seconds * SAMPLE_RATE)).copy()

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
        """Return 'wake' or 'stop' on detection, else None.

        `speaking=True` raises both thresholds (echo mitigation while our own
        output is audible).
        """
        prediction = self._model.predict(frame)
        # every model's history advances on every frame, even when an earlier
        # model already fired — a gap would corrupt the patience window
        for key, history in self._recent.items():
            history.append(float(prediction[key]))

        config = self._config
        if self._stop_key is not None:
            stop_threshold = (
                config.effective_stop_threshold_speaking
                if speaking
                else config.stop_threshold
            )
            if self._check(self._stop_key, stop_threshold, config.stop_patience):
                return "stop"
        threshold = config.threshold_speaking if speaking else config.threshold
        if self._check(self._wake_key, threshold, config.patience):
            return "wake"
        return None

    def score(self, key: str = "wake") -> float:
        model_key = self._stop_key if key == "stop" and self._stop_key else self._wake_key
        history = self._recent.get(model_key)
        return float(history[-1]) if history else 0.0
