"""openWakeWord engine (ONNX backend): wake model plus optional stop model."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

from .config import WakewordConfig
from .wakeword import STOP, WAKE, BaseWakewordDetector
from .wakeword_buffer import patch_preprocessor

log = logging.getLogger(__name__)


class OpenWakewordDetector(BaseWakewordDetector):
    """Feeds 80 ms int16 frames to openWakeWord and reports detections."""

    def __init__(self, config: WakewordConfig, frame_ms: int) -> None:
        from openwakeword.model import Model

        super().__init__(config, frame_ms)
        models = [config.model]
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
        # the patch's own ring serves openwakeword's melspectrogram; the ring
        # behind tail() belongs to BaseWakewordDetector. Kept only so a skipped
        # patch is visible at construction rather than as a mystery later.
        self._patched = patch_preprocessor(self._model) is not None
        # openwakeword builds .models by zipping paths with derived names in
        # order (model.py:143), so position maps back to our list — but only
        # while the names stay distinct and single-output
        keys = list(self._model.models.keys())
        self._check_keys(keys, models)
        # canonical key -> this engine's model name
        self._model_keys = {WAKE: keys[0]}
        if config.stop_model:
            self._model_keys[STOP] = keys[1]
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
            (self._model_keys.get(WAKE), self._config.verifier_model),
            (self._model_keys.get(STOP), self._config.stop_verifier_model),
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

    def _scores(self, frame: np.ndarray) -> dict[str, float]:
        prediction = self._model.predict(frame)
        return {key: float(prediction[name]) for key, name in self._model_keys.items()}

    def _engine_reset(self) -> None:
        self._model.reset()
