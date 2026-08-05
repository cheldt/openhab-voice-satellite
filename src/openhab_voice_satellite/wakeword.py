"""openWakeWord wrapper (ONNX backend): wake model plus optional stop model."""

from __future__ import annotations

import logging

import numpy as np

from .config import WakewordConfig

log = logging.getLogger(__name__)


class WakewordDetector:
    """Feeds 80 ms int16 frames to openWakeWord and reports detections.

    Detection is edge-triggered: once a model crosses its threshold it must
    fall below half the threshold before it can fire again, so one spoken
    wakeword yields one event even though scores stay high for several frames.
    """

    def __init__(self, config: WakewordConfig) -> None:
        from openwakeword.model import Model

        self._config = config
        models = [config.model]
        self._stop_key: str | None = None
        if config.stop_model:
            models.append(config.stop_model)
        self._model = Model(wakeword_models=models, inference_framework="onnx")
        keys = list(self._model.models.keys())
        self._wake_key = keys[0]
        if config.stop_model:
            self._stop_key = keys[1]
        self._armed: dict[str, bool] = {key: True for key in keys}
        log.info("wakeword models loaded: %s", keys)

    def reset(self) -> None:
        self._model.reset()
        for key in self._armed:
            self._armed[key] = True

    def _check(self, key: str, score: float, threshold: float) -> bool:
        if self._armed[key] and score >= threshold:
            self._armed[key] = False
            return True
        if not self._armed[key] and score < threshold / 2:
            self._armed[key] = True
        return False

    def process(self, frame: np.ndarray, speaking: bool = False) -> str | None:
        """Return 'wake' or 'stop' on detection, else None.

        `speaking=True` raises the wake threshold (echo mitigation while TTS
        output is audible).
        """
        prediction = self._model.predict(frame)
        threshold = (
            self._config.threshold_speaking if speaking else self._config.threshold
        )
        if self._stop_key is not None and self._check(
            self._stop_key, prediction[self._stop_key], self._config.stop_threshold
        ):
            return "stop"
        if self._check(self._wake_key, prediction[self._wake_key], threshold):
            return "wake"
        return None

    def score(self, key: str = "wake") -> float:
        model_key = self._stop_key if key == "stop" and self._stop_key else self._wake_key
        buf = self._model.prediction_buffer.get(model_key)
        return float(buf[-1]) if buf is not None and len(buf) else 0.0
