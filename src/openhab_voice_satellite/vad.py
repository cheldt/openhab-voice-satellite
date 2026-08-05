"""Silero VAD wrapper with trailing-silence endpointing."""

from __future__ import annotations

import logging

import numpy as np

from .config import VadConfig

log = logging.getLogger(__name__)

# pysilero-vad operates on 512-sample chunks at 16 kHz (32 ms)
VAD_CHUNK = 512


class SpeechEndpointer:
    """Consumes arbitrary-size int16 frames, tracks speech/silence timing.

    State per utterance: feed frames with `update()`; it returns
    'speech' | 'silence' per frame and exposes `endpoint_reached` /
    `speech_started` for the recorder loop.
    """

    def __init__(self, config: VadConfig, sample_rate: int = 16000) -> None:
        from pysilero_vad import SileroVoiceActivityDetector

        self._vad = SileroVoiceActivityDetector()
        self._config = config
        self._sample_rate = sample_rate
        self._residual = np.empty(0, dtype=np.int16)
        self.reset()

    def reset(self) -> None:
        self._vad.reset()
        self._residual = np.empty(0, dtype=np.int16)
        self.speech_started = False
        self._silence_samples = 0
        self._total_samples = 0

    def probability(self, chunk: np.ndarray) -> float:
        return float(self._vad(chunk.tobytes()))

    def update(self, frame: np.ndarray) -> bool:
        """Feed one frame; returns True if any chunk in it was speech."""
        self._total_samples += len(frame)
        buf = frame if len(self._residual) == 0 else np.concatenate([self._residual, frame])
        n_chunks = len(buf) // VAD_CHUNK
        self._residual = buf[n_chunks * VAD_CHUNK:]
        had_speech = False
        for i in range(n_chunks):
            chunk = buf[i * VAD_CHUNK:(i + 1) * VAD_CHUNK]
            if self.probability(chunk) >= self._config.threshold:
                had_speech = True
                self.speech_started = True
                self._silence_samples = 0
            else:
                self._silence_samples += VAD_CHUNK
        return had_speech

    @property
    def endpoint_reached(self) -> bool:
        if not self.speech_started:
            return False
        silence_ms = self._silence_samples * 1000 / self._sample_rate
        return silence_ms >= self._config.silence_ms

    @property
    def elapsed_s(self) -> float:
        return self._total_samples / self._sample_rate
