"""ViolaWake engine: a TemporalCNN head over openWakeWord's embedding backbone.

Only `WakeDetector.process()` is used — it returns the raw model score with
none of violawake's own gating applied. `detect()` is deliberately avoided:
its fourth gate is an unconditional reject whenever playback is active, which
would take out barge-in and the stop word, and its cooldown and RMS floor
would double up on the edge trigger and patience this app already applies.
See BaseWakewordDetector for where the decision actually happens.

Each configured model needs its own WakeDetector, and each of those runs its
own melspectrogram + embedding backbone — openWakeWord shares one across
models, violawake does not. Setting `stop_model` therefore roughly doubles the
per-frame cost on this engine.
"""

from __future__ import annotations

import logging

import numpy as np

from .config import SAMPLE_RATE, WakewordConfig
from .violawake_ort import patch_onnx_threads, single_threaded_sessions
from .wakeword import STOP, WAKE, BaseWakewordDetector
from .wakeword_buffer import Int16Ring

log = logging.getLogger(__name__)

# raw audio kept for tail(); matches openwakeword's own buffer so wake-audio
# dumps look the same whichever engine produced the detection
TAIL_SECONDS = 10


class ViolaWakeDetector(BaseWakewordDetector):
    """Feeds int16 frames to one violawake model per key and reports detections."""

    def __init__(self, config: WakewordConfig, frame_ms: int) -> None:
        from violawake_sdk import WakeDetector

        super().__init__(config)
        # before the first session is built: violawake would otherwise size
        # ORT's intra-op pool to the core count and spin-wait between frames
        patch_onnx_threads()
        models = {WAKE: config.model}
        if config.stop_model:
            models[STOP] = config.stop_model
        # every session this builds is forced single-threaded, wherever
        # violawake happens to construct it, and the block reports what it bound
        with single_threaded_sessions():
            self._engines = {
                key: WakeDetector(
                    model=model,
                    # the policy these feed is never consulted; process()
                    # bypasses it. Passed so a stray detect() call would not
                    # misbehave.
                    threshold=config.threshold,
                    cooldown_s=0.0,
                    backend="onnx",
                    confirm_count=1,
                )
                for key, model in models.items()
            }
        self._ring = Int16Ring(SAMPLE_RATE * TAIL_SECONDS)
        self._profiler = self._build_profiler(config, frame_ms)
        self._power = self._build_power_manager(config)
        self._adapted: float | None = None
        log.info("wakeword models loaded (violawake): %s", list(models.values()))

    @staticmethod
    def _build_profiler(config: WakewordConfig, frame_ms: int):
        adaptive = config.viola.adaptive
        if not adaptive.enabled:
            return None
        from violawake_sdk import NoiseProfiler

        profiler = NoiseProfiler(
            base_threshold=config.threshold,
            noise_window_s=adaptive.noise_window_s,
            min_threshold=adaptive.min_threshold,
            max_threshold=adaptive.max_threshold,
            snr_boost_db=adaptive.snr_boost_db,
            snr_penalty_db=adaptive.snr_penalty_db,
            # upstream defaults to 50 (its own 20 ms frames); at our frame size
            # that would make noise_window_s several times shorter than asked
            frames_per_second=1000 / frame_ms,
        )
        log.info(
            "violawake adaptive threshold on: %.2f-%.2f around a %.0fs noise window",
            adaptive.min_threshold,
            adaptive.max_threshold,
            adaptive.noise_window_s,
        )
        return profiler

    @staticmethod
    def _build_power_manager(config: WakewordConfig):
        power = config.viola.power
        if not power.enabled:
            return None
        from violawake_sdk import PowerManager

        if power.duty_cycle_n > 1:
            log.warning(
                "violawake duty_cycle_n=%d drops frames mid-stream; the model "
                "sees a discontinuity and recall suffers",
                power.duty_cycle_n,
            )
        return PowerManager(
            duty_cycle_n=power.duty_cycle_n,
            silence_rms=power.silence_rms,
            activity_threshold=power.activity_threshold,
            active_window_s=power.active_window_s,
        )

    def _scores(self, frame: np.ndarray) -> dict[str, float] | None:
        # the ring feeds tail(), so it records what the mic heard even on the
        # frames the power manager declines to score
        self._ring.extend(frame)
        if self._power is not None or self._profiler is not None:
            # both expect the int16-scale float array violawake builds internally
            pcm = frame.astype(np.float32)
            if self._power is not None and not self._power.should_process(pcm):
                return None
            if self._profiler is not None:
                self._adapted = float(self._profiler.update(pcm))
        return {key: float(engine.process(frame)) for key, engine in self._engines.items()}

    def _threshold(self, key: str, speaking: bool) -> float:
        if key != WAKE or self._adapted is None:
            return super()._threshold(key, speaking)
        threshold = self._adapted
        if speaking:
            # keep the configured echo margin on top of whatever the noise
            # profiler settled on
            threshold += self._config.threshold_speaking - self._config.threshold
        return min(max(threshold, 0.0), 1.0)

    def _engine_reset(self) -> None:
        for engine in self._engines.values():
            engine.reset()
        self._ring.clear()

    def tail(self, seconds: float) -> np.ndarray | None:
        return self._ring.tail(int(seconds * SAMPLE_RATE)).copy()
