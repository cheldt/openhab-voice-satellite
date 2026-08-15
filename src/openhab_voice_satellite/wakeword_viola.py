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
from .violawake_ort import (
    _session_options,
    patch_onnx_threads,
    single_threaded_sessions,
)
from .wakeword import STOP, WAKE, BaseWakewordDetector

log = logging.getLogger(__name__)

# the stage-2 verifier scores exactly the 1.5 s window its training used
VERIFIER_CLIP_SAMPLES = int(1.5 * SAMPLE_RATE)


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
        # outside the block: the verifier passes explicit single-thread
        # options itself, and its scipy import would otherwise trip the
        # block's OS-thread watchdog with BLAS pool threads that are not ORT's
        self._verifier = self._build_verifier(config)
        self._verify_delay_frames = max(
            1, -(-config.viola.verifier.delay_ms // frame_ms)
        )
        self._verify_countdown: int | None = None
        self.last_verifier_score: float | None = None
        self._profiler = self._build_profiler(config, frame_ms)
        self._power = self._build_power_manager(config)
        self._adapted: float | None = None
        log.info("wakeword models loaded (violawake): %s", list(models.values()))

    @staticmethod
    def _build_verifier(config: WakewordConfig):
        """The stage-2 verifier (session, frontend), or None if unconfigured.

        The verifier ONNX must never go through violawake's WakeDetector —
        it routes architectures by input rank and would silently misread the
        (batch, 40, 151) verifier as a temporal embedding model. A plain ORT
        session is the contract, with the same explicit single-thread options
        the stage-1 sessions get.
        """
        verifier = config.viola.verifier
        if not verifier.model:
            return None
        import onnxruntime as ort

        from .verifier_mel import MelPcenFrontend

        session = ort.InferenceSession(
            verifier.model,
            sess_options=_session_options(ort),
            providers=["CPUExecutionProvider"],
        )
        frontend = MelPcenFrontend(verifier.mel_basis)
        log.info(
            "wakeword verifier loaded: %s (threshold %.2f, delay %d ms)",
            verifier.model,
            verifier.threshold,
            verifier.delay_ms,
        )
        return session, frontend

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
        # the tail ring is filled by BaseWakewordDetector.process before this
        # runs, so the frames the power manager declines to score still reach it
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

    def process(self, frame: np.ndarray, speaking: bool = False) -> str | None:
        """Defer each WAKE through the stage-2 verifier when one is loaded.

        Stage 1 crosses its threshold before the phrase is finished, so the
        verifier waits `delay_ms` of further audio and then scores the last
        1.5 s from the ring. STOP is never deferred — stopping playback late
        defeats its purpose, and the stop model has no verifier anyway. The
        countdown ticks on every frame, including ones the power manager
        declines to score: the ring records them, so the audio the verifier
        needs is there either way.
        """
        # super().process runs unconditionally first: every model's score
        # history must advance on every frame (see BaseWakewordDetector),
        # and STOP must keep working while a wake verification is pending
        result = super().process(frame, speaking)
        if self._verifier is None:
            return result
        if result == WAKE:
            # a second stage-1 trigger while one is pending keeps the first
            # countdown — restarting it would push the capture window past
            # the phrase
            if self._verify_countdown is None:
                self._verify_countdown = self._verify_delay_frames
            result = None
        if self._verify_countdown is not None:
            self._verify_countdown -= 1
            if self._verify_countdown <= 0:
                self._verify_countdown = None
                score = self._verify()
                self.last_verifier_score = score
                if score >= self._config.viola.verifier.threshold:
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

    def _engine_reset(self) -> None:
        for engine in self._engines.values():
            engine.reset()
        self._verify_countdown = None
        self.last_verifier_score = None
