"""Application wiring: owns the state, the interrupt monitor and the pipeline task."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import AsyncExitStack
from pathlib import Path

import aiohttp
import numpy as np

from .audio.broadcast import AudioBroadcaster
from .audio.earcons import Earcons
from .audio.io import audio_io, verify_links
from .audio.sink import AudioSink
from .config import Config
from .deepgram import DeepgramClient, DeepgramSpeaker, DeepgramTranscriber
from .fallback import FallbackSpeaker, FallbackTranscriber, LazySpeaker
from .gemini import GeminiClient, GeminiSpeaker, GeminiTranscriber
from .openhab import OpenHABClient, make_session
from .piper_tts import PiperSpeaker
from .pipeline import Pipeline, SpeakerProtocol, TranscriberProtocol
from .state import Event, State
from .stt import Transcriber
from .tts import Speaker
from .vad import SpeechEndpointer
from .wakeword import WakewordDetector

log = logging.getLogger(__name__)

# wakeword score that triggers duck-and-confirm during playback
DUCK_PRETHRESHOLD = 0.35
DUCK_HOLD_FRAMES = 13  # ~1 s of 80 ms frames

MIC_STALL_WARN_S = 10.0  # no frames for this long -> loud warning
HEARTBEAT_S = 10.0  # capture-health DEBUG line interval


class _CaptureHealth:
    """Frame-rate/RMS bookkeeping behind the heartbeat + degraded-capture logs."""

    def __init__(self, expected_fps: float) -> None:
        self._expected_fps = expected_fps
        self._frames = 0
        self._rms = 0
        self._score = 0.0
        self._start = time.monotonic()

    def restart(self) -> None:
        """Reset the window clock (after a stall, so the gap isn't counted)."""
        self._start = time.monotonic()

    def observe(self, frame: np.ndarray, score: float) -> None:
        self._frames += 1
        self._rms = max(self._rms, int(np.sqrt(np.mean(frame.astype(np.float64) ** 2))))
        self._score = max(self._score, score)
        now = time.monotonic()
        if now - self._start < HEARTBEAT_S:
            return
        log.debug(
            "monitor: %d frames in %.1fs, peak rms=%d, peak wake score=%.3f",
            self._frames, now - self._start, self._rms, self._score,
        )
        expected = self._expected_fps * (now - self._start)
        if self._frames < 0.8 * expected:
            log.warning(
                "degraded capture: %d of %d expected mic frames in %.0fs "
                "— wakeword detection will be unreliable",
                self._frames, int(expected), now - self._start,
            )
        self._frames = 0
        self._rms = 0
        self._score = 0.0
        self._start = now


class _DuckController:
    """Duck-and-confirm: a pre-threshold score during playback lowers the
    volume so the follow-up frames reach the detector more cleanly."""

    def __init__(self) -> None:
        self._frames_left = 0

    def update(self, speaking: bool, score: float, sink: AudioSink) -> None:
        if speaking:
            if score >= DUCK_PRETHRESHOLD and self._frames_left == 0:
                sink.duck(0.2)
                self._frames_left = DUCK_HOLD_FRAMES
            elif self._frames_left > 0:
                self._frames_left -= 1
                if self._frames_left == 0:
                    sink.unduck()
        elif self._frames_left:
            self.release(sink)

    def release(self, sink: AudioSink) -> None:
        self._frames_left = 0
        sink.unduck()


def _build_speaker(config: Config, sink: AudioSink, base_dir: Path) -> SpeakerProtocol:
    """The local TTS engine (or its lazy stand-in when a cloud engine is primary)."""
    if config.tts.engine == "piper":
        return PiperSpeaker(config.piper, config.tts, sink, base_dir)
    if config.tts.engine == "kokoro":
        return Speaker(config.kokoro, config.tts, sink, base_dir)
    # cloud engine primary: kokoro stays the fallback but loads on first use
    return LazySpeaker(lambda: Speaker(config.kokoro, config.tts, sink, base_dir))


async def _build_engines(
    config: Config,
    stack: AsyncExitStack,
    sink: AudioSink,
    local_transcriber: TranscriberProtocol,
    local_speaker: SpeakerProtocol,
) -> tuple[TranscriberProtocol, SpeakerProtocol]:
    """Wrap the local engines with cloud primaries per config.

    The local engines stay loaded as the fallback for every cloud path.
    """
    transcriber = local_transcriber
    speaker = local_speaker
    cloud_engines = {config.stt.engine, config.tts.engine} - {"local", "kokoro", "piper"}
    if cloud_engines:
        # own session: openHAB's may have TLS verification disabled
        cloud_session = await stack.enter_async_context(aiohttp.ClientSession())
    if "gemini" in cloud_engines:
        gemini = GeminiClient(config.gemini, cloud_session)
        if config.stt.engine == "gemini":
            transcriber = FallbackTranscriber(
                GeminiTranscriber(gemini, config.stt, config.tts.default_language),
                local_transcriber,
                label="gemini",
            )
        if config.tts.engine == "gemini":
            speaker = FallbackSpeaker(
                GeminiSpeaker(gemini, config.tts, sink), local_speaker, label="gemini"
            )
    if "deepgram" in cloud_engines:
        deepgram = DeepgramClient(config.deepgram, cloud_session)
        if config.stt.engine == "deepgram":
            transcriber = FallbackTranscriber(
                DeepgramTranscriber(deepgram, config.stt, config.tts.default_language),
                local_transcriber,
                label="deepgram",
            )
        if config.tts.engine == "deepgram":
            speaker = FallbackSpeaker(
                DeepgramSpeaker(deepgram, config.tts, sink), local_speaker, label="deepgram"
            )
    return transcriber, speaker


class App:
    def __init__(self, config: Config, base_dir: Path | None = None) -> None:
        self._config = config
        self._base_dir = base_dir or Path.cwd()
        self.state = State.IDLE
        self._pipeline_task: asyncio.Task | None = None

    def _set_state(self, state: State) -> None:
        log.debug("state: %s -> %s", self.state.name, state.name)
        self.state = state

    async def run(self) -> None:
        config = self._config
        log.info("loading models...")
        detector = WakewordDetector(config.wakeword)
        endpointer = SpeechEndpointer(config.vad)
        transcriber = Transcriber(config.stt, config.tts.default_language)

        async with AsyncExitStack() as stack:
            source, sink = await stack.enter_async_context(audio_io(config.audio))
            earcons = Earcons(config.earcons, sink, self._base_dir)
            speaker = _build_speaker(config, sink, self._base_dir)

            broadcaster = AudioBroadcaster(source)
            wake_queue = broadcaster.subscribe()
            broadcaster.start()

            session = await stack.enter_async_context(make_session(config.openhab))
            openhab = OpenHABClient(config.openhab, session)

            final_transcriber, final_speaker = await _build_engines(
                config, stack, sink, transcriber, speaker
            )
            log.info("engines: stt=%s tts=%s", config.stt.engine, config.tts.engine)

            pipeline = Pipeline(
                config=config,
                broadcaster=broadcaster,
                endpointer=endpointer,
                transcriber=final_transcriber,
                openhab=openhab,
                speaker=final_speaker,
                earcons=earcons,
                set_state=self._set_state,
            )

            log.info("ready — say the wakeword (%s)", config.wakeword.model)
            link_check = asyncio.create_task(
                verify_links(source.target, sink.target), name="verify-links"
            )
            try:
                await self._interrupt_monitor(wake_queue, detector, pipeline, sink, earcons)
            finally:
                link_check.cancel()
                await broadcaster.stop()

    def _start_pipeline(
        self, pipeline: Pipeline, earcons: Earcons, play_wake_earcon: bool = True
    ) -> None:
        async def _run() -> None:
            try:
                event = await pipeline.run_interaction(play_wake_earcon)
            finally:
                self._set_state(State.IDLE)
                self._pipeline_task = None
            if event is not Event.ERROR:
                await earcons.play("idle")

        self._pipeline_task = asyncio.create_task(_run(), name="interaction")

    async def _cancel_pipeline(self, sink: AudioSink) -> bool:
        task = self._pipeline_task
        if task is None:
            return False
        was_speaking = self.state is State.SPEAKING
        sink.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        self._set_state(State.IDLE)
        self._pipeline_task = None
        log.info("interaction cancelled (barge-in)")
        return was_speaking

    async def _interrupt_monitor(
        self,
        wake_queue: asyncio.Queue,
        detector: WakewordDetector,
        pipeline: Pipeline,
        sink: AudioSink,
        earcons: Earcons,
    ) -> None:
        """Always-on wakeword loop; starts or cancels the interaction task."""
        audio = self._config.audio
        health = _CaptureHealth(audio.sample_rate / audio.frame_samples)
        duck = _DuckController()
        while True:
            try:
                frame = await asyncio.wait_for(wake_queue.get(), timeout=MIC_STALL_WARN_S)
            except asyncio.TimeoutError:
                log.warning(
                    "no mic frames for %.0fs — capture stream stalled?", MIC_STALL_WARN_S
                )
                health.restart()
                continue
            if frame is None:
                log.info("audio source closed, monitor exiting")
                return

            speaking = self.state in (State.THINKING, State.SPEAKING)
            detection = detector.process(frame, speaking=speaking)
            score = detector.score("wake")
            health.observe(frame, score)
            duck.update(self.state is State.SPEAKING, score, sink)

            if detection is None:
                continue

            if self.state is State.IDLE and detection == "wake":
                log.info("wakeword detected (score %.2f)", score)
                detector.reset()
                self._start_pipeline(pipeline, earcons)
            elif self.state in (State.LISTENING, State.THINKING, State.SPEAKING):
                # wakeword or stop-word during an interaction = barge-in
                was_speaking = await self._cancel_pipeline(sink)
                duck.release(sink)
                detector.reset()
                if (
                    detection == "wake"
                    and was_speaking
                    and self._config.barge_in.resume_listening
                ):
                    self._start_pipeline(pipeline, earcons, play_wake_earcon=True)
                else:
                    await earcons.play("idle")
