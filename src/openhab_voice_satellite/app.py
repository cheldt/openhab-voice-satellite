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
            if config.tts.engine == "piper":
                speaker: SpeakerProtocol = PiperSpeaker(
                    config.piper, config.tts, sink, self._base_dir
                )
            elif config.tts.engine == "kokoro":
                speaker = Speaker(config.kokoro, config.tts, sink, self._base_dir)
            else:
                # cloud engine primary: kokoro stays the fallback but loads on first use
                speaker = LazySpeaker(
                    lambda: Speaker(config.kokoro, config.tts, sink, self._base_dir)
                )

            broadcaster = AudioBroadcaster(source)
            wake_queue = broadcaster.subscribe()
            broadcaster.start()

            session = await stack.enter_async_context(make_session(config.openhab))
            openhab = OpenHABClient(config.openhab, session)

            # local engines above stay loaded as fallback for the cloud path
            final_transcriber: TranscriberProtocol = transcriber
            final_speaker: SpeakerProtocol = speaker
            cloud_engines = {config.stt.engine, config.tts.engine} - {"local", "kokoro", "piper"}
            if cloud_engines:
                # own session: openHAB's may have TLS verification disabled
                cloud_session = await stack.enter_async_context(aiohttp.ClientSession())
            if "gemini" in cloud_engines:
                gemini = GeminiClient(config.gemini, cloud_session)
                if config.stt.engine == "gemini":
                    final_transcriber = FallbackTranscriber(
                        GeminiTranscriber(gemini, config.stt, config.tts.default_language),
                        transcriber,
                        label="gemini",
                    )
                if config.tts.engine == "gemini":
                    final_speaker = FallbackSpeaker(
                        GeminiSpeaker(gemini, config.tts, sink),
                        speaker,
                        label="gemini",
                    )
            if "deepgram" in cloud_engines:
                deepgram = DeepgramClient(config.deepgram, cloud_session)
                if config.stt.engine == "deepgram":
                    final_transcriber = FallbackTranscriber(
                        DeepgramTranscriber(deepgram, config.stt, config.tts.default_language),
                        transcriber,
                        label="deepgram",
                    )
                if config.tts.engine == "deepgram":
                    final_speaker = FallbackSpeaker(
                        DeepgramSpeaker(deepgram, config.tts, sink),
                        speaker,
                        label="deepgram",
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
        duck_frames_left = 0
        beat_frames = 0
        beat_rms = 0
        beat_score = 0.0
        beat_start = time.monotonic()
        audio = self._config.audio
        expected_fps = audio.sample_rate / audio.frame_samples
        while True:
            try:
                frame = await asyncio.wait_for(wake_queue.get(), timeout=MIC_STALL_WARN_S)
            except asyncio.TimeoutError:
                log.warning(
                    "no mic frames for %.0fs — capture stream stalled?", MIC_STALL_WARN_S
                )
                beat_start = time.monotonic()
                continue
            if frame is None:
                log.info("audio source closed, monitor exiting")
                return

            beat_frames += 1
            beat_rms = max(beat_rms, int(np.sqrt(np.mean(frame.astype(np.float64) ** 2))))
            now = time.monotonic()
            if now - beat_start >= HEARTBEAT_S:
                log.debug(
                    "monitor: %d frames in %.1fs, peak rms=%d, peak wake score=%.3f",
                    beat_frames, now - beat_start, beat_rms, beat_score,
                )
                expected = expected_fps * (now - beat_start)
                if beat_frames < 0.8 * expected:
                    log.warning(
                        "degraded capture: %d of %d expected mic frames in %.0fs "
                        "— wakeword detection will be unreliable",
                        beat_frames, int(expected), now - beat_start,
                    )
                beat_frames = 0
                beat_rms = 0
                beat_score = 0.0
                beat_start = now

            speaking = self.state in (State.THINKING, State.SPEAKING)
            detection = detector.process(frame, speaking=speaking)
            beat_score = max(beat_score, detector.score("wake"))

            # duck-and-confirm: pre-threshold score during playback lowers the
            # volume so the follow-up frames reach the detector more cleanly
            if self.state is State.SPEAKING:
                score = detector.score("wake")
                if score >= DUCK_PRETHRESHOLD and duck_frames_left == 0:
                    sink.duck(0.2)
                    duck_frames_left = DUCK_HOLD_FRAMES
                elif duck_frames_left > 0:
                    duck_frames_left -= 1
                    if duck_frames_left == 0:
                        sink.unduck()
            elif duck_frames_left:
                duck_frames_left = 0
                sink.unduck()

            if detection is None:
                continue

            if self.state is State.IDLE and detection == "wake":
                log.info("wakeword detected (score %.2f)", detector.score("wake"))
                detector.reset()
                self._start_pipeline(pipeline, earcons)
            elif self.state in (State.LISTENING, State.THINKING, State.SPEAKING):
                # wakeword or stop-word during an interaction = barge-in
                was_speaking = await self._cancel_pipeline(sink)
                sink.unduck()
                duck_frames_left = 0
                detector.reset()
                if (
                    detection == "wake"
                    and was_speaking
                    and self._config.barge_in.resume_listening
                ):
                    self._start_pipeline(pipeline, earcons, play_wake_earcon=True)
                else:
                    await earcons.play("idle")
