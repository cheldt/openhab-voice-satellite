"""Application wiring: owns the state, the interrupt monitor and the pipeline task."""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from pathlib import Path

import aiohttp

from .audio.broadcast import AudioBroadcaster
from .audio.earcons import Earcons
from .audio.sink import SounddeviceSink
from .audio.source import SounddeviceSource
from .config import Config
from .deepgram import DeepgramClient, DeepgramSpeaker, DeepgramTranscriber
from .fallback import FallbackSpeaker, FallbackTranscriber
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
        endpointer = SpeechEndpointer(config.vad, config.audio.sample_rate)
        transcriber = Transcriber(config.stt, config.tts.default_language)

        source = SounddeviceSource(
            sample_rate=config.audio.sample_rate,
            frame_samples=config.audio.frame_samples,
            device=config.audio.input_device,
        )
        sink = SounddeviceSink(
            device=config.audio.output_device,
            lead_in_ms=config.audio.output_lead_in_ms,
        )
        earcons = Earcons(config.earcons, sink, self._base_dir)
        if config.tts.engine == "piper":
            speaker: SpeakerProtocol = PiperSpeaker(config.piper, config.tts, sink, self._base_dir)
        else:
            # kokoro; also stays loaded as fallback for the cloud engines
            speaker = Speaker(config.kokoro, config.tts, sink, self._base_dir)

        broadcaster = AudioBroadcaster(source)
        wake_queue = broadcaster.subscribe()
        broadcaster.start()

        async with AsyncExitStack() as stack:
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
                        GeminiSpeaker(gemini, config.gemini, config.tts, sink),
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
                        DeepgramSpeaker(deepgram, config.deepgram, config.tts, sink),
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
                sink=sink,
                earcons=earcons,
                set_state=self._set_state,
            )

            log.info("ready — say the wakeword (%s)", config.wakeword.model)
            try:
                await self._interrupt_monitor(wake_queue, detector, pipeline, sink, earcons)
            finally:
                await broadcaster.stop()
                source.close()

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

    async def _cancel_pipeline(self, sink: SounddeviceSink) -> bool:
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
        sink: SounddeviceSink,
        earcons: Earcons,
    ) -> None:
        """Always-on wakeword loop; starts or cancels the interaction task."""
        duck_frames_left = 0
        while True:
            frame = await wake_queue.get()
            if frame is None:
                log.info("audio source closed, monitor exiting")
                return

            speaking = self.state in (State.THINKING, State.SPEAKING)
            detection = detector.process(frame, speaking=speaking)

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
