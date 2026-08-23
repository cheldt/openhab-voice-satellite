"""Application wiring: owns the state, the interrupt monitor and the pipeline task."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import AsyncExitStack
from pathlib import Path

import aiohttp
import numpy as np

from .audio.broadcast import AudioBroadcaster, drain_stale
from .audio.earcons import Earcons
from .audio.gst_source import CaptureStats
from .audio.io import audio_io, verify_links
from .audio.sink import AudioSink
from .audio.source import AudioSource
from .audio.wav import rms, write_wav
from .config import SAMPLE_RATE, Config
from .deepgram import DeepgramClient, DeepgramSpeaker, DeepgramTranscriber
from .fallback import FallbackSpeaker, FallbackTranscriber, LazySpeaker
from .gemini import GeminiClient, GeminiSpeaker, GeminiTranscriber
from .openhab import OpenHABClient, make_session
from .piper_tts import PiperSpeaker
from .pipeline import Pipeline, SpeakerProtocol, TranscriberProtocol
from .state import Event, State
from .stt import Transcriber
from .vad import SpeechEndpointer
from .wakeword import WakewordProtocol, build_detector

log = logging.getLogger(__name__)

# wakeword score that triggers duck-and-confirm during playback
DUCK_PRETHRESHOLD = 0.35
DUCK_HOLD_FRAMES = 13  # ~1 s of 80 ms frames

MIC_STALL_WARN_S = 10.0  # no frames for this long -> loud warning
HEARTBEAT_S = 10.0  # capture-health DEBUG line interval


WAKE_DUMP_PREROLL_S = 2.5  # audio kept before a detection, incl. the wakeword


class CaptureClosedError(RuntimeError):
    """The capture stream ended mid-run (bus ERROR/EOS, e.g. PipeWire node loss).

    Deliberately fatal: in-process recovery would need set_state() from the
    sync bus handler (which deadlocks, see gst_common.install_sync_handler),
    and a wedged PipeWire stream rarely recovers its scheduling anyway. The
    process exits non-zero so the systemd unit (Restart=on-failure) restarts
    it with a fresh graph connection.
    """


def _log_wake_detection(detector: WakewordProtocol, score: float) -> None:
    """Log the candidate's stage-1 peak, not this frame's decayed score.

    With a verifier the detection lands `delay_ms` past the peak that fired
    it; the verifier's own score goes into the same line, since accepts are
    otherwise invisible.
    """
    trigger_score = detector.last_trigger_score
    trigger_score = score if trigger_score is None else trigger_score
    if detector.last_verifier_score is not None:
        log.info(
            "wakeword detected (score %.2f, verifier %.3f)",
            trigger_score, detector.last_verifier_score,
        )
    else:
        log.info("wakeword detected (score %.2f)", trigger_score)


def _wake_score_floor() -> float | None:
    """Parse $OVS_DUMP_WAKE_SCORE once, at monitor start.

    Parsing it per frame would let a typoed value raise ValueError inside
    the monitor loop and kill the process outside the CaptureClosedError
    exit path; a bad value here costs a warning, not the satellite.
    """
    raw = os.environ.get("OVS_DUMP_WAKE_SCORE")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        log.warning(
            "$OVS_DUMP_WAKE_SCORE=%r is not a number — near-miss dumps disabled", raw
        )
        return None


def _dump_wake_audio(
    detector: WakewordProtocol,
    detection: str | None,
    score: float,
    state: State,
    score_floor: float | None,
) -> None:
    """Write the audio around a detection to $OVS_DUMP_WAKE for field debugging.

    Unlike the utterance dump this captures what actually fired the detector,
    which is the only way to collect real false accepts. $OVS_DUMP_WAKE_SCORE
    additionally catches near misses — the frames that almost triggered — and
    verifier rejections, gated on the candidate's stage-1 peak because the
    current frame's score has decayed by the time the verdict lands. Those
    rejections are the hard negatives the retraining loop feeds on.
    """
    dump_dir = os.environ.get("OVS_DUMP_WAKE")
    if not dump_dir:
        return
    if detection is None:
        if score_floor is None:
            return
        rejected = detector.last_rejection
        if rejected is not None and rejected >= score_floor:
            label, score = "rejected", rejected
        elif score >= score_floor:
            label = "near"
        else:
            return
    else:
        label = detection
        if detection == "wake" and detector.last_trigger_score is not None:
            score = detector.last_trigger_score
    path = Path(dump_dir) / f"{label}-{score:.2f}-{state.name}-{time.strftime('%H%M%S')}.wav"
    pcm = detector.tail(WAKE_DUMP_PREROLL_S)
    if not len(pcm):
        return
    try:
        write_wav(path, pcm, SAMPLE_RATE)
        log.info("wake audio dumped: %s", path)
    except OSError:
        log.exception("wake audio dump failed")


class _CaptureHealth:
    """Frame-rate/RMS bookkeeping behind the heartbeat + degraded-capture logs."""

    def __init__(self, expected_fps: float, capture: CaptureStats | None = None) -> None:
        self._expected_fps = expected_fps
        self._frames = 0
        self._rms = 0
        self._score = 0.0
        self._dropped = 0  # queue evictions seen at the start of this window
        self._capture = capture or CaptureStats()
        self._start = time.monotonic()

    def restart(self, capture: CaptureStats | None = None) -> None:
        """Reset the window clock (after a stall, so the gap isn't counted).

        The capture baseline moves with it: a window that is not counted must
        not lend its buffers to the next one.
        """
        if capture is not None:
            self._capture = capture
        self._start = time.monotonic()

    def observe(
        self,
        frame: np.ndarray,
        score: float,
        dropped: int = 0,
        capture: CaptureStats | None = None,
    ) -> None:
        self._frames += 1
        self._rms = max(self._rms, rms(frame))
        self._score = max(self._score, score)
        now = time.monotonic()
        if now - self._start < HEARTBEAT_S:
            return
        evicted = dropped - self._dropped
        capture = capture or CaptureStats()
        window = capture.since(self._capture)
        log.debug(
            "monitor: %d frames in %.1fs, peak rms=%d, peak wake score=%.3f, "
            "dropped=%d, %s",
            self._frames, now - self._start, self._rms, self._score, evicted,
            window.describe(),
        )
        expected = self._expected_fps * (now - self._start)
        if self._frames < 0.8 * expected:
            # the capture line names the stage: buffers short of the frame
            # count means the graph under-fed us, buffers full means we lost
            # them ourselves between the appsink and this loop
            log.warning(
                "degraded capture: %d of %d expected mic frames in %.0fs "
                "(%s) — wakeword detection will be unreliable",
                self._frames, int(expected), now - self._start, window.describe(),
            )
        if evicted:
            # the wake queue holds 4 s; losing frames means the loop stalled
            # that long, and the spliced audio silently breaks a wakeword
            log.warning(
                "%d mic frames evicted from the wakeword queue — the event "
                "loop stalled; detection was deaf across the gap",
                evicted,
            )
        self._frames = 0
        self._rms = 0
        self._score = 0.0
        self._dropped = dropped
        self._capture = capture
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


def _build_speaker(config: Config, sink: AudioSink) -> SpeakerProtocol:
    """The local TTS engine (or its lazy stand-in when a cloud engine is primary)."""
    if config.tts.engine == "piper":
        return PiperSpeaker(config.piper, config.tts, sink)
    # cloud engine primary: piper stays the fallback but loads on first use
    return LazySpeaker(lambda: PiperSpeaker(config.piper, config.tts, sink))


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
    cloud_engines = {config.stt.engine, config.tts.engine} - {"local", "piper"}
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
    def __init__(self, config: Config) -> None:
        self._config = config
        self.state = State.IDLE
        self._pipeline_task: asyncio.Task | None = None

    def _set_state(self, state: State) -> None:
        log.debug("state: %s -> %s", self.state.name, state.name)
        self.state = state

    async def run(self) -> None:
        config = self._config
        log.info("loading models...")
        detector = build_detector(config)
        endpointer = SpeechEndpointer(config.vad)
        transcriber = Transcriber(config.stt, config.tts.default_language)

        async with AsyncExitStack() as stack:
            # capture starts once everything else is loaded: piper alone blocks
            # this loop for ~3.5s, and a live mic stream that nobody services
            # xruns its way out of PipeWire's scheduling for good
            source, sink = await stack.enter_async_context(
                audio_io(config.audio, start_capture=False)
            )
            earcons = Earcons(config.earcons, sink)
            speaker = _build_speaker(config, sink)

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

            source.start()  # nothing blocking left; the monitor is next
            log.info(
                "ready — say the wakeword (%s via %s)",
                config.wakeword.model,
                config.wakeword.engine,
            )
            link_check = asyncio.create_task(
                verify_links(source.target, sink.target), name="verify-links"
            )
            try:
                await self._interrupt_monitor(
                    wake_queue, detector, pipeline, sink, earcons, source
                )
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
        log.info("interaction cancelled")
        return was_speaking

    def _resync_detector(self, detector: WakewordProtocol, wake_queue: asyncio.Queue) -> None:
        """Clear the detector and abandon the mic backlog that outran it.

        Nothing downstream reads wake_queue — an interaction subscribes for
        itself — so everything queued here is audio the detector has already
        fallen behind on, most of it our own playback from just before a
        barge-in. Replaying it after the reset would score TTS echo against
        the IDLE threshold, which is exactly how the assistant wakes itself.
        """
        detector.reset()
        stale = drain_stale(wake_queue)
        if stale:
            log.debug(
                "dropped %d stale mic frames (%.1fs) after detector reset",
                stale, stale * self._config.audio.frame_ms / 1000,
            )

    async def _interrupt_monitor(
        self,
        wake_queue: asyncio.Queue,
        detector: WakewordProtocol,
        pipeline: Pipeline,
        sink: AudioSink,
        earcons: Earcons,
        source: AudioSource,
    ) -> None:
        """Always-on wakeword loop; starts or cancels the interaction task."""
        audio = self._config.audio
        health = _CaptureHealth(audio.sample_rate / audio.frame_samples, source.stats())
        duck = _DuckController()
        score_floor = _wake_score_floor()
        try:
            while True:
                try:
                    frame = await asyncio.wait_for(wake_queue.get(), timeout=MIC_STALL_WARN_S)
                except asyncio.TimeoutError:
                    log.warning(
                        "no mic frames for %.0fs — capture stream stalled? (%s)",
                        MIC_STALL_WARN_S, source.stats().describe(),
                    )
                    health.restart(source.stats())
                    continue
                if frame is None:
                    raise CaptureClosedError(
                        "audio capture stream closed mid-run "
                        f"({source.stats().describe()})"
                    )

                # the sink, not the state, knows whether the room is loud:
                # THINKING is silent for its whole whisper roundtrip, while
                # the wake and ack earcons are audible outside SPEAKING
                detection = detector.process(frame, speaking=sink.is_playing)
                score = detector.score("wake")
                health.observe(
                    frame, score, getattr(wake_queue, "dropped", 0), source.stats()
                )
                # two gates on purpose: the threshold raise above covers
                # everything audible (earcons included), ducking covers TTS
                # only — earcons are too short to duck, and one started
                # between frames may begin ducked until the next update
                duck.update(self.state is State.SPEAKING, score, sink)
                _dump_wake_audio(detector, detection, score, self.state, score_floor)

                if detection is None:
                    continue

                if self.state is State.IDLE and detection == "wake":
                    _log_wake_detection(detector, score)
                    self._resync_detector(detector, wake_queue)
                    self._start_pipeline(pipeline, earcons)
                elif self.state in (State.LISTENING, State.THINKING, State.SPEAKING):
                    # wakeword or stop-word during an interaction = barge-in
                    was_speaking = await self._cancel_pipeline(sink)
                    duck.release(sink)
                    self._resync_detector(detector, wake_queue)
                    if (
                        detection == "wake"
                        and was_speaking
                        and self._config.barge_in.resume_listening
                    ):
                        self._start_pipeline(pipeline, earcons, play_wake_earcon=True)
                    else:
                        await earcons.play("idle")
                else:
                    # "stop" with no interaction running: nothing to stop, and
                    # the hysteresis re-arms on its own a few frames later
                    log.debug("stop word ignored while idle (score %.2f)",
                              detector.score("stop"))
        finally:
            # shutdown (source closed or Ctrl-C): a still-running interaction
            # must die, and its conversation cleanup finish, before App.run's
            # exit stack closes the session and sink they use
            await self._cancel_pipeline(sink)
            await pipeline.close()
