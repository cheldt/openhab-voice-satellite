"""Application wiring: owns the state, the interrupt monitor and the pipeline task."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Callable

import aiohttp
import numpy as np

from .audio.broadcast import AudioBroadcaster
from .audio.earcons import Earcons
from .audio.io import audio_io, verify_links
from .audio.sink import AudioSink
from .audio.source import AudioSource
from .audio.wav import rms
from .config import Config
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

if TYPE_CHECKING:
    from .audio.gst_source import CaptureStats

log = logging.getLogger(__name__)

# wakeword score that triggers duck-and-confirm during playback
DUCK_PRETHRESHOLD = 0.35
DUCK_HOLD_FRAMES = 13  # ~1 s of 80 ms frames

MIC_STALL_WARN_S = 10.0  # no frames for this long -> loud warning
HEARTBEAT_S = 10.0  # capture-health DEBUG line interval


class CaptureClosedError(RuntimeError):
    """The capture stream ended mid-run (bus ERROR/EOS, e.g. PipeWire node loss).

    Deliberately fatal: in-process recovery would need set_state() from the
    sync bus handler (which deadlocks, see gst_common.install_sync_handler),
    and a wedged PipeWire stream rarely recovers its scheduling anyway. The
    process exits non-zero so the systemd unit (Restart=on-failure) restarts
    it with a fresh graph connection.
    """


def _detection_scores(detector: WakewordProtocol, detection: str, score: float) -> str:
    """The score behind a detection, as a log suffix.

    The candidate's trigger score rather than this frame's, where the engine
    records one: on a window-scoring engine the frame that reports the
    detection can already be a hop past the evaluation that fired it. A stop
    reports its own live score; the wake head's last_trigger_score would be
    stale from the accept that opened the interaction.
    """
    if detection == "stop":
        return f"score {detector.score('stop'):.2f}"
    trigger_score = getattr(detector, "last_trigger_score", None)
    return f"score {score if trigger_score is None else trigger_score:.2f}"


class _CaptureHealth:
    """Frame-rate/RMS bookkeeping behind the heartbeat + degraded-capture logs."""

    def __init__(
        self, expected_fps: float, graph: Callable[[], CaptureStats] | None = None
    ) -> None:
        self._expected_fps = expected_fps
        # the source's cumulative capture accounting; differenced per window
        # so a degraded window says whether PipeWire under-fed the graph or
        # this loop fell behind
        self._graph = graph
        self._graph_at_start = graph() if graph else None
        self._frames = 0
        self._rms = 0
        self._score = 0.0
        self._start = time.monotonic()

    def graph_report(self) -> str:
        """What the graph delivered in the current window, or "" without a source."""
        if self._graph is None:
            return "no capture accounting"
        return self._graph().since(self._graph_at_start).describe()

    def _rewind_window(self, now: float) -> None:
        self._frames = 0
        self._rms = 0
        self._score = 0.0
        self._start = now
        if self._graph is not None:
            self._graph_at_start = self._graph()

    def restart(self) -> None:
        """Reset the window clock (after a stall, so the gap isn't counted).

        Every counter moves with it, not just the clock: a window that is not
        counted must not lend anything to the next one. Leaving `_frames`
        behind inflated the frame count against a shortened window, which
        muted the degraded-capture warning for exactly the window after a
        stall — the one where it matters.
        """
        self._rewind_window(time.monotonic())

    def observe(self, frame: np.ndarray, score: float) -> None:
        self._frames += 1
        self._rms = max(self._rms, rms(frame))
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
                "— wakeword detection will be unreliable (%s)",
                self._frames, int(expected), now - self._start, self.graph_report(),
            )
        self._rewind_window(now)


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
            # xruns its way out of PipeWire's scheduling for good — measured
            # on the Pi as 17 of 140 expected frames per window, for the life
            # of the process
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
                try:
                    event = await pipeline.run_interaction(play_wake_earcon)
                finally:
                    # IDLE before the earcon, so the monitor treats a wakeword
                    # during the tail as a fresh interaction, not a barge-in
                    self._set_state(State.IDLE)
                if event is not Event.ERROR:
                    # still the tracked task: the tail has to die with it at
                    # shutdown, before App.run's exit stack closes the sink
                    await earcons.play("idle")
            finally:
                # only our own reference. The state is IDLE while the tail
                # plays, so a wakeword in that window starts the next
                # interaction and re-points _pipeline_task at it; clearing
                # unconditionally here would orphan that task — barge-in and
                # the shutdown cancel would then find nothing to cancel.
                if self._pipeline_task is asyncio.current_task():
                    self._pipeline_task = None

        self._pipeline_task = asyncio.create_task(_run(), name="interaction")

    async def _cancel_pipeline(self, sink: AudioSink) -> bool:
        task = self._pipeline_task
        if task is None:
            return False
        was_speaking = self.state is State.SPEAKING
        sink.stop()
        task.cancel()
        # `await task` raises CancelledError for two indistinguishable reasons:
        # the child finished unwinding, or *we* were cancelled while parked on
        # it (Ctrl-C during a barge-in unwind), in which case Task.cancel
        # forwarded the cancel to this very await. Swallowing both meant the
        # shutdown cancel vanished and the monitor resumed its loop, so the
        # first Ctrl-C did nothing. The cancelling() count is what tells them
        # apart; the shutdown `finally` path enters with it already raised, so
        # it stays unaffected.
        cancels = asyncio.current_task().cancelling()
        try:
            await task
        except asyncio.CancelledError:
            if asyncio.current_task().cancelling() > cancels:
                raise
        self._set_state(State.IDLE)
        self._pipeline_task = None
        log.info("interaction cancelled")
        return was_speaking

    async def _interrupt_monitor(
        self,
        wake_queue: asyncio.Queue,
        detector: WakewordProtocol,
        pipeline: Pipeline,
        sink: AudioSink,
        earcons: Earcons,
        source: AudioSource | None = None,
    ) -> None:
        """Always-on wakeword loop; starts or cancels the interaction task.

        `source` is only read for its capture accounting, so a shortfall can
        be attributed to the graph or to this loop; None reports neither.
        """
        audio = self._config.audio
        graph = source.stats if source is not None else None
        health = _CaptureHealth(audio.sample_rate / audio.frame_samples, graph)
        duck = _DuckController()
        try:
            while True:
                try:
                    # asyncio.timeout, not wait_for: on 3.11 (the deployment
                    # target) wait_for swallows an external cancel that races
                    # a completed inner await (gh-86296), and this loop gets a
                    # completed get() every ~80 ms while being the coroutine
                    # shutdown's cancel has to reach. Same reason as
                    # recorder._next_frame and gst_sink._await_playout.
                    async with asyncio.timeout(MIC_STALL_WARN_S):
                        frame = await wake_queue.get()
                except asyncio.TimeoutError:
                    log.warning(
                        "no mic frames for %.0fs — capture stream stalled? (%s)",
                        MIC_STALL_WARN_S, health.graph_report(),
                    )
                    health.restart()
                    continue
                if frame is None:
                    raise CaptureClosedError(
                        f"audio capture stream closed mid-run ({health.graph_report()})"
                    )

                speaking = self.state in (State.THINKING, State.SPEAKING)
                detection = detector.process(frame, speaking=speaking)
                score = detector.score("wake")
                health.observe(frame, score)
                # two gates on purpose: the threshold raise above covers
                # everything audible (earcons included), ducking covers TTS
                # only — earcons are too short to duck, and one started
                # between frames may begin ducked until the next update
                duck.update(self.state is State.SPEAKING, score, sink)

                if detection is None:
                    continue

                if self.state is State.IDLE and detection == "wake":
                    log.info(
                        "wakeword detected (%s)", _detection_scores(detector, detection, score)
                    )
                    detector.reset()
                    self._start_pipeline(pipeline, earcons)
                elif self.state in (State.LISTENING, State.THINKING, State.SPEAKING):
                    # wakeword or stop-word during an interaction = barge-in.
                    # Logged before the cancel, with the score that caused it
                    # and whether our own output was audible at the time:
                    # without those, a deliberate interruption and the
                    # assistant self-triggering on its own TTS echo are the
                    # same "interaction cancelled" line in the journal, and
                    # the echo case is the one the raised speaking threshold
                    # exists to prevent — so it is the one worth seeing.
                    log.info(
                        "barge-in: %s during %s (%s%s)",
                        detection, self.state.name,
                        _detection_scores(detector, detection, score),
                        ", our output was audible" if speaking else "",
                    )
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
        finally:
            # shutdown (source closed or Ctrl-C): a still-running interaction
            # must die, and its conversation cleanup finish, before App.run's
            # exit stack closes the session and sink they use
            await self._cancel_pipeline(sink)
            await pipeline.close()
