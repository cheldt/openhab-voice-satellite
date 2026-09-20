"""App._interrupt_monitor behavior: wake dispatch, barge-in, ducking, health.

The monitor takes every collaborator as a parameter, so it is driven directly
with scripted fakes — no models, no GStreamer, no network.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack

import numpy as np
import pytest

import openhab_voice_satellite.app as app_module
from openhab_voice_satellite.app import (
    App,
    CaptureClosedError,
    _build_engines,
    _build_speaker,
)
from openhab_voice_satellite.config import Config
from openhab_voice_satellite.fallback import (
    FallbackSpeaker,
    FallbackTranscriber,
    LazySpeaker,
)
from openhab_voice_satellite.state import Event, State

from .fakes import (
    BufferAudioSink,
    FakePipeline,
    LocalSpeakerStub,
    LocalTranscriberStub,
    RecordingEarcons,
    ScriptedDetector,
)

FRAME = np.zeros(1280, dtype=np.int16)


class Monitor:
    """Runs _interrupt_monitor as a task around scripted collaborators."""

    def __init__(self, config: Config | None = None, detector=None, pipeline_kwargs=None):
        self.app = App(config or Config())
        self.queue: asyncio.Queue = asyncio.Queue()
        self.detector = detector or ScriptedDetector()
        self.sink = BufferAudioSink()
        self.earcons = RecordingEarcons()
        self.pipeline = FakePipeline(set_state=self.app._set_state, **(pipeline_kwargs or {}))
        self.task: asyncio.Task | None = None

    async def __aenter__(self):
        self.task = asyncio.create_task(
            self.app._interrupt_monitor(
                self.queue, self.detector, self.pipeline, self.sink, self.earcons
            )
        )
        return self

    async def __aexit__(self, *exc):
        if not self.task.done():  # a test may have cancelled the monitor itself
            self.queue.put_nowait(None)  # source-closed sentinel ends the monitor
            with pytest.raises(CaptureClosedError):
                await asyncio.wait_for(self.task, timeout=2.0)

    async def feed(self, n: int = 1) -> None:
        for _ in range(n):
            self.queue.put_nowait(FRAME)
        await asyncio.sleep(0.05)  # let the monitor drain and dispatch


async def test_wake_starts_pipeline():
    detector = ScriptedDetector(detections={0: "wake"})
    async with Monitor(
        detector=detector, pipeline_kwargs={"event": Event.PLAYBACK_DONE}
    ) as m:
        await m.feed()
        assert m.pipeline.calls == [True]
        assert detector.resets == 1
        assert m.earcons.played == ["idle"]  # after a clean interaction
        assert m.app.state is State.IDLE


async def test_error_event_suppresses_idle_earcon():
    detector = ScriptedDetector(detections={0: "wake"})
    async with Monitor(
        detector=detector, pipeline_kwargs={"event": Event.ERROR}
    ) as m:
        await m.feed()
        assert m.pipeline.calls == [True]
        assert "idle" not in m.earcons.played


async def test_wake_during_thinking_is_barge_in():
    detector = ScriptedDetector(detections={0: "wake", 1: "wake"})
    async with Monitor(
        detector=detector,
        pipeline_kwargs={"state_on_run": State.THINKING, "hold_s": 10.0},
    ) as m:
        await m.feed()
        assert m.app.state is State.THINKING
        await m.feed()
        assert m.sink.stopped
        assert m.app.state is State.IDLE
        assert m.pipeline.calls == [True]  # not speaking -> no restart
        assert m.earcons.played == ["idle"]


async def test_barge_in_while_speaking_resumes():
    detector = ScriptedDetector(detections={0: "wake", 1: "wake"})
    async with Monitor(
        detector=detector,
        pipeline_kwargs={"state_on_run": State.SPEAKING, "hold_s": 10.0},
    ) as m:
        assert m.app._config.barge_in.resume_listening  # default on
        await m.feed()  # wake -> pipeline starts, state SPEAKING
        await m.feed()  # wake during SPEAKING -> barge-in
        assert m.sink.stopped
        assert m.pipeline.calls == [True, True]  # re-listens after barge-in


async def test_barge_in_no_resume_plays_idle():
    config = Config.model_validate({"barge_in": {"resume_listening": False}})
    detector = ScriptedDetector(detections={0: "wake", 1: "wake"})
    async with Monitor(
        config=config,
        detector=detector,
        pipeline_kwargs={"state_on_run": State.SPEAKING, "hold_s": 10.0},
    ) as m:
        await m.feed()
        await m.feed()
        assert m.pipeline.calls == [True]
        assert m.earcons.played == ["idle"]


async def test_stop_word_never_resumes():
    detector = ScriptedDetector(detections={0: "wake", 1: "stop"})
    async with Monitor(
        detector=detector,
        pipeline_kwargs={"state_on_run": State.SPEAKING, "hold_s": 10.0},
    ) as m:
        assert m.app._config.barge_in.resume_listening
        await m.feed()
        await m.feed()
        assert m.pipeline.calls == [True]  # stop word cancels without restart
        assert m.earcons.played == ["idle"]
        assert m.app.state is State.IDLE


async def test_duck_on_prethreshold_then_unduck_after_hold():
    # score crosses DUCK_PRETHRESHOLD once while SPEAKING, then stays low
    scores = {1: 0.4}
    detector = ScriptedDetector(scores=scores)
    async with Monitor(detector=detector) as m:
        m.app.state = State.SPEAKING  # duck logic needs only the state
        await m.feed()  # frame 0: low score, nothing happens
        assert m.sink.duck_calls == []
        await m.feed()  # frame 1: 0.4 >= 0.35 -> duck
        assert m.sink.duck_calls == [0.2]
        await m.feed(app_module.DUCK_HOLD_FRAMES - 1)
        assert m.sink.unduck_calls == 0  # hold not elapsed yet
        await m.feed()
        assert m.sink.unduck_calls == 1  # released after DUCK_HOLD_FRAMES
        assert m.sink.duck_calls == [0.2]  # and not re-ducked


async def test_duck_released_on_leaving_speaking():
    detector = ScriptedDetector(scores={0: 0.4})
    async with Monitor(detector=detector) as m:
        m.app.state = State.SPEAKING
        await m.feed()
        assert m.sink.duck_calls == [0.2]
        m.app.state = State.IDLE  # playback over mid-hold
        await m.feed()
        assert m.sink.unduck_calls == 1


async def test_speaking_flag_tracks_state():
    detector = ScriptedDetector()
    async with Monitor(detector=detector) as m:
        await m.feed()  # IDLE
        m.app.state = State.THINKING
        await m.feed()
        m.app.state = State.SPEAKING
        await m.feed()
        m.app.state = State.IDLE
        await m.feed()
        assert detector.speaking_flags == [False, True, True, False]


async def test_none_frame_is_fatal():
    # capture death must not exit 0: systemd only restarts on failure
    async with Monitor() as m:
        pass  # __aexit__ sends None and expects CaptureClosedError
    assert m.task.done()
    assert isinstance(m.task.exception(), CaptureClosedError)


async def test_monitor_exit_cancels_running_interaction():
    # source death mid-interaction: the monitor's exit path must cancel the
    # held interaction and close the pipeline before App.run unwinds its stack
    detector = ScriptedDetector(detections={0: "wake"})
    async with Monitor(
        detector=detector,
        pipeline_kwargs={"state_on_run": State.THINKING, "hold_s": 10.0},
    ) as m:
        await m.feed()  # interaction starts and holds
        assert m.app._pipeline_task is not None
    assert m.app._pipeline_task is None
    assert m.app.state is State.IDLE
    assert m.pipeline.closed


async def test_monitor_cancellation_cancels_running_interaction():
    # Ctrl-C path: cancelling the monitor task itself must also tear down the
    # in-flight interaction and close the pipeline
    detector = ScriptedDetector(detections={0: "wake"})
    async with Monitor(
        detector=detector,
        pipeline_kwargs={"state_on_run": State.SPEAKING, "hold_s": 10.0},
    ) as m:
        await m.feed()
        assert m.app._pipeline_task is not None
        m.task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await m.task
        assert m.app._pipeline_task is None
        assert m.pipeline.closed


async def test_a_shutdown_cancel_during_a_barge_in_unwind_is_not_swallowed():
    """Ctrl-C while _cancel_pipeline is parked on the cancelled child.

    `await task` raises CancelledError for the child finishing *and* for our
    own cancel being forwarded to that same await, and the blanket except
    suppressed both — so the shutdown cancel vanished, the monitor resumed its
    loop, and the first Ctrl-C appeared to do nothing.
    """
    started = asyncio.Event()

    async def slow_to_unwind():
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            await asyncio.sleep(0.05)  # unwinding takes a beat
            raise

    app = App(Config())
    app.state = State.SPEAKING
    app._pipeline_task = asyncio.create_task(slow_to_unwind())
    await started.wait()

    monitor = asyncio.create_task(app._cancel_pipeline(BufferAudioSink()))
    await asyncio.sleep(0.01)  # parked in `await task`
    monitor.cancel()
    with pytest.raises(asyncio.CancelledError):
        await monitor


async def test_a_child_cancel_alone_is_still_swallowed():
    # the counterpart: a plain barge-in unwind must not propagate, or every
    # barge-in would kill the monitor
    app = App(Config())
    app.state = State.SPEAKING

    async def holder():
        await asyncio.sleep(10)

    app._pipeline_task = asyncio.create_task(holder())
    await asyncio.sleep(0)
    assert await app._cancel_pipeline(BufferAudioSink()) is True
    assert app._pipeline_task is None


async def test_mic_stall_warns_and_continues(monkeypatch, caplog):
    monkeypatch.setattr(app_module, "MIC_STALL_WARN_S", 0.05)
    async with Monitor() as m:
        with caplog.at_level(logging.WARNING):
            await asyncio.sleep(0.15)  # starve the queue past the stall timeout
        assert "capture stream stalled" in caplog.text
        assert not m.task.done()  # the loop keeps running


async def test_degraded_capture_warning(monkeypatch, caplog):
    monkeypatch.setattr(app_module, "HEARTBEAT_S", 0.05)
    async with Monitor() as m:
        with caplog.at_level(logging.WARNING):
            await m.feed()
            await asyncio.sleep(0.3)  # far fewer frames than expected_fps
            await m.feed()
        assert "degraded capture" in caplog.text


def test_a_stall_does_not_lend_its_frame_count_to_the_next_window(caplog, monkeypatch):
    """restart() must zero the counters, not just the clock.

    A leaked frame count is charged against a shortened window, so the
    degraded-capture check reads a delivery rate the mic never achieved and
    stays silent for exactly the window after a stall.
    """
    monkeypatch.setattr(app_module, "HEARTBEAT_S", 10.0)
    health = app_module._CaptureHealth(12.5)
    for _ in range(100):  # 8 s of normal capture, no heartbeat yet
        health.observe(FRAME, 0.9)
    health.restart()  # the stall
    with caplog.at_level(logging.DEBUG):
        # 62 frames where a full window expects 125: must warn
        for _ in range(61):
            health.observe(FRAME, 0.0)
        health._start -= app_module.HEARTBEAT_S
        health.observe(FRAME, 0.0)
    assert "degraded capture" in caplog.text
    assert "peak wake score=0.000" in caplog.text  # pre-stall peak gone too


# --- engine wiring ---------------------------------------------------------


class _StubLocalSpeaker:
    def __init__(self, *args, **kwargs):
        pass


async def test_build_speaker_per_engine(monkeypatch):
    monkeypatch.setattr(app_module, "PiperSpeaker", _StubLocalSpeaker)
    sink = BufferAudioSink()

    piper = _build_speaker(Config(), sink)  # piper is the default
    assert isinstance(piper, _StubLocalSpeaker)

    cloud = _build_speaker(
        Config.model_validate(
            {"tts": {"engine": "gemini"}, "gemini": {"api_key": "k"}}
        ),
        sink,
    )
    assert isinstance(cloud, LazySpeaker)  # piper fallback loads on first use


@pytest.mark.parametrize("provider", ["gemini", "deepgram"])
async def test_build_engines_wraps_cloud_with_fallback(provider):
    config = Config.model_validate(
        {
            "stt": {"engine": provider},
            "tts": {"engine": provider},
            provider: {"api_key": "k"},
        }
    )
    local_t, local_s = LocalTranscriberStub(), LocalSpeakerStub()
    async with AsyncExitStack() as stack:
        transcriber, speaker = await _build_engines(
            config, stack, BufferAudioSink(), local_t, local_s
        )
        assert isinstance(transcriber, FallbackTranscriber)
        assert isinstance(speaker, FallbackSpeaker)
        assert transcriber._label == provider
        assert speaker._label == provider
        assert transcriber._fallback is local_t
        assert speaker._fallback is local_s


async def test_build_engines_local_passthrough():
    local_t, local_s = LocalTranscriberStub(), LocalSpeakerStub()
    async with AsyncExitStack() as stack:
        transcriber, speaker = await _build_engines(
            Config(), stack, BufferAudioSink(), local_t, local_s
        )
        assert transcriber is local_t
        assert speaker is local_s


async def test_capture_close_is_fatal_so_systemd_restarts_us():
    """A closed capture stream must not unwind quietly.

    Returning here exits 0, which Restart=on-failure never restarts — a
    mid-run PipeWire node loss would leave the satellite alive and deaf.
    """
    async with Monitor() as m:
        m.queue.put_nowait(None)
        with pytest.raises(CaptureClosedError, match="closed mid-run"):
            await asyncio.wait_for(m.task, timeout=2.0)


# --- the idle tail --------------------------------------------------------


async def test_a_wake_during_the_idle_tail_does_not_orphan_the_next_interaction():
    """_run's finally must clear only its own task reference.

    The state is IDLE while the idle earcon plays, so a wakeword in that
    window starts the next interaction and re-points _pipeline_task at it.
    The first task's finally then ran and set it to None — the second
    interaction kept running with nothing tracking it, so barge-in found no
    task to cancel and shutdown closed the sink underneath it.
    """
    class TwoPhasePipeline:
        def __init__(self):
            self.calls = 0

        async def run_interaction(self, play_wake_earcon=True):
            self.calls += 1
            if self.calls == 1:
                return Event.PLAYBACK_DONE  # first answer finishes at once
            await asyncio.sleep(10)  # second interaction runs long

    class SlowIdleEarcons(RecordingEarcons):
        async def play(self, name):
            await super().play(name)
            await asyncio.sleep(0.1)  # the tail is audible for a while

    app = App(Config())
    pipeline, earcons = TwoPhasePipeline(), SlowIdleEarcons()
    app._start_pipeline(pipeline, earcons)
    first = app._pipeline_task
    await asyncio.sleep(0.02)  # answer done, tail playing, state IDLE
    assert app.state is State.IDLE and app._pipeline_task is first

    app._start_pipeline(pipeline, earcons)  # the wake during the tail
    second = app._pipeline_task
    await asyncio.sleep(0.15)  # the first task's tail ends, its finally runs
    assert first.done()
    assert not second.done()
    assert app._pipeline_task is second  # still tracked

    # and the tracked task is cancellable, as barge-in and shutdown need
    app.state = State.LISTENING
    assert await app._cancel_pipeline(BufferAudioSink()) is False  # not speaking
    assert second.cancelled()
    assert app._pipeline_task is None
