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
from openhab_voice_satellite.audio.broadcast import SubscriberQueue
from openhab_voice_satellite.audio.gst_source import CaptureStats
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
    SilenceAudioSource,
)

FRAME = np.zeros(1280, dtype=np.int16)


class Monitor:
    """Runs _interrupt_monitor as a task around scripted collaborators."""

    def __init__(self, config: Config | None = None, detector=None, pipeline_kwargs=None):
        self.app = App(config or Config())
        self.queue = SubscriberQueue(maxsize=50)
        self.detector = detector or ScriptedDetector()
        self.sink = BufferAudioSink()
        self.earcons = RecordingEarcons()
        self.pipeline = FakePipeline(set_state=self.app._set_state, **(pipeline_kwargs or {}))
        self.source = SilenceAudioSource()  # only its stats() is used here
        self.task: asyncio.Task | None = None

    async def __aenter__(self):
        self.task = asyncio.create_task(
            self.app._interrupt_monitor(
                self.queue, self.detector, self.pipeline, self.sink, self.earcons,
                self.source,
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

    def stuff(self, n: int) -> None:
        """Queue frames without yielding, so the monitor cannot consume them."""
        for _ in range(n):
            self.queue.put_nowait(FRAME)


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


async def test_wake_drops_the_stale_backlog():
    detector = ScriptedDetector(detections={0: "wake"})
    async with Monitor(
        detector=detector, pipeline_kwargs={"event": Event.PLAYBACK_DONE}
    ) as m:
        m.stuff(20)  # queued behind the triggering frame, never yielded to
        await m.feed()
        assert m.queue.qsize() == 0
        assert detector.frames_seen == 1  # the 20 stale frames were abandoned


async def test_barge_in_drops_the_stale_backlog():
    """The backlog during a cancel is our own TTS; replaying it re-triggers."""
    detector = ScriptedDetector(detections={0: "wake", 1: "wake"})
    async with Monitor(
        detector=detector,
        pipeline_kwargs={"state_on_run": State.SPEAKING, "hold_s": 10.0},
    ) as m:
        await m.feed()
        m.stuff(30)  # what the mic captured while the cancel was in flight
        await m.feed()
        assert m.queue.qsize() == 0
        assert detector.frames_seen == 2
        assert detector.resets == 2


async def test_drain_keeps_the_sentinel_so_the_monitor_still_exits():
    detector = ScriptedDetector(detections={0: "wake"})
    async with Monitor(
        detector=detector, pipeline_kwargs={"event": Event.PLAYBACK_DONE}
    ) as m:
        m.stuff(5)
        m.queue.put_nowait(None)  # source closed while the backlog sat there
        await m.feed()
        with pytest.raises(CaptureClosedError):
            await asyncio.wait_for(m.task, timeout=2.0)  # sentinel survived the drain


async def test_drain_does_not_count_as_backpressure_loss():
    detector = ScriptedDetector(detections={0: "wake"})
    async with Monitor(
        detector=detector, pipeline_kwargs={"event": Event.PLAYBACK_DONE}
    ) as m:
        m.stuff(10)
        await m.feed()
        assert m.queue.dropped == 0  # deliberate drops are not lost frames


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


async def test_speaking_flag_tracks_sink_playout_not_state():
    """The raised threshold follows audible audio, not the interaction state.

    THINKING is silent for the whole whisper roundtrip, so raising the bar
    there is pure recall loss; conversely the wake earcon is audible during
    LISTENING, where the state-based flag used to leave the bar low.
    """
    detector = ScriptedDetector()
    async with Monitor(detector=detector) as m:
        await m.feed()  # idle and quiet
        m.app.state = State.THINKING  # thinking, but nothing is playing
        await m.feed()
        m.sink.is_playing = True  # an earcon, in any state
        await m.feed()
        m.app.state = State.SPEAKING
        await m.feed()
        m.sink.is_playing = False
        await m.feed()
        assert detector.speaking_flags == [False, False, True, True, False]


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


async def test_degraded_capture_names_the_losing_stage(monkeypatch, caplog):
    """The warning has to say whether the graph or we lost the frames."""
    monkeypatch.setattr(app_module, "HEARTBEAT_S", 0.05)
    async with Monitor() as m:
        await m.feed()  # the monitor takes its baseline on the first pass
        m.source.capture = CaptureStats(
            buffers=400, samples=512000, pts_gaps=3, pts_gap_s=9.5, dropped=7
        )
        with caplog.at_level(logging.WARNING):
            await asyncio.sleep(0.3)
            await m.feed()
        assert "400 buffers / 512000 samples" in caplog.text
        assert "3 PTS gaps totalling 9.5s" in caplog.text
        assert "source queue drops 7" in caplog.text


def _closed_window(health, capture: CaptureStats) -> None:
    """Push one frame with the window already expired, so it reports."""
    health._start -= app_module.HEARTBEAT_S
    health.observe(FRAME, 0.0, 0, capture)


def test_capture_accounting_is_per_window(caplog):
    """A window reports its own buffers, not everything since startup."""
    health = app_module._CaptureHealth(12.5, CaptureStats())
    with caplog.at_level(logging.WARNING):
        _closed_window(health, CaptureStats(buffers=100, samples=128000))
        _closed_window(health, CaptureStats(buffers=130, samples=166400))
    assert "100 buffers / 128000 samples" in caplog.messages[0]
    assert "30 buffers / 38400 samples" in caplog.messages[1]


def test_a_stall_does_not_lend_its_buffers_to_the_next_window(caplog):
    health = app_module._CaptureHealth(12.5, CaptureStats())
    # frames that arrived during the stall belong to the window that was
    # thrown away, not to the one starting now
    health.restart(CaptureStats(buffers=500, samples=640000))
    with caplog.at_level(logging.WARNING):
        _closed_window(health, CaptureStats(buffers=500, samples=640000))
    assert "0 buffers / 0 samples" in caplog.messages[0]


def test_a_stall_does_not_lend_its_frame_count_either(caplog, monkeypatch):
    """The monitor-side counters must be zeroed by restart(), not only the
    capture baseline.

    A leaked frame count is charged against a shortened window, so the
    degraded-capture check reads a delivery rate the mic never achieved and
    stays silent for exactly the window after a stall.
    """
    monkeypatch.setattr(app_module, "HEARTBEAT_S", 10.0)
    health = app_module._CaptureHealth(12.5, CaptureStats())
    for _ in range(100):  # 8 s of normal capture, no heartbeat yet
        health.observe(FRAME, 0.9, 0, CaptureStats())
    health.restart(CaptureStats(), 0)  # the stall
    with caplog.at_level(logging.DEBUG):
        # 62 frames where a full window expects 125: must warn
        for _ in range(61):
            health.observe(FRAME, 0.0, 0, CaptureStats())
        _closed_window(health, CaptureStats())
    assert "degraded capture" in caplog.text
    assert "62 of 125 expected" in caplog.text
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


async def test_stop_word_while_idle_is_ignored_and_logged(caplog):
    detector = ScriptedDetector(detections={0: "stop"})
    async with Monitor(detector=detector) as m:
        with caplog.at_level(logging.DEBUG):
            await m.feed()
        assert m.pipeline.calls == []  # nothing to stop
        assert m.app.state is State.IDLE
        assert detector.resets == 0  # a no-op must not wipe the wake context
        assert "stop word ignored while idle" in caplog.text


async def test_wake_audio_dump_writes_detections(tmp_path, monkeypatch):
    monkeypatch.setenv("OVS_DUMP_WAKE", str(tmp_path))
    detector = ScriptedDetector(detections={0: "wake"}, scores={0: 0.83})
    async with Monitor(
        detector=detector, pipeline_kwargs={"event": Event.PLAYBACK_DONE}
    ) as m:
        await m.feed()
    dumped = list(tmp_path.glob("*.wav"))
    assert len(dumped) == 1
    assert dumped[0].name.startswith("wake-0.83-IDLE-")


async def test_wake_audio_dump_catches_near_misses(tmp_path, monkeypatch):
    monkeypatch.setenv("OVS_DUMP_WAKE", str(tmp_path))
    monkeypatch.setenv("OVS_DUMP_WAKE_SCORE", "0.3")
    detector = ScriptedDetector(scores={0: 0.44, 1: 0.10})
    async with Monitor(detector=detector) as m:
        await m.feed(2)
    dumped = [p.name for p in tmp_path.glob("*.wav")]
    assert len(dumped) == 1  # only the frame over the floor
    assert dumped[0].startswith("near-0.44-")


async def test_wake_audio_dump_catches_verifier_rejections(tmp_path, monkeypatch):
    # the verdict lands delay_ms after the peak, so the live score has decayed
    # below the floor — the gate must read the rejected candidate's peak
    monkeypatch.setenv("OVS_DUMP_WAKE", str(tmp_path))
    monkeypatch.setenv("OVS_DUMP_WAKE_SCORE", "0.3")
    detector = ScriptedDetector(scores={0: 0.05}, rejections={0: 0.62})
    async with Monitor(detector=detector) as m:
        await m.feed()
    dumped = [p.name for p in tmp_path.glob("*.wav")]
    assert len(dumped) == 1
    assert dumped[0].startswith("rejected-0.62-")


async def test_wake_audio_dump_survives_a_typoed_score_floor(tmp_path, monkeypatch, caplog):
    # a non-numeric floor used to raise ValueError inside the monitor loop
    monkeypatch.setenv("OVS_DUMP_WAKE", str(tmp_path))
    monkeypatch.setenv("OVS_DUMP_WAKE_SCORE", "banana")
    detector = ScriptedDetector(scores={0: 0.44})
    with caplog.at_level(logging.WARNING):
        async with Monitor(detector=detector) as m:
            await m.feed()
    assert list(tmp_path.glob("*.wav")) == []  # near-miss dumps off, monitor alive
    assert "OVS_DUMP_WAKE_SCORE" in caplog.text


async def test_wake_audio_dump_is_off_without_the_env_var(tmp_path):
    detector = ScriptedDetector(detections={0: "wake"})
    async with Monitor(
        detector=detector, pipeline_kwargs={"event": Event.PLAYBACK_DONE}
    ) as m:
        await m.feed()
    assert list(tmp_path.glob("*.wav")) == []
