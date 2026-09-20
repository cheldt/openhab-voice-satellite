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
from openhab_voice_satellite.audio.wav import read_wav_mono
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


def test_degraded_capture_warning_attributes_the_shortfall(caplog, monkeypatch):
    """The warning says whether the graph under-fed us or this loop fell behind."""
    from openhab_voice_satellite.audio.gst_source import CaptureStats

    snapshots = iter([
        CaptureStats(buffers=100, samples=80000),                 # at window start
        CaptureStats(buffers=112, samples=89600, pts_gaps=3, pts_gap_s=8.2),
        CaptureStats(buffers=112, samples=89600, pts_gaps=3, pts_gap_s=8.2),  # next window start
    ])
    health = app_module._CaptureHealth(12.5, lambda: next(snapshots))
    with caplog.at_level(logging.WARNING):
        for _ in range(10):
            health.observe(FRAME, 0.0)
        health._start -= app_module.HEARTBEAT_S
        health.observe(FRAME, 0.0)
    assert "degraded capture" in caplog.text
    # differenced against the window start, not cumulative
    assert "12 buffers / 9600 samples, 3 PTS gaps totalling 8.2s" in caplog.text


def test_health_without_a_source_still_reports(caplog):
    health = app_module._CaptureHealth(12.5)
    assert health.graph_report() == "no capture accounting"


# --- barge-in logging -----------------------------------------------------


async def test_barge_in_logs_the_score_that_caused_it(caplog):
    """A barge-in used to log only "interaction cancelled".

    That makes a deliberate interruption and the assistant self-triggering on
    its own TTS echo the same line in the journal — and the echo case is the
    one the raised speaking threshold exists to prevent, so it is the one
    worth being able to see. Only the IDLE branch logged its score.
    """
    detector = ScriptedDetector(
        detections={0: "wake", 1: "wake"}, scores={0: 0.9, 1: 0.42}, trace="0.12 0.90*"
    )
    async with Monitor(
        detector=detector,
        pipeline_kwargs={"state_on_run": State.SPEAKING, "hold_s": 10.0},
    ) as m:
        with caplog.at_level(logging.INFO):
            await m.feed()  # wake from IDLE
            await m.feed()  # wake during SPEAKING -> barge-in
    assert "wakeword detected (score 0.90, trace 0.12 0.90*)" in caplog.text
    assert (
        "barge-in: wake during SPEAKING (score 0.42, trace 0.12 0.90*, "
        "our output was audible)"
    ) in caplog.text


async def test_a_barge_in_in_a_quiet_room_says_nothing_about_our_output(caplog):
    # the clause is the informative half; it must not appear while the state
    # is silent, or every barge-in reads as a possible echo
    detector = ScriptedDetector(
        detections={0: "wake", 1: "wake"}, scores={1: 0.5}, trace="0.50*"
    )
    async with Monitor(
        detector=detector,
        pipeline_kwargs={"state_on_run": State.LISTENING, "hold_s": 10.0},
    ) as m:
        with caplog.at_level(logging.INFO):
            await m.feed()
            await m.feed()
    assert "barge-in: wake during LISTENING (score 0.50, trace 0.50*)" in caplog.text
    assert "audible" not in caplog.text


async def test_a_stop_barge_in_reports_the_stop_head_not_the_wake_peak(caplog):
    """A stop reports its own live score, not the wake head's trigger score,
    which is stale from the accept that started the interaction."""
    detector = ScriptedDetector(
        detections={0: "wake", 1: "stop"}, scores={0: 0.9, 1: 0.55}, trace="0.55*"
    )
    async with Monitor(
        detector=detector,
        pipeline_kwargs={"state_on_run": State.SPEAKING, "hold_s": 10.0},
    ) as m:
        with caplog.at_level(logging.INFO):
            await m.feed()
            await m.feed()
    assert (
        "barge-in: stop during SPEAKING (score 0.55, trace 0.55*, "
        "our output was audible)"
    ) in caplog.text


async def test_a_detection_carries_the_score_history_that_led_to_it(caplog):
    """The trace is what separates a phrase from a transient without audio.

    A false accept on room speech is one high evaluation in an otherwise
    quiet run; a real wakeword climbs and holds. At patience 1 that
    difference is invisible in the single score above, and it is exactly the
    question every tuning round asks of the journal.
    """
    detector = ScriptedDetector(
        detections={0: "wake"}, scores={0: 0.85}, trace="0.01 0.02 0.31 0.85*"
    )
    async with Monitor(detector=detector) as m:
        with caplog.at_level(logging.INFO):
            await m.feed()
    assert "wakeword detected (score 0.85, trace 0.01 0.02 0.31 0.85*)" in caplog.text


# --- wake-audio dumps -----------------------------------------------------


TONE = (np.arange(16000, dtype=np.int16) % 1000) - 500  # recognisable payload
AFTER = 8  # frames the dump collects past a detection (WAKE_DUMP_AFTER_S / 80 ms)


async def test_the_dump_spans_the_detection_not_just_its_run_up(tmp_path, monkeypatch):
    """The file has to carry the evaluations `patience` would have counted.

    detector.tail() ends on the evaluation that fired, so a dump of the tail
    alone replays as "no patience above 1 would ever have fired" — for every
    recording, true and false alike, which is the one question the dumps are
    collected to answer.
    """
    monkeypatch.setenv("OVS_DUMP_WAKE", str(tmp_path))
    detector = ScriptedDetector(detections={0: "wake"}, scores={0: 0.85}, tail=TONE)
    async with Monitor(
        detector=detector, pipeline_kwargs={"event": Event.PLAYBACK_DONE}
    ) as m:
        await m.feed()
        assert list(tmp_path.iterdir()) == []  # still collecting the trailing half
        await m.feed(AFTER)
    dumps = sorted(tmp_path.iterdir())
    assert len(dumps) == 1
    assert dumps[0].name.startswith("wake-") and dumps[0].name.endswith("-0.85.wav")
    pcm, rate = read_wav_mono(dumps[0])
    assert rate == 16000
    assert np.array_equal(pcm[:len(TONE)], TONE)
    assert len(pcm) == len(TONE) + AFTER * len(FRAME)


async def test_the_snapshot_is_taken_before_the_detector_is_reset(tmp_path, monkeypatch):
    # reset() clears the ring tail() reads, so an arm() placed after it would
    # silently write nothing — the fake's tail() empties on reset to make that
    # a failure rather than a comment
    monkeypatch.setenv("OVS_DUMP_WAKE", str(tmp_path))
    detector = ScriptedDetector(detections={0: "wake"}, scores={0: 0.85}, tail=TONE)
    async with Monitor(
        detector=detector, pipeline_kwargs={"event": Event.PLAYBACK_DONE}
    ) as m:
        await m.feed(1 + AFTER)
        assert detector.resets == 1
    assert len(list(tmp_path.iterdir())) == 1


async def test_a_barge_in_dumps_its_audio_too(tmp_path, monkeypatch):
    # the branch that keeps being the forgotten one: it is also the branch
    # where a TTS-echo accept would show up
    monkeypatch.setenv("OVS_DUMP_WAKE", str(tmp_path))
    detector = ScriptedDetector(
        detections={0: "wake", 1 + AFTER: "stop"}, scores={0: 0.9, 1 + AFTER: 0.55},
        tail=TONE,
    )
    async with Monitor(
        detector=detector,
        pipeline_kwargs={"state_on_run": State.SPEAKING, "hold_s": 10.0},
    ) as m:
        await m.feed(1 + AFTER)   # wake from IDLE, then its trailing window
        await m.feed(1 + AFTER)   # stop during SPEAKING -> barge-in
    names = sorted(path.name for path in tmp_path.iterdir())
    assert len(names) == 2
    assert names[0].startswith("stop-") and names[0].endswith("-0.55.wav")
    assert names[1].startswith("wake-")


async def test_a_second_detection_inside_the_window_flushes_the_first(tmp_path, monkeypatch):
    # a truncated dump of the first is still evidence; dropping it is not
    monkeypatch.setenv("OVS_DUMP_WAKE", str(tmp_path))
    detector = ScriptedDetector(
        detections={0: "wake", 1: "stop"}, scores={0: 0.9, 1: 0.55}, tail=TONE
    )
    async with Monitor(
        detector=detector,
        pipeline_kwargs={"state_on_run": State.SPEAKING, "hold_s": 10.0},
    ) as m:
        await m.feed(2 + AFTER)
    assert len(list(tmp_path.iterdir())) == 2


async def test_nothing_is_written_without_the_env_var(tmp_path):
    detector = ScriptedDetector(detections={0: "wake"}, scores={0: 0.85}, tail=TONE)
    async with Monitor(
        detector=detector, pipeline_kwargs={"event": Event.PLAYBACK_DONE}
    ) as m:
        await m.feed(1 + AFTER)
        assert m.pipeline.calls == [True]
    assert list(tmp_path.iterdir()) == []


async def test_an_empty_ring_writes_no_file(tmp_path, monkeypatch):
    # a detection before the ring filled (or after a reset) must not leave a
    # 0-byte WAV behind for --score-wav to choke on
    monkeypatch.setenv("OVS_DUMP_WAKE", str(tmp_path))
    detector = ScriptedDetector(detections={0: "wake"}, scores={0: 0.85})
    async with Monitor(
        detector=detector, pipeline_kwargs={"event": Event.PLAYBACK_DONE}
    ) as m:
        await m.feed(1 + AFTER)
    assert list(tmp_path.iterdir()) == []


async def test_a_failing_dump_does_not_swallow_the_wake(tmp_path, monkeypatch, caplog):
    """Diagnostics must never cost an interaction.

    An unwritable dump directory is a debugging mistake; losing the wakeword
    over it would turn a mistake into an outage.
    """
    not_a_directory = tmp_path / "file"
    not_a_directory.write_bytes(b"")
    monkeypatch.setenv("OVS_DUMP_WAKE", str(not_a_directory))
    detector = ScriptedDetector(detections={0: "wake"}, scores={0: 0.85}, tail=TONE)
    async with Monitor(
        detector=detector, pipeline_kwargs={"event": Event.PLAYBACK_DONE}
    ) as m:
        with caplog.at_level(logging.INFO):
            await m.feed(1 + AFTER)
        assert m.pipeline.calls == [True]
    assert "wake audio dump failed" in caplog.text
