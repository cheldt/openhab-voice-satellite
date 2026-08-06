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
from openhab_voice_satellite.app import App, _build_engines, _build_speaker
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
        self.queue.put_nowait(None)  # source-closed sentinel ends the monitor
        await asyncio.wait_for(self.task, timeout=2.0)
        if self.app._pipeline_task is not None:
            self.app._pipeline_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await self.app._pipeline_task

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


async def test_none_frame_exits_monitor():
    async with Monitor() as m:
        pass  # __aexit__ sends None and awaits a clean return
    assert m.task.done() and m.task.exception() is None


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


# --- engine wiring ---------------------------------------------------------


class _StubLocalSpeaker:
    def __init__(self, *args, **kwargs):
        pass


async def test_build_speaker_per_engine(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "PiperSpeaker", _StubLocalSpeaker)
    sink = BufferAudioSink()

    piper = _build_speaker(Config(), sink, tmp_path)  # piper is the default
    assert isinstance(piper, _StubLocalSpeaker)

    cloud = _build_speaker(
        Config.model_validate(
            {"tts": {"engine": "gemini"}, "gemini": {"api_key": "k"}}
        ),
        sink,
        tmp_path,
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
