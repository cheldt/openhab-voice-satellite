import asyncio
import uuid

import aiohttp
import pytest
from aiohttp.test_utils import TestServer

from openhab_voice_satellite.audio.broadcast import AudioBroadcaster
from openhab_voice_satellite.config import Config
from openhab_voice_satellite.openhab import OpenHABClient
from openhab_voice_satellite.pipeline import Pipeline
from openhab_voice_satellite.state import Event, State

from .fakes import (
    BufferAudioSink,
    FakeEndpointer,
    FakeOpenHAB,
    FakeSpeaker,
    FakeTranscriber,
    NullEarcons,
    SilenceAudioSource,
)


@pytest.fixture
async def env():
    fake_oh = FakeOpenHAB(response="Das Licht ist an.")
    server = TestServer(fake_oh.build_app())
    await server.start_server(shutdown_timeout=0.2)

    config = Config.model_validate({"openhab": {"url": str(server.make_url(""))}})
    session = aiohttp.ClientSession()
    openhab = OpenHABClient(config.openhab, session)

    source = SilenceAudioSource()
    broadcaster = AudioBroadcaster(source)
    broadcaster.start()

    yield config, fake_oh, openhab, broadcaster

    await broadcaster.stop()
    source.close()
    await session.close()
    await server.close()


def _make_pipeline(config, openhab, broadcaster, sink, speaker,
                   states: list[State], transcriber: FakeTranscriber | None = None,
                   endpointer: FakeEndpointer | None = None,
                   earcons=None) -> Pipeline:
    return Pipeline(
        config=config,
        broadcaster=broadcaster,
        endpointer=endpointer or FakeEndpointer(speech_at=2, endpoint_at=5),
        transcriber=transcriber or FakeTranscriber(),
        openhab=openhab,
        speaker=speaker,
        earcons=earcons or NullEarcons(),
        set_state=states.append,
    )


async def _await_cleanup(pipeline: Pipeline) -> None:
    """Wait for the fire-and-forget conversation DELETE tasks."""
    await pipeline.close()


async def test_full_interaction(env):
    config, fake_oh, openhab, broadcaster = env
    config.dialog.followup_timeout_s = 0.3
    sink = BufferAudioSink()
    speaker = FakeSpeaker(sink)
    states: list[State] = []
    endpointer = FakeEndpointer(speech_at=2, endpoint_at=5, later=[(None, None)])
    pipeline = _make_pipeline(config, openhab, broadcaster, sink, speaker, states,
                              endpointer=endpointer)

    result = await pipeline.run_interaction()

    assert result is Event.PLAYBACK_DONE
    # a follow-up LISTENING precedes the silence that ends the conversation
    assert states == [State.LISTENING, State.THINKING, State.SPEAKING, State.LISTENING]
    assert fake_oh.commands == ["schalte das licht an"]
    assert speaker.spoken == [("Das Licht ist an.", "de")]
    assert len(sink.played) == 1
    uuid.UUID(fake_oh.conversations[0])  # valid conversation id was sent
    await _await_cleanup(pipeline)
    assert fake_oh.deleted == [fake_oh.conversations[0]]


async def test_barge_in_cancels_speaking(env):
    config, fake_oh, openhab, broadcaster = env
    sink = BufferAudioSink()
    speaker = FakeSpeaker(sink, duration_s=5.0)
    states: list[State] = []
    pipeline = _make_pipeline(config, openhab, broadcaster, sink, speaker, states)

    task = asyncio.create_task(pipeline.run_interaction())

    async def _until_speaking() -> None:
        while State.SPEAKING not in states:
            await asyncio.sleep(0.01)
            assert not task.done()

    await asyncio.wait_for(_until_speaking(), timeout=5.0)
    sink.stop()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert sink.stopped
    # recorder queue was released
    assert broadcaster._subscribers == []
    # conversation is deleted even when the pipeline task was cancelled
    await _await_cleanup(pipeline)
    assert fake_oh.deleted == [fake_oh.conversations[0]]


async def test_dialog_follow_up(env):
    config, fake_oh, openhab, broadcaster = env
    config.dialog.followup_timeout_s = 0.3
    fake_oh.responses = ["Welches Licht meinst du? Optionen: Küche, Wohnzimmer.", "Ok."]
    sink = BufferAudioSink()
    speaker = FakeSpeaker(sink)
    states: list[State] = []
    transcriber = FakeTranscriber(["schalte das licht an", "das im wohnzimmer"])
    endpointer = FakeEndpointer(speech_at=2, endpoint_at=5,
                                later=[(2, 5), (None, None)])
    pipeline = _make_pipeline(config, openhab, broadcaster, sink, speaker, states,
                              transcriber=transcriber, endpointer=endpointer)

    result = await pipeline.run_interaction()

    assert result is Event.PLAYBACK_DONE
    assert states == [State.LISTENING, State.THINKING, State.SPEAKING] * 2 + [State.LISTENING]
    assert fake_oh.commands == ["schalte das licht an", "das im wohnzimmer"]
    assert speaker.spoken == [
        ("Welches Licht meinst du? Optionen: Küche, Wohnzimmer.", "de"),
        ("Ok.", "de"),
    ]
    # both requests carry the same conversation id; deleted exactly once
    assert fake_oh.conversations[0] == fake_oh.conversations[1]
    await _await_cleanup(pipeline)
    assert fake_oh.deleted == [fake_oh.conversations[0]]


async def test_dialog_disabled(env):
    config, fake_oh, openhab, broadcaster = env
    config.dialog.enabled = False
    fake_oh.response = "Welches Licht meinst du?"
    sink = BufferAudioSink()
    speaker = FakeSpeaker(sink)
    states: list[State] = []
    pipeline = _make_pipeline(config, openhab, broadcaster, sink, speaker, states)

    result = await pipeline.run_interaction()

    assert result is Event.PLAYBACK_DONE
    assert states == [State.LISTENING, State.THINKING, State.SPEAKING]
    assert len(fake_oh.commands) == 1
    assert fake_oh.conversations == [None]
    await _await_cleanup(pipeline)
    assert fake_oh.deleted == []


async def test_dialog_follow_up_no_speech_ends(env):
    config, fake_oh, openhab, broadcaster = env
    config.dialog.followup_timeout_s = 0.5
    fake_oh.response = "Welches Licht meinst du?"
    sink = BufferAudioSink()
    speaker = FakeSpeaker(sink)
    states: list[State] = []
    endpointer = FakeEndpointer(speech_at=2, endpoint_at=5, later=[(None, None)])
    pipeline = _make_pipeline(config, openhab, broadcaster, sink, speaker, states,
                              endpointer=endpointer)

    result = await pipeline.run_interaction()

    assert result is Event.PLAYBACK_DONE  # graceful end, not an error
    assert len(fake_oh.commands) == 1
    assert speaker.spoken == [("Welches Licht meinst du?", "de")]
    await _await_cleanup(pipeline)
    assert fake_oh.deleted == [fake_oh.conversations[0]]


async def test_mic_subscribed_before_earcons(env):
    # speech during/right after the entry earcon must land in the queue;
    # previously the subscription was created only after the earcon finished
    config, fake_oh, openhab, broadcaster = env
    config.dialog.followup_timeout_s = 0.3
    sink = BufferAudioSink()
    speaker = FakeSpeaker(sink)
    states: list[State] = []

    class RecordingEarcons:
        def __init__(self) -> None:
            self.subscribers_at_play: list[tuple[str, int]] = []

        async def play(self, name: str) -> None:
            self.subscribers_at_play.append((name, len(broadcaster._subscribers)))

    earcons = RecordingEarcons()
    endpointer = FakeEndpointer(speech_at=2, endpoint_at=5, later=[(None, None)])
    pipeline = _make_pipeline(config, openhab, broadcaster, sink, speaker, states,
                              endpointer=endpointer, earcons=earcons)

    await pipeline.run_interaction()

    entry_earcons = [
        (name, n) for name, n in earcons.subscribers_at_play if name != "ack"
    ]
    assert entry_earcons  # wake + dialog follow-up rounds
    assert all(n >= 1 for _, n in entry_earcons)
    assert broadcaster._subscribers == []  # every round released its queue


async def test_barge_in_during_earcon_releases_subscription(env):
    config, fake_oh, openhab, broadcaster = env
    sink = BufferAudioSink()
    speaker = FakeSpeaker(sink)
    states: list[State] = []

    class BlockingEarcons:
        def __init__(self) -> None:
            self.playing = asyncio.Event()

        async def play(self, name: str) -> None:
            self.playing.set()
            await asyncio.sleep(30)

    earcons = BlockingEarcons()
    pipeline = _make_pipeline(config, openhab, broadcaster, sink, speaker, states,
                              earcons=earcons)

    task = asyncio.create_task(pipeline.run_interaction())
    await asyncio.wait_for(earcons.playing.wait(), timeout=2.0)
    assert broadcaster._subscribers != []  # subscription opened before the earcon
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert broadcaster._subscribers == []


async def test_no_speech_first_round(env):
    config, fake_oh, openhab, broadcaster = env
    config.vad.no_speech_timeout_s = 0.3
    sink = BufferAudioSink()
    speaker = FakeSpeaker(sink)
    states: list[State] = []
    endpointer = FakeEndpointer(speech_at=None, endpoint_at=None)
    pipeline = _make_pipeline(config, openhab, broadcaster, sink, speaker, states,
                              endpointer=endpointer)

    result = await pipeline.run_interaction()

    assert result is Event.NO_SPEECH
    assert fake_oh.commands == []
    # no request was sent, so no server-side conversation to delete
    assert pipeline._cleanup_tasks == set()
    assert fake_oh.deleted == []


async def test_openhab_timeout_returns_error(env):
    config, fake_oh, openhab, broadcaster = env
    fake_oh.response_delay_s = 10
    config.openhab.response_timeout_s = 0.2
    sink = BufferAudioSink()
    speaker = FakeSpeaker(sink)
    states: list[State] = []
    pipeline = _make_pipeline(config, openhab, broadcaster, sink, speaker, states)

    result = await pipeline.run_interaction()

    assert result is Event.ERROR
    assert speaker.spoken == []
    # the POST was attempted, so the conversation still gets deleted
    await _await_cleanup(pipeline)
    assert fake_oh.deleted == [fake_oh.conversations[0]]


async def test_timeout_elsewhere_is_not_an_openhab_timeout(env, caplog):
    # a TimeoutError leaked by any other component must hit the generic
    # failure path (with traceback), not be misread as an openHAB timeout
    config, fake_oh, openhab, broadcaster = env

    class TimingOutSpeaker:
        async def speak(self, text: str, language: str) -> None:
            raise TimeoutError("tts hung")

    states: list[State] = []
    pipeline = _make_pipeline(config, openhab, broadcaster, BufferAudioSink(),
                              TimingOutSpeaker(), states)

    result = await pipeline.run_interaction()

    assert result is Event.ERROR
    assert "openHAB response timed out" not in caplog.text
    assert "pipeline failed" in caplog.text
    await _await_cleanup(pipeline)


def test_truncate_for_log_short_string_unchanged():
    from openhab_voice_satellite.pipeline import _truncate_for_log

    assert _truncate_for_log("hello") == "hello"
    assert _truncate_for_log("a" * 200) == "a" * 200


def test_truncate_for_log_long_string_gets_marker():
    from openhab_voice_satellite.pipeline import _truncate_for_log

    result = _truncate_for_log("a" * 250)
    assert result.startswith("a" * 200)
    assert result.endswith("… (+50 chars)")
