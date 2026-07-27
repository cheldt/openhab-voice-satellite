import asyncio

import aiohttp
import numpy as np
import pytest
from aiohttp.test_utils import TestServer

from stt_proxy.audio.broadcast import AudioBroadcaster
from stt_proxy.config import Config
from stt_proxy.openhab import OpenHABClient
from stt_proxy.pipeline import Pipeline
from stt_proxy.state import Event, State
from stt_proxy.stt import Transcript

from .fakes import BufferAudioSink, FakeOpenHAB, SilenceAudioSource
from .test_recorder import FakeEndpointer


class FakeTranscriber:
    def __init__(self, texts: str | list[str] = "schalte das licht an", language: str = "de") -> None:
        self._texts = [texts] if isinstance(texts, str) else list(texts)
        self._language = language

    async def transcribe(self, pcm: np.ndarray) -> Transcript:
        await asyncio.sleep(0.01)
        text = self._texts.pop(0) if len(self._texts) > 1 else self._texts[0]
        return Transcript(text=text, language=self._language)


class FakeSpeaker:
    def __init__(self, sink: BufferAudioSink, duration_s: float = 0.0) -> None:
        self._sink = sink
        self._duration_s = duration_s
        self.spoken: list[tuple[str, str]] = []

    async def speak(self, text: str, language: str) -> None:
        self.spoken.append((text, language))
        await self._sink.play(np.zeros(160, dtype=np.int16), 16000)
        if self._duration_s:
            await asyncio.sleep(self._duration_s)


class NullEarcons:
    async def play(self, name: str) -> None:
        pass


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
                   endpointer: FakeEndpointer | None = None) -> Pipeline:
    return Pipeline(
        config=config,
        broadcaster=broadcaster,
        endpointer=endpointer or FakeEndpointer(speech_at=2, endpoint_at=5),
        transcriber=transcriber or FakeTranscriber(),
        openhab=openhab,
        speaker=speaker,
        sink=sink,
        earcons=NullEarcons(),
        set_state=states.append,
    )


async def test_full_interaction(env):
    config, fake_oh, openhab, broadcaster = env
    sink = BufferAudioSink()
    speaker = FakeSpeaker(sink)
    states: list[State] = []
    pipeline = _make_pipeline(config, openhab, broadcaster, sink, speaker, states)

    result = await pipeline.run_interaction()

    assert result is Event.PLAYBACK_DONE
    assert states == [State.LISTENING, State.THINKING, State.SPEAKING]
    assert fake_oh.commands == ["schalte das licht an"]
    assert speaker.spoken == [("Das Licht ist an.", "de")]
    assert len(sink.played) == 1


async def test_barge_in_cancels_speaking(env):
    config, fake_oh, openhab, broadcaster = env
    sink = BufferAudioSink()
    speaker = FakeSpeaker(sink, duration_s=5.0)
    states: list[State] = []
    pipeline = _make_pipeline(config, openhab, broadcaster, sink, speaker, states)

    task = asyncio.create_task(pipeline.run_interaction())
    while State.SPEAKING not in states:
        await asyncio.sleep(0.01)
        assert not task.done()
    sink.stop()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert sink.stopped
    # recorder queue was released
    assert broadcaster._subscribers == []


async def test_dialog_follow_up(env):
    config, fake_oh, openhab, broadcaster = env
    fake_oh.responses = ["Welches Licht meinst du? Optionen: Küche, Wohnzimmer.", "Ok."]
    sink = BufferAudioSink()
    speaker = FakeSpeaker(sink)
    states: list[State] = []
    transcriber = FakeTranscriber(["schalte das licht an", "das im wohnzimmer"])
    pipeline = _make_pipeline(config, openhab, broadcaster, sink, speaker, states,
                              transcriber=transcriber)

    result = await pipeline.run_interaction()

    assert result is Event.PLAYBACK_DONE
    assert states == [State.LISTENING, State.THINKING, State.SPEAKING] * 2
    assert fake_oh.commands == ["schalte das licht an", "das im wohnzimmer"]
    assert speaker.spoken == [
        ("Welches Licht meinst du? Optionen: Küche, Wohnzimmer.", "de"),
        ("Ok.", "de"),
    ]


async def test_dialog_stitch_mode(env):
    config, fake_oh, openhab, broadcaster = env
    config.dialog.context_mode = "stitch"
    fake_oh.responses = ["Welches Licht meinst du?", "Ok."]
    sink = BufferAudioSink()
    speaker = FakeSpeaker(sink)
    states: list[State] = []
    transcriber = FakeTranscriber(["schalte das licht an", "das im wohnzimmer"])
    pipeline = _make_pipeline(config, openhab, broadcaster, sink, speaker, states,
                              transcriber=transcriber)

    result = await pipeline.run_interaction()

    assert result is Event.PLAYBACK_DONE
    assert fake_oh.commands == [
        "schalte das licht an",
        "User: schalte das licht an\n"
        "Assistant: Welches Licht meinst du?\n"
        "User: das im wohnzimmer",
    ]


async def test_dialog_max_turns(env):
    config, fake_oh, openhab, broadcaster = env
    config.dialog.max_turns = 1
    fake_oh.response = "Und welche Helligkeit?"  # every answer is a question
    sink = BufferAudioSink()
    speaker = FakeSpeaker(sink)
    states: list[State] = []
    pipeline = _make_pipeline(config, openhab, broadcaster, sink, speaker, states)

    result = await pipeline.run_interaction()

    assert result is Event.PLAYBACK_DONE
    assert len(fake_oh.commands) == 2
    assert states.count(State.LISTENING) == 2
    assert len(speaker.spoken) == 2


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


async def test_dialog_follow_up_no_speech_ends(env):
    config, fake_oh, openhab, broadcaster = env
    config.vad.no_speech_timeout_s = 0.5
    fake_oh.response = "Welches Licht meinst du?"
    sink = BufferAudioSink()
    speaker = FakeSpeaker(sink)
    states: list[State] = []
    endpointer = FakeEndpointer(speech_at=2, endpoint_at=5, later=[(None, None)])
    pipeline = _make_pipeline(config, openhab, broadcaster, sink, speaker, states,
                              endpointer=endpointer)

    result = await pipeline.run_interaction()

    assert result is Event.NO_SPEECH
    assert len(fake_oh.commands) == 1
    assert speaker.spoken == [("Welches Licht meinst du?", "de")]


def test_stitch_context():
    from stt_proxy.pipeline import stitch_context

    assert stitch_context([], "hallo") == "User: hallo"
    history = [("mach licht an", "Welches Licht?"), ("küche", "Welche Helligkeit?")]
    assert stitch_context(history, "50 prozent") == (
        "User: mach licht an\n"
        "Assistant: Welches Licht?\n"
        "User: küche\n"
        "Assistant: Welche Helligkeit?\n"
        "User: 50 prozent"
    )


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
