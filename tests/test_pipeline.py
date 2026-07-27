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
    def __init__(self, text: str = "schalte das licht an", language: str = "de") -> None:
        self._result = Transcript(text=text, language=language)

    async def transcribe(self, pcm: np.ndarray) -> Transcript:
        await asyncio.sleep(0.01)
        return self._result


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
                   states: list[State]) -> Pipeline:
    return Pipeline(
        config=config,
        broadcaster=broadcaster,
        endpointer=FakeEndpointer(speech_at=2, endpoint_at=5),
        transcriber=FakeTranscriber(),
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
