import asyncio
import io
import wave

import aiohttp
import numpy as np
import pytest
from aiohttp.test_utils import TestServer

from stt_proxy.config import DeepgramConfig, SttConfig, TtsConfig
from stt_proxy.deepgram import (
    TTS_CHUNK_CHARS,
    DeepgramClient,
    DeepgramError,
    DeepgramSpeaker,
    DeepgramTranscriber,
    tts_chunks,
)
from stt_proxy.fallback import FallbackSpeaker, FallbackTranscriber
from stt_proxy.stt import Transcript

from .fakes import BufferAudioSink, FakeDeepgram


@pytest.fixture
async def fake_deepgram():
    fake = FakeDeepgram()
    server = TestServer(fake.build_app())
    await server.start_server(shutdown_timeout=0.2)
    yield fake, server
    await server.close()


@pytest.fixture
async def session():
    session = aiohttp.ClientSession()
    yield session
    await session.close()


def _config(server: TestServer, **overrides) -> DeepgramConfig:
    return DeepgramConfig(
        base_url=str(server.make_url("")), api_key="test-key", **overrides
    )


def _transcriber(fake_deepgram, session, **stt_overrides) -> DeepgramTranscriber:
    _, server = fake_deepgram
    client = DeepgramClient(_config(server), session)
    return DeepgramTranscriber(client, SttConfig(**stt_overrides), "de")


def _speaker(fake_deepgram, session, sink, **overrides) -> DeepgramSpeaker:
    _, server = fake_deepgram
    config = _config(server, **overrides)
    client = DeepgramClient(config, session)
    return DeepgramSpeaker(client, config, TtsConfig(), sink)


class LocalTranscriberStub:
    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    async def transcribe(self, pcm: np.ndarray) -> Transcript:
        self.calls.append(pcm)
        return Transcript(text="local fallback", language="de")


class LocalSpeakerStub:
    def __init__(self) -> None:
        self.spoken: list[tuple[str, str]] = []

    async def speak(self, text: str, language: str) -> None:
        self.spoken.append((text, language))


async def test_transcribe(fake_deepgram, session):
    fake, _ = fake_deepgram
    fake.stt_text, fake.stt_language = "schalte das licht an", "de"
    transcriber = _transcriber(fake_deepgram, session)
    result = await transcriber.transcribe(np.zeros(1600, dtype=np.int16))
    assert result == Transcript(text="schalte das licht an", language="de")
    assert fake.auth_headers == ["Token test-key"]

    query, body = fake.listen_requests[0]
    assert ("model", "nova-3") in query
    assert ("smart_format", "true") in query
    # both configured languages restrict the detection candidate set
    assert ("detect_language", "de") in query
    assert ("detect_language", "en") in query
    assert not any(k == "language" for k, _ in query)
    # uploaded body is a valid wav containing the input pcm
    with wave.open(io.BytesIO(body), "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnframes() == 1600


async def test_transcribe_single_language_sends_language_param(fake_deepgram, session):
    fake, _ = fake_deepgram
    transcriber = _transcriber(fake_deepgram, session, languages=["de"])
    result = await transcriber.transcribe(np.zeros(160, dtype=np.int16))
    query, _ = fake.listen_requests[0]
    assert ("language", "de") in query
    assert not any(k == "detect_language" for k, _ in query)
    assert result.language == "de"


async def test_transcribe_clamps_unknown_language(fake_deepgram, session):
    fake, _ = fake_deepgram
    fake.stt_language = "fr"
    transcriber = _transcriber(fake_deepgram, session)
    result = await transcriber.transcribe(np.zeros(160, dtype=np.int16))
    assert result.language == "de"  # default


async def test_transcribe_strips_bcp47_region(fake_deepgram, session):
    fake, _ = fake_deepgram
    fake.stt_language = "en-US"
    transcriber = _transcriber(fake_deepgram, session)
    result = await transcriber.transcribe(np.zeros(160, dtype=np.int16))
    assert result.language == "en"


async def test_speak(fake_deepgram, session):
    fake, _ = fake_deepgram
    sink = BufferAudioSink()
    speaker = _speaker(fake_deepgram, session, sink)
    await speaker.speak("Licht ist an.", "de")
    assert len(sink.played) == 1
    pcm, rate = sink.played[0]
    assert rate == 24000
    assert np.array_equal(pcm, fake.tts_pcm)

    query, payload = fake.speak_requests[0]
    assert payload == {"text": "Licht ist an."}
    assert ("model", "aura-2-viktoria-de") in query
    assert ("encoding", "linear16") in query
    assert ("sample_rate", "24000") in query
    assert ("container", "none") in query


async def test_speak_voice_per_language(fake_deepgram, session):
    fake, _ = fake_deepgram
    speaker = _speaker(fake_deepgram, session, BufferAudioSink())
    await speaker.speak("hello", "en")
    query, _ = fake.speak_requests[0]
    assert ("model", "aura-2-thalia-en") in query


async def test_speak_unknown_language_uses_default_voice(fake_deepgram, session):
    fake, _ = fake_deepgram
    speaker = _speaker(fake_deepgram, session, BufferAudioSink())
    await speaker.speak("bonjour", "fr")
    query, _ = fake.speak_requests[0]
    assert ("model", "aura-2-viktoria-de") in query  # tts default_language is "de"


def test_tts_chunks_splits_sentences():
    assert tts_chunks("Licht ist an. Rollladen fährt hoch.") == [
        "Licht ist an.",
        "Rollladen fährt hoch.",
    ]


def test_tts_chunks_splits_long_comma_list():
    # a "list all items" answer: one giant comma sentence
    text = ", ".join(f"item number {i} living room lamp" for i in range(80))
    chunks = tts_chunks(text)
    assert len(chunks) > 1
    assert all(len(c) <= TTS_CHUNK_CHARS for c in chunks)
    # nothing lost apart from separators
    assert "".join(chunks).replace(" ", "").replace(",", "") == text.replace(
        " ", ""
    ).replace(",", "")


async def test_speak_long_text_pipelines_chunks(fake_deepgram, session):
    fake, _ = fake_deepgram
    sink = BufferAudioSink()
    speaker = _speaker(fake_deepgram, session, sink)
    text = ", ".join(f"item number {i} living room lamp" for i in range(80))
    await speaker.speak(text, "en")
    assert len(fake.speak_requests) == len(tts_chunks(text))
    assert len(sink.played) == len(fake.speak_requests)
    assert all(len(payload["text"]) <= TTS_CHUNK_CHARS for _, payload in fake.speak_requests)


async def test_speak_midstream_failure_falls_back_with_remainder(
    fake_deepgram, session, caplog
):
    fake, _ = fake_deepgram
    fake.speak_statuses = [200, 500]  # first chunk plays, second fails
    sink = BufferAudioSink()
    local = LocalSpeakerStub()
    wrapper = FallbackSpeaker(_speaker(fake_deepgram, session, sink), local, label="deepgram")
    await wrapper.speak("First sentence. Second sentence. Third sentence.", "en")
    assert len(sink.played) == 1  # only the first chunk came from deepgram
    assert local.spoken == [("Second sentence. Third sentence.", "en")]
    assert "local speaks the rest" in caplog.text


async def test_check_auth(fake_deepgram, session):
    fake, server = fake_deepgram
    client = DeepgramClient(_config(server), session)
    await client.check_auth()
    assert fake.auth_headers == ["Token test-key"]

    fake.status = 403
    with pytest.raises(DeepgramError):
        await client.check_auth()


async def test_fallback_transcriber_on_http_error(fake_deepgram, session, caplog):
    fake, _ = fake_deepgram
    fake.status = 500
    local = LocalTranscriberStub()
    wrapper = FallbackTranscriber(
        _transcriber(fake_deepgram, session), local, label="deepgram"
    )
    result = await wrapper.transcribe(np.zeros(160, dtype=np.int16))
    assert result.text == "local fallback"
    assert len(local.calls) == 1
    assert "deepgram STT failed" in caplog.text


async def test_fallback_transcriber_on_timeout(fake_deepgram, session):
    fake, server = fake_deepgram
    fake.response_delay_s = 10
    config = _config(server, stt_timeout_s=0.2)
    transcriber = DeepgramTranscriber(DeepgramClient(config, session), SttConfig(), "de")
    local = LocalTranscriberStub()
    wrapper = FallbackTranscriber(transcriber, local)
    result = await wrapper.transcribe(np.zeros(160, dtype=np.int16))
    assert result.text == "local fallback"


async def test_fallback_transcriber_on_connection_error(session):
    config = DeepgramConfig(base_url="http://127.0.0.1:1", api_key="k")
    transcriber = DeepgramTranscriber(DeepgramClient(config, session), SttConfig(), "de")
    local = LocalTranscriberStub()
    wrapper = FallbackTranscriber(transcriber, local)
    result = await wrapper.transcribe(np.zeros(160, dtype=np.int16))
    assert result.text == "local fallback"


async def test_fallback_speaker_on_http_error(fake_deepgram, session, caplog):
    fake, _ = fake_deepgram
    fake.status = 500
    local = LocalSpeakerStub()
    wrapper = FallbackSpeaker(
        _speaker(fake_deepgram, session, BufferAudioSink()), local, label="deepgram"
    )
    await wrapper.speak("Licht ist an.", "de")
    assert local.spoken == [("Licht ist an.", "de")]
    assert "deepgram TTS failed" in caplog.text


async def test_fallback_propagates_cancellation(fake_deepgram, session):
    fake, _ = fake_deepgram
    fake.response_delay_s = 10
    local = LocalSpeakerStub()
    wrapper = FallbackSpeaker(_speaker(fake_deepgram, session, BufferAudioSink()), local)
    task = asyncio.create_task(wrapper.speak("hi", "de"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert local.spoken == []  # barge-in must not trigger the fallback
