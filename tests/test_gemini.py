import asyncio
import base64
import io
import wave

import aiohttp
import numpy as np
import pytest
from aiohttp.test_utils import TestServer

from openhab_voice_satellite.config import GeminiConfig, SttConfig, TtsConfig
from openhab_voice_satellite.fallback import FallbackSpeaker, FallbackTranscriber
from openhab_voice_satellite.audio.wav import pcm_to_wav_bytes
from openhab_voice_satellite.gemini import (
    GeminiClient,
    GeminiError,
    GeminiSpeaker,
    GeminiTranscriber,
)
from openhab_voice_satellite.stt import Transcript

from .fakes import BufferAudioSink, FakeGemini


@pytest.fixture
async def fake_gemini():
    fake = FakeGemini()
    server = TestServer(fake.build_app())
    await server.start_server(shutdown_timeout=0.2)
    yield fake, server
    await server.close()


@pytest.fixture
async def session():
    session = aiohttp.ClientSession()
    yield session
    await session.close()


def _config(server: TestServer, **overrides) -> GeminiConfig:
    return GeminiConfig(
        base_url=str(server.make_url("")), api_key="test-key", **overrides
    )


def _transcriber(fake_gemini, session, **stt_overrides) -> GeminiTranscriber:
    _, server = fake_gemini
    client = GeminiClient(_config(server), session)
    return GeminiTranscriber(client, SttConfig(**stt_overrides), "de")


def _speaker(fake_gemini, session, sink, **overrides) -> GeminiSpeaker:
    _, server = fake_gemini
    config = _config(server, **overrides)
    client = GeminiClient(config, session)
    return GeminiSpeaker(client, TtsConfig(), sink)


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


def test_pcm_to_wav_roundtrip():
    pcm = np.arange(-100, 100, dtype=np.int16)
    data = pcm_to_wav_bytes(pcm, 16000)
    with wave.open(io.BytesIO(data), "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        decoded = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    assert np.array_equal(decoded, pcm)


async def test_transcribe(fake_gemini, session):
    fake, _ = fake_gemini
    fake.stt_text, fake.stt_language = "schalte das licht an", "de"
    transcriber = _transcriber(fake_gemini, session)
    result = await transcriber.transcribe(np.zeros(1600, dtype=np.int16))
    assert result == Transcript(text="schalte das licht an", language="de")
    assert fake.api_keys == ["test-key"]

    model, payload = fake.requests[0]
    assert model == "gemini-3.6-flash"
    # uploaded inline data is a valid wav containing the input pcm
    wav_b64 = payload["contents"][0]["parts"][1]["inlineData"]["data"]
    with wave.open(io.BytesIO(base64.b64decode(wav_b64)), "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnframes() == 1600
    schema = payload["generationConfig"]["responseSchema"]
    assert schema["properties"]["language"]["enum"] == ["de", "en"]


async def test_transcribe_clamps_unknown_language(fake_gemini, session):
    fake, _ = fake_gemini
    fake.stt_language = "fr"
    transcriber = _transcriber(fake_gemini, session)
    result = await transcriber.transcribe(np.zeros(160, dtype=np.int16))
    assert result.language == "de"  # default


async def test_transcribe_survives_malformed_json(fake_gemini, session):
    fake, _ = fake_gemini
    fake.stt_response = "not json at all"
    transcriber = _transcriber(fake_gemini, session)
    result = await transcriber.transcribe(np.zeros(160, dtype=np.int16))
    assert result == Transcript(text="not json at all", language="de")


async def test_speak(fake_gemini, session):
    fake, _ = fake_gemini
    sink = BufferAudioSink()
    speaker = _speaker(fake_gemini, session, sink)
    await speaker.speak("Licht ist an.", "de")
    assert len(sink.played) == 1
    pcm, rate = sink.played[0]
    assert rate == 24000
    assert np.array_equal(pcm, fake.tts_pcm)

    model, payload = fake.requests[0]
    assert model == "gemini-3.1-flash-tts-preview"
    voice = payload["generationConfig"]["speechConfig"]["voiceConfig"][
        "prebuiltVoiceConfig"
    ]["voiceName"]
    assert voice == "Kore"


async def test_speak_voice_per_language(fake_gemini, session):
    fake, _ = fake_gemini
    speaker = _speaker(fake_gemini, session, BufferAudioSink())
    await speaker.speak("hello", "en")
    assert (
        fake.requests[0][1]["generationConfig"]["speechConfig"]["voiceConfig"][
            "prebuiltVoiceConfig"
        ]["voiceName"]
        == "Puck"
    )


async def test_speak_unknown_language_uses_default_voice(fake_gemini, session):
    fake, _ = fake_gemini
    speaker = _speaker(fake_gemini, session, BufferAudioSink())
    await speaker.speak("bonjour", "fr")
    assert (
        fake.requests[0][1]["generationConfig"]["speechConfig"]["voiceConfig"][
            "prebuiltVoiceConfig"
        ]["voiceName"]
        == "Kore"  # tts default_language is "de"
    )


async def test_speak_uses_rate_from_mime(fake_gemini, session):
    fake, _ = fake_gemini
    fake.tts_mime = "audio/L16;codec=pcm;rate=48000"
    sink = BufferAudioSink()
    speaker = _speaker(fake_gemini, session, sink)
    await speaker.speak("hi", "de")
    assert sink.played[0][1] == 48000


async def test_check_model(fake_gemini, session):
    fake, server = fake_gemini
    client = GeminiClient(_config(server), session)
    await client.check_model("gemini-3.6-flash")
    assert fake.checked_models == ["gemini-3.6-flash"]

    fake.status = 403
    with pytest.raises(GeminiError):
        await client.check_model("gemini-3.6-flash")


async def test_fallback_transcriber_on_http_error(fake_gemini, session, caplog):
    fake, _ = fake_gemini
    fake.status = 500
    local = LocalTranscriberStub()
    wrapper = FallbackTranscriber(_transcriber(fake_gemini, session), local)
    result = await wrapper.transcribe(np.zeros(160, dtype=np.int16))
    assert result.text == "local fallback"
    assert len(local.calls) == 1
    assert "falling back to local" in caplog.text


async def test_fallback_transcriber_on_timeout(fake_gemini, session):
    fake, server = fake_gemini
    fake.response_delay_s = 10
    config = _config(server, stt_timeout_s=0.2)
    transcriber = GeminiTranscriber(GeminiClient(config, session), SttConfig(), "de")
    local = LocalTranscriberStub()
    wrapper = FallbackTranscriber(transcriber, local)
    result = await wrapper.transcribe(np.zeros(160, dtype=np.int16))
    assert result.text == "local fallback"


async def test_fallback_transcriber_on_connection_error(session):
    config = GeminiConfig(base_url="http://127.0.0.1:1", api_key="k")
    transcriber = GeminiTranscriber(GeminiClient(config, session), SttConfig(), "de")
    local = LocalTranscriberStub()
    wrapper = FallbackTranscriber(transcriber, local)
    result = await wrapper.transcribe(np.zeros(160, dtype=np.int16))
    assert result.text == "local fallback"


async def test_fallback_speaker_on_http_error(fake_gemini, session, caplog):
    fake, _ = fake_gemini
    fake.status = 500
    local = LocalSpeakerStub()
    wrapper = FallbackSpeaker(_speaker(fake_gemini, session, BufferAudioSink()), local)
    await wrapper.speak("Licht ist an.", "de")
    assert local.spoken == [("Licht ist an.", "de")]
    assert "falling back to local" in caplog.text


async def test_fallback_propagates_cancellation(fake_gemini, session):
    fake, _ = fake_gemini
    fake.response_delay_s = 10
    local = LocalSpeakerStub()
    wrapper = FallbackSpeaker(_speaker(fake_gemini, session, BufferAudioSink()), local)
    task = asyncio.create_task(wrapper.speak("hi", "de"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert local.spoken == []  # barge-in must not trigger the fallback
