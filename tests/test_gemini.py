import base64
import io
import wave

import aiohttp
import numpy as np
import pytest
from aiohttp.test_utils import TestServer

from openhab_voice_satellite.config import GeminiConfig, SttConfig, TtsConfig
from openhab_voice_satellite.gemini import (
    GeminiClient,
    GeminiError,
    GeminiSpeaker,
    GeminiTranscriber,
)
from openhab_voice_satellite.stt import Transcript

from .fakes import BufferAudioSink, FakeGemini

STT_MODEL = "gemini-stt-test"
TTS_MODEL = "gemini-tts-test"


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
    overrides.setdefault("stt_model", STT_MODEL)
    overrides.setdefault("tts_model", TTS_MODEL)
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


async def test_transcribe(fake_gemini, session):
    fake, _ = fake_gemini
    fake.stt_text, fake.stt_language = "schalte das licht an", "de"
    transcriber = _transcriber(fake_gemini, session)
    result = await transcriber.transcribe(np.zeros(1600, dtype=np.int16))
    assert result == Transcript(text="schalte das licht an", language="de")
    assert fake.api_keys == ["test-key"]

    model, payload = fake.requests[0]
    assert model == STT_MODEL
    # uploaded inline data is a valid wav containing the input pcm
    wav_b64 = payload["contents"][0]["parts"][1]["inlineData"]["data"]
    with wave.open(io.BytesIO(base64.b64decode(wav_b64)), "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnframes() == 1600
    schema = payload["generationConfig"]["responseSchema"]
    assert schema["properties"]["language"]["enum"] == ["de", "en"]


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
    assert model == TTS_MODEL
    voice = payload["generationConfig"]["speechConfig"]["voiceConfig"][
        "prebuiltVoiceConfig"
    ]["voiceName"]
    assert voice == "Kore"


async def test_speak_voice_per_language(fake_gemini, session):
    # the per-language voice reaches the wire in gemini's payload shape
    fake, _ = fake_gemini
    speaker = _speaker(
        fake_gemini, session, BufferAudioSink(),
        tts_voices={"de": "Kore", "en": "Puck"},
    )
    await speaker.speak("hello", "en")
    assert (
        fake.requests[0][1]["generationConfig"]["speechConfig"]["voiceConfig"][
            "prebuiltVoiceConfig"
        ]["voiceName"]
        == "Puck"
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
    await client.check_model(STT_MODEL)
    assert fake.checked_models == [STT_MODEL]

    fake.status = 403
    with pytest.raises(GeminiError):
        await client.check_model(STT_MODEL)
