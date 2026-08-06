"""fallback.py is provider-agnostic — test it against stub primaries, no HTTP."""

import asyncio

import aiohttp
import numpy as np
import pytest

from openhab_voice_satellite.fallback import (
    CloudEngineError,
    FallbackSpeaker,
    FallbackTranscriber,
    PartialSpeechError,
)

from .fakes import LocalSpeakerStub, LocalTranscriberStub

PCM = np.zeros(160, dtype=np.int16)


class FailingTranscriber:
    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    async def transcribe(self, pcm):
        raise self._exc


class FailingSpeaker:
    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    async def speak(self, text, language):
        raise self._exc


class HangingSpeaker:
    async def speak(self, text, language):
        await asyncio.sleep(10)


@pytest.mark.parametrize(
    "exc",
    [
        CloudEngineError("HTTP 500 from /v1/listen: boom"),
        TimeoutError(),
        aiohttp.ClientConnectionError("connection refused"),
    ],
    ids=["http-error", "timeout", "connection-error"],
)
async def test_transcriber_falls_back(exc, caplog):
    local = LocalTranscriberStub()
    wrapper = FallbackTranscriber(FailingTranscriber(exc), local, label="cloud")
    result = await wrapper.transcribe(PCM)
    assert result.text == "local fallback"
    assert len(local.calls) == 1
    assert "falling back to local" in caplog.text


async def test_speaker_falls_back_and_respeaks_everything(caplog):
    local = LocalSpeakerStub()
    wrapper = FallbackSpeaker(FailingSpeaker(CloudEngineError("boom")), local)
    await wrapper.speak("Licht ist an.", "de")
    assert local.spoken == [("Licht ist an.", "de")]
    assert "falling back to local" in caplog.text


async def test_speaker_partial_failure_speaks_only_remainder(caplog):
    local = LocalSpeakerStub()
    exc = PartialSpeechError("HTTP 500", remaining="Second sentence. Third sentence.")
    wrapper = FallbackSpeaker(FailingSpeaker(exc), local)
    await wrapper.speak("First sentence. Second sentence. Third sentence.", "en")
    assert local.spoken == [("Second sentence. Third sentence.", "en")]
    assert "local speaks the rest" in caplog.text


async def test_unexpected_error_propagates():
    local = LocalTranscriberStub()
    wrapper = FallbackTranscriber(FailingTranscriber(ValueError("bug")), local)
    with pytest.raises(ValueError):
        await wrapper.transcribe(PCM)
    assert local.calls == []  # a code bug must not look like a cloud outage


async def test_cancellation_propagates_without_fallback():
    local = LocalSpeakerStub()
    wrapper = FallbackSpeaker(HangingSpeaker(), local)
    task = asyncio.create_task(wrapper.speak("hi", "de"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert local.spoken == []  # barge-in must not trigger the fallback
