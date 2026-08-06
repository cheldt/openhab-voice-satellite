import asyncio

import numpy as np
import pytest

from openhab_voice_satellite import recorder
from openhab_voice_satellite.config import VadConfig
from openhab_voice_satellite.recorder import NoSpeechError, record_utterance

from .fakes import FRAME, FakeEndpointer


async def _fill_queue(n_frames: int) -> asyncio.Queue:
    queue: asyncio.Queue = asyncio.Queue()
    for _ in range(n_frames):
        queue.put_nowait(np.zeros(FRAME, dtype=np.int16))
    return queue


async def test_records_until_endpoint():
    queue = await _fill_queue(50)
    endpointer = FakeEndpointer(speech_at=2, endpoint_at=10)
    pcm = await record_utterance(queue, endpointer, VadConfig())
    assert len(pcm) == 10 * FRAME


async def test_no_speech_timeout():
    queue = await _fill_queue(200)
    endpointer = FakeEndpointer(speech_at=None, endpoint_at=None)
    config = VadConfig(no_speech_timeout_s=1.0)
    with pytest.raises(NoSpeechError):
        await record_utterance(queue, endpointer, config)


async def test_no_speech_timeout_override():
    queue = await _fill_queue(200)
    endpointer = FakeEndpointer(speech_at=None, endpoint_at=None)
    config = VadConfig(no_speech_timeout_s=100.0)
    with pytest.raises(NoSpeechError):
        await record_utterance(queue, endpointer, config, no_speech_timeout_s=0.5)
    assert endpointer.elapsed_s < 1.0


async def test_max_utterance_cutoff():
    queue = await _fill_queue(200)
    endpointer = FakeEndpointer(speech_at=1, endpoint_at=None)
    config = VadConfig(max_utterance_s=2.0)
    pcm = await record_utterance(queue, endpointer, config)
    assert len(pcm) / 16000 == pytest.approx(2.0, abs=0.1)


async def test_source_closed_raises():
    queue: asyncio.Queue = asyncio.Queue()
    queue.put_nowait(None)
    endpointer = FakeEndpointer(speech_at=None, endpoint_at=None)
    with pytest.raises(NoSpeechError):
        await record_utterance(queue, endpointer, VadConfig())


async def test_mic_stall_raises_within_wall_clock_timeout(monkeypatch):
    # sample-counted timeouts can never fire on a silent queue; the
    # wall-clock backstop must exit LISTENING instead of hanging forever
    monkeypatch.setattr(recorder, "MIC_STALL_TIMEOUT_S", 0.1)
    queue: asyncio.Queue = asyncio.Queue()  # never delivers a frame
    endpointer = FakeEndpointer(speech_at=None, endpoint_at=None)
    with pytest.raises(NoSpeechError, match="stalled"):
        await asyncio.wait_for(
            record_utterance(queue, endpointer, VadConfig()), timeout=2.0
        )


async def test_mic_stall_mid_speech_returns_partial(monkeypatch):
    monkeypatch.setattr(recorder, "MIC_STALL_TIMEOUT_S", 0.1)
    queue = await _fill_queue(5)  # speech starts, then the mic goes silent
    endpointer = FakeEndpointer(speech_at=1, endpoint_at=None)
    pcm = await asyncio.wait_for(
        record_utterance(queue, endpointer, VadConfig()), timeout=2.0
    )
    assert len(pcm) == 5 * FRAME


async def test_dropped_frames_warned(caplog):
    queue = await _fill_queue(10)
    queue.dropped = 3  # what a SubscriberQueue reports after backpressure
    endpointer = FakeEndpointer(speech_at=1, endpoint_at=5)
    with caplog.at_level("WARNING"):
        await record_utterance(queue, endpointer, VadConfig())
    assert "dropped mic frames" in caplog.text
