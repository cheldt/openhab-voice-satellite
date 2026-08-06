import asyncio

import numpy as np
import pytest

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
