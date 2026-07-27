import asyncio

import numpy as np
import pytest

from stt_proxy.config import VadConfig
from stt_proxy.recorder import NoSpeechError, record_utterance

FRAME = 1280  # 80 ms at 16 kHz


class FakeEndpointer:
    """Scripted endpointer: speech starts at frame `speech_at`, ends at `endpoint_at`."""

    def __init__(self, speech_at: int | None, endpoint_at: int | None) -> None:
        self._speech_at = speech_at
        self._endpoint_at = endpoint_at
        self.reset()

    def reset(self) -> None:
        self._frames = 0
        self.speech_started = False

    def update(self, frame: np.ndarray) -> bool:
        self._frames += 1
        if self._speech_at is not None and self._frames >= self._speech_at:
            self.speech_started = True
        return self.speech_started

    @property
    def endpoint_reached(self) -> bool:
        return self._endpoint_at is not None and self._frames >= self._endpoint_at

    @property
    def elapsed_s(self) -> float:
        return self._frames * FRAME / 16000


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
