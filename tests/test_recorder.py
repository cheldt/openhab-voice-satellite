import asyncio

import numpy as np
import pytest

from stt_proxy.config import VadConfig
from stt_proxy.recorder import NoSpeechError, record_utterance

FRAME = 1280  # 80 ms at 16 kHz


class FakeEndpointer:
    """Scripted endpointer: speech starts at frame `speech_at`, ends at `endpoint_at`.

    `later` holds (speech_at, endpoint_at) scripts for follow-up recordings;
    each `record_utterance` reset advances to the next one (last repeats).
    """

    def __init__(
        self,
        speech_at: int | None,
        endpoint_at: int | None,
        later: list[tuple[int | None, int | None]] | None = None,
    ) -> None:
        self._scripts = [(speech_at, endpoint_at)] + list(later or [])
        self._resets = 0
        self.reset()

    def reset(self) -> None:
        # __init__ and the first recording both use scripts[0].
        idx = min(max(self._resets - 1, 0), len(self._scripts) - 1)
        self._speech_at, self._endpoint_at = self._scripts[idx]
        self._resets += 1
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
