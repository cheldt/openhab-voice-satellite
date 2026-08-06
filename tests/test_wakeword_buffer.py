"""Int16Ring semantics and the fail-open contract of use_ring_buffer.

The equivalence test drives a stock-copied deque implementation (verbatim
from openwakeword 0.6.0 utils.py) and the patched ring through identical
frame sequences and asserts the melspectrogram model would see identical
input — that is the whole correctness contract of the patch.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from openhab_voice_satellite import wakeword_buffer
from openhab_voice_satellite.wakeword_buffer import Int16Ring, use_ring_buffer

rng = np.random.default_rng(seed=42)


def _samples(n: int) -> np.ndarray:
    return rng.integers(-32768, 32767, n, dtype=np.int16)


class FakeFeatures:
    """Replicates the openwakeword 0.6.0 AudioFeatures seams the patch uses."""

    def __init__(self, maxlen: int = 4096) -> None:
        self.raw_data_buffer: deque = deque(maxlen=maxlen)
        self.melspectrogram_buffer = np.ones((76, 32))
        self.melspectrogram_max_len = 970
        self.melspec_inputs: list[np.ndarray] = []
        self.reset_calls = 0

    def _get_melspectrogram(self, x):
        arr = np.asarray(x, dtype=np.int16)
        self.melspec_inputs.append(arr.copy())
        return np.zeros((8, 32))

    # stock-copied from openwakeword 0.6.0 utils.py
    def _buffer_raw_data(self, x):
        self.raw_data_buffer.extend(x.tolist() if isinstance(x, np.ndarray) else x)

    def _streaming_melspectrogram(self, n_samples):
        if len(self.raw_data_buffer) < 400:
            raise ValueError("The number of input frames must be at least 400 samples @ 16khz (25 ms)!")
        self.melspectrogram_buffer = np.vstack(
            (self.melspectrogram_buffer,
             self._get_melspectrogram(list(self.raw_data_buffer)[-n_samples - 160 * 3:]))
        )
        if self.melspectrogram_buffer.shape[0] > self.melspectrogram_max_len:
            self.melspectrogram_buffer = self.melspectrogram_buffer[-self.melspectrogram_max_len:, :]

    def reset(self):
        self.reset_calls += 1
        self.raw_data_buffer.clear()
        self.melspectrogram_buffer = np.ones((76, 32))


class FakeModel:
    def __init__(self, features: FakeFeatures | None = None) -> None:
        if features is not None:
            self.preprocessor = features


@pytest.fixture
def patchable(monkeypatch):
    """Make the version gate pass regardless of the local environment."""
    monkeypatch.setattr(
        wakeword_buffer.metadata, "version", lambda name: wakeword_buffer.PATCHED_VERSION
    )


def test_ring_wraparound():
    ring = Int16Ring(8)
    ring.extend(np.arange(5, dtype=np.int16))
    ring.extend(np.arange(5, 10, dtype=np.int16))
    assert np.array_equal(ring.tail(6), np.arange(4, 10, dtype=np.int16))


def test_ring_tail_clamps_to_available():
    ring = Int16Ring(8)
    ring.extend(np.arange(3, dtype=np.int16))
    assert np.array_equal(ring.tail(100), np.arange(3, dtype=np.int16))


def test_ring_oversized_extend_keeps_newest():
    ring = Int16Ring(4)
    ring.extend(np.arange(10, dtype=np.int16))
    assert np.array_equal(ring.tail(4), np.arange(6, 10, dtype=np.int16))


def test_ring_accepts_list_input():
    ring = Int16Ring(8)
    ring.extend([1, 2, 3])
    assert np.array_equal(ring.tail(3), np.array([1, 2, 3], dtype=np.int16))


def test_ring_clear():
    ring = Int16Ring(8)
    ring.extend(np.arange(5, dtype=np.int16))
    ring.clear()
    assert len(ring) == 0
    assert ring.tail(5).size == 0


def test_patch_equivalence_with_stock_deque(patchable):
    stock = FakeFeatures()
    patched_features = FakeFeatures()
    assert use_ring_buffer(FakeModel(patched_features)) is True

    # 1280-frames, one >1280 chunk, enough total to exceed maxlen and wrap
    chunks = [1280, 1280, 2560, 1280, 1280, 1280]
    for n in chunks:
        data = _samples(n)
        stock._buffer_raw_data(data)
        stock._streaming_melspectrogram(n)
        patched_features._buffer_raw_data(data)
        patched_features._streaming_melspectrogram(n)

    assert len(stock.melspec_inputs) == len(patched_features.melspec_inputs)
    for ref, got in zip(stock.melspec_inputs, patched_features.melspec_inputs):
        assert np.array_equal(ref, got)
    assert np.array_equal(stock.melspectrogram_buffer, patched_features.melspectrogram_buffer)


def test_patch_fails_open_on_version_mismatch(monkeypatch, caplog):
    monkeypatch.setattr(wakeword_buffer.metadata, "version", lambda name: "0.7.0")
    features = FakeFeatures()
    with caplog.at_level("WARNING"):
        assert use_ring_buffer(FakeModel(features)) is False
    assert "patch skipped" in caplog.text
    assert "reset" not in vars(features)  # no partial patch


def test_patch_fails_open_on_missing_seam(patchable, caplog):
    with caplog.at_level("WARNING"):
        assert use_ring_buffer(FakeModel()) is False  # no .preprocessor at all
    features = FakeFeatures()
    del features.melspectrogram_max_len
    with caplog.at_level("WARNING"):
        assert use_ring_buffer(FakeModel(features)) is False
    for name in ("_buffer_raw_data", "_streaming_melspectrogram", "reset"):
        assert name not in vars(features)  # lookups before assignments


def test_patched_reset_clears_ring_and_calls_stock(patchable):
    features = FakeFeatures()
    assert use_ring_buffer(FakeModel(features)) is True
    features._buffer_raw_data(_samples(1280))
    features._streaming_melspectrogram(1280)  # works: ring holds samples
    features.reset()
    assert features.reset_calls == 1
    with pytest.raises(ValueError):  # ring is empty again (< 400 samples)
        features._streaming_melspectrogram(1280)
