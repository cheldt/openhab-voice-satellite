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
from openhab_voice_satellite.wakeword_buffer import Int16Ring, patch_preprocessor

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
        self.accumulated_samples = 0
        self.raw_data_remainder = np.empty(0)
        self.reset_calls = 0
        self.embed_calls = 0
        self.feature_buffer = self._get_embeddings()

    def _get_embeddings(self) -> np.ndarray:
        """Stands in for the 4 s random-audio ONNX inference at utils.py:178."""
        self.embed_calls += 1
        return np.full((41, 96), float(self.embed_calls), dtype=np.float32)

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
        self.accumulated_samples = 0
        self.raw_data_remainder = np.empty(0)
        self.feature_buffer = self._get_embeddings()


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
    assert patch_preprocessor(FakeModel(patched_features)) is not None

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
        assert patch_preprocessor(FakeModel(features)) is None
    assert "patch skipped" in caplog.text
    assert "reset" not in vars(features)  # no partial patch


def test_patch_fails_open_on_missing_seam(patchable, caplog):
    with caplog.at_level("WARNING"):
        assert patch_preprocessor(FakeModel()) is None  # no .preprocessor at all
    for missing in (
        "melspectrogram_max_len",
        "melspectrogram_buffer",
        "accumulated_samples",
        "raw_data_remainder",
        "feature_buffer",
    ):
        features = FakeFeatures()
        delattr(features, missing)
        with caplog.at_level("WARNING"):
            assert patch_preprocessor(FakeModel(features)) is None, missing
        for name in ("_buffer_raw_data", "_streaming_melspectrogram", "reset"):
            assert name not in vars(features)  # lookups before assignments


def test_patch_refuses_a_preprocessor_that_already_saw_audio(patchable, caplog):
    """The primed snapshot stands in for every later reset, so it must be pristine."""
    features = FakeFeatures()
    features._buffer_raw_data(_samples(1280))
    with caplog.at_level("WARNING"):
        assert patch_preprocessor(FakeModel(features)) is None
    assert "already fed audio" in caplog.text


def test_patched_reset_restores_primed_buffer_without_re_embedding(patchable):
    features = FakeFeatures()
    assert patch_preprocessor(FakeModel(features)) is not None
    primed = features.feature_buffer.copy()
    embeds_before = features.embed_calls

    features._buffer_raw_data(_samples(1280))
    features._streaming_melspectrogram(1280)  # works: ring holds samples
    features.accumulated_samples = 1280
    features.raw_data_remainder = _samples(7)
    features.reset()

    assert features.embed_calls == embeds_before  # the 4 s inference never ran
    assert features.reset_calls == 0  # stock reset is no longer delegated to
    assert np.array_equal(features.feature_buffer, primed)
    assert np.array_equal(features.melspectrogram_buffer, np.ones((76, 32)))
    assert features.accumulated_samples == 0
    assert features.raw_data_remainder.size == 0
    with pytest.raises(ValueError):  # ring is empty again (< 400 samples)
        features._streaming_melspectrogram(1280)


def _real_models() -> list[str]:
    from importlib import metadata
    from pathlib import Path

    # This runs at import time (the skipif below), so it must not raise:
    # openwakeword is deliberately absent from pyproject.toml and uv.lock
    # (its metadata hard-requires tflite-runtime), so CI has no such
    # distribution and an uncaught PackageNotFoundError would fail
    # collection of this whole module instead of skipping one test.
    try:
        installed = metadata.version("openwakeword")
    except metadata.PackageNotFoundError:
        return []
    if installed != wakeword_buffer.PATCHED_VERSION:
        return []
    root = Path(__file__).resolve().parent.parent / "models" / "wakeword"
    return [str(p) for p in sorted(root.glob("*.onnx"))[:1]]


@pytest.mark.skipif(not _real_models(), reason="no local wakeword model to drive")
def test_real_model_reset_matches_stock_trajectory():
    """The only test that actually proves the primed restore is equivalent.

    Snapshot-restore and the stock 4 s re-embed both leave feature_buffer
    holding embeddings of arbitrary noise, so post-reset scores must track.
    """
    from openwakeword.model import Model

    models = _real_models()
    frames = [np.zeros(1280, dtype=np.int16) for _ in range(20)]

    def trajectory(model) -> list[float]:
        model.reset()
        key = list(model.models.keys())[0]
        return [float(model.predict(f)[key]) for f in frames]

    stock = trajectory(Model(wakeword_models=models, inference_framework="onnx", ncpu=1))
    patched_model = Model(wakeword_models=models, inference_framework="onnx", ncpu=1)
    assert patch_preprocessor(patched_model) is not None
    patched = trajectory(patched_model)

    assert stock[:5] == [0.0] * 5 == patched[:5]  # the documented 5-frame mute
    assert np.allclose(stock, patched, atol=1e-3)


def test_patched_reset_hands_out_a_fresh_copy_each_time(patchable):
    features = FakeFeatures()
    assert patch_preprocessor(FakeModel(features)) is not None
    primed = features.feature_buffer.copy()

    features.reset()
    features.feature_buffer[0, 0] = 12345.0  # a caller mutating in place
    features.reset()

    assert np.array_equal(features.feature_buffer, primed)
