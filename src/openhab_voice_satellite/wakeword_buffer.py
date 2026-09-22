"""Performance patches for openwakeword's audio preprocessor.

Two independent hot spots, both patched onto our Model instance under one
version gate. Either way the patch fails open: any mismatch leaves stock
behavior in place with a warning instead of crashing.

1. Streaming buffer. openwakeword 0.6.0 keeps 10 s of raw audio in a deque of
   Python ints and copies the WHOLE deque to a list on every 80 ms frame just
   to slice off the newest ~1760 samples (utils.py `_buffer_raw_data` /
   `_streaming_melspectrogram`). That is ~16 000 int objects created and
   ~2 million pointer copies per second on the event-loop thread, holding the
   GIL. Replaced with a fixed numpy int16 ring with identical semantics.

2. Reset. `AudioFeatures.reset()` ends by re-embedding four seconds of freshly
   drawn random audio (utils.py:178) — a synchronous ONNX inference measured at
   35-39 ms on x86, and we call it on the event loop at every detection and
   every barge-in. Upstream's own docstring concedes it "may not be efficient
   when called too frequently". Since that buffer only ever holds embeddings of
   arbitrary noise, the draw computed at construction is just as valid as a
   fresh one; we snapshot it once and restore a copy instead.
"""

from __future__ import annotations

import logging
from importlib import metadata

import numpy as np

log = logging.getLogger(__name__)

# the exact release whose utils.py internals the patches mirror; keep in
# sync with the openwakeword pin in deploy/install.md
PATCHED_VERSION = "0.6.0"


class Int16Ring:
    """Fixed-capacity int16 ring buffer replacing the stock deque."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._buf = np.zeros(capacity, dtype=np.int16)
        self._write = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def extend(self, x) -> None:
        x = np.asarray(x, dtype=np.int16)
        n = len(x)
        if n >= self._capacity:
            # oversized chunk keeps only the newest samples (deque maxlen)
            self._buf[:] = x[-self._capacity:]
            self._write = 0
            self._size = self._capacity
            return
        end = self._write + n
        if end <= self._capacity:
            self._buf[self._write:end] = x
        else:
            split = self._capacity - self._write
            self._buf[self._write:] = x[:split]
            self._buf[:end - self._capacity] = x[split:]
        self._write = end % self._capacity
        self._size = min(self._size + n, self._capacity)

    def tail(self, n: int) -> np.ndarray:
        """Newest `n` samples; all available if fewer (like a list slice).

        Contiguous case returns a zero-copy view — safe because the caller
        (`_get_melspectrogram`) copies via astype before inference.
        """
        n = min(n, self._size)
        start = (self._write - n) % self._capacity
        if start + n <= self._capacity:
            return self._buf[start:start + n]
        # wrapped: start+n ≡ _write (mod capacity), so the front part always
        # ends exactly at the write cursor
        return np.concatenate((self._buf[start:], self._buf[:self._write]))

    def clear(self) -> None:
        self._write = 0
        self._size = 0


def _bind_seams(model):
    """Look up every seam the patch relies on; raises before any assignment."""
    installed = metadata.version("openwakeword")
    if installed != PATCHED_VERSION:
        raise RuntimeError(
            f"openwakeword {installed} != patched version {PATCHED_VERSION}"
        )
    features = model.preprocessor
    capacity = features.raw_data_buffer.maxlen
    if capacity is None:
        raise RuntimeError("raw_data_buffer has no maxlen")
    for name in ("melspectrogram_buffer", "accumulated_samples", "raw_data_remainder"):
        if not hasattr(features, name):
            raise RuntimeError(f"no {name}")
    if not isinstance(getattr(features, "feature_buffer", None), np.ndarray):
        raise RuntimeError("feature_buffer is not an ndarray")
    # the snapshot below stands in for every future reset, so it is only
    # honest while the preprocessor is still in its construction state
    if len(features.raw_data_buffer) or features.accumulated_samples:
        raise RuntimeError("preprocessor already fed audio; primed buffer unavailable")
    return (
        features,
        capacity,
        features._get_melspectrogram,
        features.melspectrogram_max_len,
    )


def patch_preprocessor(model) -> Int16Ring | None:
    """Patch `model.preprocessor`'s streaming buffer and reset path.

    Returns the installed ring (also the caller's window onto the last 10 s of
    raw audio), or None when the patch was skipped. All lookups happen before
    any assignment, so a failure leaves zero partial state; any surprise
    (version, missing seam) logs a warning and leaves stock behavior intact.
    """
    try:
        features, capacity, get_melspec, max_len = _bind_seams(model)
    except Exception as exc:
        log.warning(
            "openwakeword preprocessor patch skipped (%s); "
            "stock streaming buffer and 4 s re-embedding reset stay (slower)",
            exc,
        )
        return None

    ring = Int16Ring(capacity)
    primed = features.feature_buffer.copy()

    def buffer_raw_data(x) -> None:
        ring.extend(x)

    def streaming_melspectrogram(n_samples: int) -> None:
        # mirror of openwakeword 0.6.0 utils.py minus the full-buffer copy
        if len(ring) < 400:
            raise ValueError(
                "The number of input frames must be at least 400 samples @ 16khz (25 ms)!"
            )
        features.melspectrogram_buffer = np.vstack(
            (features.melspectrogram_buffer, get_melspec(ring.tail(n_samples + 160 * 3)))
        )
        if features.melspectrogram_buffer.shape[0] > max_len:
            features.melspectrogram_buffer = features.melspectrogram_buffer[-max_len:, :]

    def reset() -> None:
        # mirror of openwakeword 0.6.0 utils.py reset() minus its final line,
        # which re-embeds 4 s of fresh noise at ~35 ms a call
        features.raw_data_buffer.clear()
        features.melspectrogram_buffer = np.ones((76, 32))
        features.accumulated_samples = 0
        features.raw_data_remainder = np.empty(0)
        features.feature_buffer = primed.copy()
        ring.clear()

    # assignments last; they cannot fail, so the patch is all-or-nothing
    features._buffer_raw_data = buffer_raw_data
    features._streaming_melspectrogram = streaming_melspectrogram
    features.reset = reset
    log.debug("openwakeword preprocessor patched: numpy ring + primed reset")
    return ring
