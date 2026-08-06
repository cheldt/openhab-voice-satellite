"""Ring-buffer patch for openwakeword's streaming feature buffer.

openwakeword 0.6.0 keeps 10 s of raw audio in a deque of Python ints and
copies the WHOLE deque to a list on every 80 ms frame just to slice off the
newest ~1760 samples (utils.py `_buffer_raw_data`/`_streaming_melspectrogram`).
That is ~16 000 int objects created and ~2 million pointer copies per second
on the event-loop thread, holding the GIL. `use_ring_buffer` swaps the two
methods on our Model instance for a fixed numpy int16 ring with identical
semantics. The patch is version-gated and fails open: any mismatch leaves
stock behavior in place with a warning instead of crashing.
"""

from __future__ import annotations

import logging
from importlib import metadata

import numpy as np

log = logging.getLogger(__name__)

# the exact release whose utils.py internals the patch mirrors; keep in
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
    if not hasattr(features, "melspectrogram_buffer"):
        raise RuntimeError("no melspectrogram_buffer")
    return (
        features,
        capacity,
        features._get_melspectrogram,
        features.reset,
        features.melspectrogram_max_len,
    )


def use_ring_buffer(model) -> bool:
    """Swap `model.preprocessor`'s streaming buffer for an Int16Ring.

    All lookups happen before any assignment, so a failure leaves zero
    partial state; any surprise (version, missing seam) logs a warning and
    returns False with stock behavior intact.
    """
    try:
        features, capacity, get_melspec, stock_reset, max_len = _bind_seams(model)
    except Exception as exc:
        log.warning(
            "openwakeword ring-buffer patch skipped (%s); "
            "stock streaming buffer stays (slower)",
            exc,
        )
        return False

    ring = Int16Ring(capacity)

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
        stock_reset()
        ring.clear()

    # assignments last; they cannot fail, so the patch is all-or-nothing
    features._buffer_raw_data = buffer_raw_data
    features._streaming_melspectrogram = streaming_melspectrogram
    features.reset = reset
    log.debug("openwakeword streaming buffer replaced with numpy ring")
    return True
