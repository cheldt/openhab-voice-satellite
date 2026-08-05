"""Re-chunk arbitrary-length sample buffers into fixed-length frames."""

from __future__ import annotations

import numpy as np


class FrameChunker:
    """Accumulates int16 samples and emits frames of exactly `frame_samples`."""

    def __init__(self, frame_samples: int) -> None:
        self._frame_samples = frame_samples
        self._rest = np.empty(0, dtype=np.int16)

    def push(self, samples: np.ndarray) -> list[np.ndarray]:
        buf = np.concatenate([self._rest, samples])
        n = self._frame_samples
        n_full = len(buf) // n
        # copies: gst buffers are unmapped after the callback returns
        frames = [buf[i * n:(i + 1) * n].copy() for i in range(n_full)]
        self._rest = buf[n_full * n:].copy()
        return frames

    def reset(self) -> None:
        self._rest = np.empty(0, dtype=np.int16)
