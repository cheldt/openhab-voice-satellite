"""Audio playback protocol; the PipeWire implementation lives in gst_sink."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class AudioSink(Protocol):
    async def play(self, pcm: np.ndarray, sample_rate: int) -> None:
        """Play mono int16 PCM to completion; must be cancellable."""
        ...

    def stop(self) -> None: ...

    def duck(self, factor: float) -> None: ...

    def unduck(self) -> None: ...
