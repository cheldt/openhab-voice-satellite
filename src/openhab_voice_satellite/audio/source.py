"""Audio capture protocol; the PipeWire implementation lives in gst_source."""

from __future__ import annotations

from typing import AsyncIterator, Protocol

import numpy as np


class AudioSource(Protocol):
    async def frames(self) -> AsyncIterator[np.ndarray]:
        """Yield mono int16 frames of fixed length until the source closes."""
        ...

    def close(self) -> None: ...
