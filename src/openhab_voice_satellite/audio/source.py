"""Audio capture protocol; the PipeWire implementation lives in gst_source."""

from __future__ import annotations

from typing import AsyncIterator, Protocol

import numpy as np


class AudioSource(Protocol):
    # plain def: implementations are async *generator* functions, whose type
    # is Callable[[], AsyncIterator] — an `async def` here would demand a
    # coroutine that returns the iterator instead
    def frames(self) -> AsyncIterator[np.ndarray]:
        """Yield mono int16 frames of fixed length until the source closes."""
        ...

    def close(self) -> None: ...
