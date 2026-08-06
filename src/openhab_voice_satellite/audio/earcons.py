"""Short feedback sounds (wake/ack/error/idle), pre-decoded to PCM at startup."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .sink import AudioSink
from .wav import read_wav_mono
from ..config import EarconsConfig, resolve_path

log = logging.getLogger(__name__)


class Earcons:
    def __init__(self, config: EarconsConfig, sink: AudioSink, base_dir: Path | None = None) -> None:
        self._sink = sink
        self._sounds: dict[str, tuple[np.ndarray, int]] = {}
        if not config.enabled:
            return
        base = base_dir or Path.cwd()
        for name, rel in (
            ("wake", config.wake),
            ("ack", config.ack),
            ("error", config.error),
            ("idle", config.idle),
        ):
            path = resolve_path(rel, base)
            if path.exists():
                self._sounds[name] = read_wav_mono(path)
            else:
                log.warning("earcon file missing, skipping: %s", path)

    async def play(self, name: str) -> None:
        sound = self._sounds.get(name)
        if sound is None:
            return
        pcm, rate = sound
        try:
            await self._sink.play(pcm, rate)
        except Exception:
            log.exception("earcon playback failed")
