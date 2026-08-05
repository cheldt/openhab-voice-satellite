"""Short feedback sounds (wake/ack/error), pre-decoded to PCM at startup."""

from __future__ import annotations

import logging
import wave
from pathlib import Path

import numpy as np

from .sink import AudioSink
from ..config import EarconsConfig

log = logging.getLogger(__name__)


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError(f"{path}: earcons must be 16-bit PCM")
        rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
        pcm = np.frombuffer(raw, dtype=np.int16)
        if wav.getnchannels() > 1:
            pcm = pcm.reshape(-1, wav.getnchannels())[:, 0].copy()
    return pcm, rate


class Earcons:
    def __init__(self, config: EarconsConfig, sink: AudioSink, base_dir: Path | None = None) -> None:
        self._sink = sink
        self._sounds: dict[str, tuple[np.ndarray, int]] = {}
        if not config.enabled:
            return
        base = base_dir or Path.cwd()
        for name, rel in (("wake", config.wake), ("ack", config.ack), ("error", config.error)):
            path = Path(rel) if Path(rel).is_absolute() else base / rel
            if path.exists():
                self._sounds[name] = _load_wav(path)
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
