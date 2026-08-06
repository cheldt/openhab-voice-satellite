"""Shared 16-bit PCM helpers: WAV read/write/encode and RMS."""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np


def rms(pcm: np.ndarray) -> int:
    """RMS amplitude of int16 PCM; 0 for empty input.

    The float64 cast is load-bearing: squaring int16 overflows.
    """
    if not len(pcm):
        return 0
    return int(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))


def pcm_to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Encode mono int16 PCM as an in-memory WAV (for cloud STT uploads)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.astype(np.int16).tobytes())
    return buf.getvalue()


def write_wav(path: Path, pcm: np.ndarray, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    """Read a 16-bit WAV; multichannel is downmixed by taking channel 0."""
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM")
        rate = wav.getframerate()
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
        if wav.getnchannels() > 1:
            pcm = pcm.reshape(-1, wav.getnchannels())[:, 0].copy()
    return pcm, rate
