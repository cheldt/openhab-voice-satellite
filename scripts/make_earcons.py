#!/usr/bin/env python3
"""Generate the feedback sounds (sounds/*.wav) as short sine sweeps."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

RATE = 16000
SOUNDS = Path(__file__).resolve().parent.parent / "sounds"


def tone(freqs: list[float], duration: float = 0.12, gap: float = 0.02) -> np.ndarray:
    parts = []
    for freq in freqs:
        t = np.linspace(0, duration, int(RATE * duration), endpoint=False)
        envelope = np.hanning(len(t))
        parts.append(np.sin(2 * np.pi * freq * t) * envelope)
        parts.append(np.zeros(int(RATE * gap)))
    pcm = np.concatenate(parts)
    return (pcm * 0.5 * 32767).astype(np.int16)


def write(name: str, pcm: np.ndarray) -> None:
    SOUNDS.mkdir(exist_ok=True)
    path = SOUNDS / name
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(pcm.tobytes())
    print(f"wrote {path}")


if __name__ == "__main__":
    write("wake.wav", tone([660, 880]))          # rising: I'm listening
    write("ack.wav", tone([880]))                # single blip: got it, thinking
    write("error.wav", tone([440, 330], 0.15))   # falling: something failed
    write("idle.wav", tone([880, 660]))          # falling: session over, idle
