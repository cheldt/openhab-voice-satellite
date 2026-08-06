#!/usr/bin/env python3
"""Record a node's monitor and print the duration of each sound burst.

Bisects where earcon clipping happens. On the box, WITH THE APP RUNNING:

    .venv/bin/python analyze_dings.py echo-cancel-sink.monitor 40

Then trigger the wakeword twice (a few seconds apart) within the recording
window. The script prints one line per detected burst. The wake earcon is
0.280 s long:

  - first burst ~0.28 s like the rest -> our stream delivers it complete;
    the clipping happens downstream (AEC playback -> USB device leg)
  - first burst shorter than the rest -> the clip is already in our
    playback stream / pulse layer
"""

from __future__ import annotations

import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

RATE = 16000
OUT = Path("/tmp/dings.wav")


def main() -> None:
    device = sys.argv[1] if len(sys.argv) > 1 else "echo-cancel-sink.monitor"
    seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    print(f"recording {device} for {seconds}s — trigger the wakeword twice now")
    subprocess.run(
        ["timeout", str(seconds), "parec", f"--device={device}",
         f"--rate={RATE}", "--channels=1", "--format=s16le",
         "--file-format=wav", str(OUT)],
        check=False,
    )
    with wave.open(str(OUT)) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    loud = np.abs(pcm) > 1000
    # close small gaps (<150 ms) so one earcon counts as one burst
    d = np.diff(np.concatenate([[0], loud.astype(int), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    merged: list[list[int]] = []
    for s, e in zip(starts, ends):
        if merged and s - merged[-1][1] < RATE * 0.15:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    if not merged:
        print("no bursts found — was anything played?")
        return
    for i, (s, e) in enumerate(merged):
        rms = int(np.sqrt(np.mean(pcm[s:e].astype(np.float64) ** 2)))
        print(f"burst {i}: at {s / RATE:7.2f}s  duration {(e - s) / RATE:.3f}s  rms={rms}")


main()
