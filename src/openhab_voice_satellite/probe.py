"""Field diagnostic: show what the capture stream really hears (--probe-mic).

Opens the same PipewireSource/PipewireSink the app uses (same config
resolution) and prints, once per second: RMS amplitude, peak, and the
openWakeWord score. Plays the wake earcon through the configured output at
t=8s and t=18s, exactly like the app does after a detection.

Say the wakeword three times: once early (t 2-6), once right after the
first earcon (t 10-14), once after the second (t 20-26). Note whether the
earcons are audible from the speaker. Interpretation:

  - rms ~0 the whole time      -> capture is linked to a silent node
  - rms high even when quiet   -> capture grabbed the wrong/noisy node
                                  (silent default-node fallback)
  - rms tracks your voice but score stays ~0 -> audio corrupted en route
                                  (e.g. graph/device clock-rate mismatch)
  - wakeword fires before the earcon but not after -> playback poisons
    capture (AEC far-end interaction)
  - earcon not audible         -> output path broken or speaker asleep

The captured audio is written to diagnose_capture.wav — play it back to
hear exactly what the app hears.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable

import numpy as np

from .audio.wav import rms
from .config import Config
from .wakeword import WakewordDetector

RUN_S = 30
EARCON_AT_S = (8, 18)  # seconds at which the wake earcon is played
DUMP_WAV = Path("diagnose_capture.wav")


def _print_sources(config: Config) -> None:
    from .audio.gst_devices import list_audio_nodes, match_node

    nodes = list_audio_nodes()
    print(f"config audio.input_device = {config.audio.input_device!r}")
    print("Audio/Source nodes visible to the app:")
    for n in nodes:
        if n.media_class.startswith("Audio/Source"):
            print(f"  - {n.name}  ({n.description})")
    resolved = match_node(nodes, config.audio.input_device, "input")
    print(f"resolved capture target = {resolved!r}  (None = PipeWire default source)\n")


def _load_earcon(config: Config) -> tuple[np.ndarray, int]:
    from .audio.wav import read_wav_mono

    earcon_path = Path(config.earcons.wake)
    if earcon_path.exists():
        return read_wav_mono(earcon_path)
    print(f"(wake earcon missing at {earcon_path} — skipping playback checks)")
    return np.zeros(0, dtype=np.int16), 16000


async def _play_earcon(sink, earcon: np.ndarray, rate: int, tag: str) -> None:
    if not len(earcon):
        return
    print(f"       >>> playing wake earcon through "
          f"{sink.target or 'default sink'} ({tag}) — audible?")
    try:
        await sink.play(earcon, rate)
        print(f"       >>> earcon play() returned normally ({tag})")
    except Exception as exc:  # noqa: BLE001 - a probe reports, never crashes
        print(f"       >>> earcon play() FAILED ({tag}): {exc}")


class _SecondStats:
    """Aggregates frames into once-per-second RMS/peak/score lines."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._buf: list[np.ndarray] = []
        self._samples = 0
        self._best = 0.0
        self.t = 0
        self.peak_score = 0.0

    def add(self, frame: np.ndarray, score: float) -> bool:
        """Accumulate one frame; True when a full second was printed."""
        self._buf.append(frame)
        self._samples += len(frame)
        self._best = max(self._best, score)
        if self._samples < self._config.audio.sample_rate:
            return False
        pcm = np.concatenate(self._buf)
        mark = "  <-- WAKE" if self._best >= self._config.wakeword.threshold else ""
        print(f"{self.t:>4} {rms(pcm):>7} {int(np.abs(pcm).max()):>7} {self._best:>10.3f}{mark}")
        self.peak_score = max(self.peak_score, self._best)
        self._buf.clear()
        self._samples = 0
        self._best = 0.0
        self.t += 1
        return True


async def _capture_loop(
    config: Config, source, detector: WakewordDetector,
    captured: list[np.ndarray], on_second: "Callable[[int], None]",
) -> float:
    """Print per-second stats until RUN_S; returns the peak wake score."""
    stats = _SecondStats(config)
    async for frame in source.frames():
        captured.append(frame)
        detector.process(frame)
        if stats.add(frame, detector.score("wake")):
            on_second(stats.t)
            if stats.t >= RUN_S:
                break
    return stats.peak_score


async def _probe(config: Config) -> None:
    from .audio.io import audio_io, verify_links

    _print_sources(config)
    detector = WakewordDetector(config.wakeword)
    earcon, earcon_rate = _load_earcon(config)

    async with audio_io(config.audio) as (source, sink):
        print(f"listening for {RUN_S}s — say the wakeword "
              f"({config.wakeword.model}) a few times")
        print(f"{'sec':>4} {'rms':>7} {'peak':>7} {'wake_score':>10}")
        captured: list[np.ndarray] = []
        side_tasks = [asyncio.create_task(verify_links(source.target, sink.target))]

        def on_second(t: int) -> None:
            if t in EARCON_AT_S:
                side_tasks.append(asyncio.create_task(
                    _play_earcon(sink, earcon, earcon_rate, f"t={t}")
                ))

        peak_score = 0.0
        try:
            peak_score = await asyncio.wait_for(
                _capture_loop(config, source, detector, captured, on_second),
                timeout=RUN_S + 10,
            )
        except asyncio.TimeoutError:
            print("TIMED OUT waiting for audio frames — capture stream is stalled/unlinked")
        finally:
            for task in side_tasks:
                task.cancel()

    if captured:
        from .audio.wav import write_wav

        write_wav(DUMP_WAV, np.concatenate(captured), config.audio.sample_rate)
        print(f"\npeak wake score: {peak_score:.3f} (threshold {config.wakeword.threshold})")
        print(f"captured audio written to {DUMP_WAV} — play it back to hear what the app hears")


def probe_mic(config: Config) -> int:
    asyncio.run(_probe(config))
    return 0
