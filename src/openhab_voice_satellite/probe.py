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

import numpy as np

from .config import Config, resolve_path
from .wakeword import WakewordDetector

RUN_S = 30
DUMP_WAV = Path("diagnose_capture.wav")


async def _probe(config: Config, base_dir: Path) -> None:
    from .audio.gst_devices import list_audio_nodes, match_node
    from .audio.io import audio_io

    nodes = list_audio_nodes()
    print(f"config audio.input_device = {config.audio.input_device!r}")
    print("Audio/Source nodes visible to the app:")
    for n in nodes:
        if n.media_class.startswith("Audio/Source"):
            print(f"  - {n.name}  ({n.description})")
    resolved = match_node(nodes, config.audio.input_device, "input")
    print(f"resolved capture target = {resolved!r}  (None = PipeWire default source)\n")

    detector = WakewordDetector(config.wakeword)
    async with audio_io(config.audio) as (source, sink):
        await _probe_streams(config, base_dir, detector, source, sink)


async def _probe_streams(config, base_dir, detector, source, sink) -> None:
    from .audio.io import verify_links
    from .audio.wav import read_wav_mono

    earcon_path = resolve_path(config.earcons.wake, base_dir)
    if earcon_path.exists():
        earcon, earcon_rate = read_wav_mono(earcon_path)
    else:
        print(f"(wake earcon missing at {earcon_path} — skipping playback checks)")
        earcon, earcon_rate = np.zeros(0, dtype=np.int16), 16000

    async def play_earcon(tag: str) -> None:
        if not len(earcon):
            return
        print(f"       >>> playing wake earcon through "
              f"{sink.target or 'default sink'} ({tag}) — audible?")
        try:
            await sink.play(earcon, earcon_rate)
            print(f"       >>> earcon play() returned normally ({tag})")
        except Exception as exc:  # noqa: BLE001 - a probe reports, never crashes
            print(f"       >>> earcon play() FAILED ({tag}): {exc}")

    print(f"listening for {RUN_S}s — say the wakeword ({config.wakeword.model}) a few times")
    print(f"{'sec':>4} {'rms':>7} {'peak':>7} {'wake_score':>10}")
    captured: list[np.ndarray] = []
    side_tasks: list[asyncio.Task] = [
        asyncio.create_task(verify_links(source.target, sink.target))
    ]
    peak_score = 0.0
    t = 0

    async def run() -> None:
        nonlocal peak_score, t
        sec_buf: list[np.ndarray] = []
        sec_samples = 0
        best = 0.0
        async for frame in source.frames():
            if t in (8, 18) and len(side_tasks) < (2 if t == 8 else 3):
                side_tasks.append(asyncio.create_task(play_earcon(f"t={t}")))
            captured.append(frame)
            sec_buf.append(frame)
            sec_samples += len(frame)
            detector.process(frame)
            best = max(best, detector.score("wake"))
            if sec_samples >= config.audio.sample_rate:
                pcm = np.concatenate(sec_buf)
                rms = int(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
                mark = "  <-- WAKE" if best >= config.wakeword.threshold else ""
                print(f"{t:>4} {rms:>7} {int(np.abs(pcm).max()):>7} {best:>10.3f}{mark}")
                peak_score = max(peak_score, best)
                sec_buf.clear()
                sec_samples = 0
                best = 0.0
                t += 1
                if t >= RUN_S:
                    break

    try:
        await asyncio.wait_for(run(), timeout=RUN_S + 10)
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


def probe_mic(config: Config, base_dir: Path) -> int:
    asyncio.run(_probe(config, base_dir))
    return 0
