"""Field diagnostic: show what the capture stream really hears (--probe-mic).

Opens the same PipewireSource/PipewireSink the app uses (same config
resolution) and prints, once per second: RMS amplitude, peak, and the stage-1
wake score. The WAKE mark comes from `process()`'s verdict — patience, the
state-dependent threshold and the stage-2 verifier included — so a marked row
is what the running app would have acted on, not merely a high stage-1 score.
Plays the wake earcon through the configured output at t=8s and t=18s,
exactly like the app does after a detection.

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
  - score high but rows say "rejected by stage 2" -> stage 1 hears you and
    the verifier does not; re-record and evaluate with --score-wav
  - earcon not audible         -> output path broken or speaker asleep

The last two columns are what the engine costs on this machine: wall and
process CPU milliseconds per scored frame, against a budget of audio.frame_ms.

  - ms/frame near the frame budget -> this engine cannot keep up here
  - cpu_ms far above ms/frame      -> a native thread pool is spinning
                                      (process CPU counts every thread)

The captured audio is written to diagnose_capture.wav — play it back to
hear exactly what the app hears.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .audio.wav import rms
from .config import Config
from .wakeword import STOP, WAKE, WakewordProtocol, build_detector

RUN_S = 30
STALL_GRACE_S = 10  # extra wall time before a frameless run counts as stalled
EARCON_AT_S = (8, 18)  # seconds at which the wake earcon is played
DUMP_WAV = Path("diagnose_capture.wav")


def _print_sources(config: Config) -> bool:
    """List the visible capture nodes; False when PipeWire is unreachable."""
    from .audio.gst_devices import list_audio_nodes, match_node

    try:
        nodes = list_audio_nodes()
    except Exception as exc:  # noqa: BLE001 - a probe reports, never crashes
        print(f"cannot list PipeWire nodes: {exc}")
        print("(requires PipeWire running and the GStreamer pipewire plugin, "
              "see deploy/install.md)")
        return False
    print(f"config audio.input_device = {config.audio.input_device!r}")
    print("Audio/Source nodes visible to the app:")
    for n in nodes:
        if n.media_class.startswith("Audio/Source"):
            print(f"  - {n.name}  ({n.description})")
    resolved = match_node(nodes, config.audio.input_device, "input")
    print(f"resolved capture target = {resolved!r}  (None = PipeWire default source)\n")
    return True


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


class _Cost:
    """Mean per-frame cost of scoring, in milliseconds.

    `process_time` is process-wide, so a spinning ORT pool shows up here even
    though it never runs on this thread — which is the whole point of printing
    it next to the wall clock.
    """

    def __init__(self) -> None:
        self._wall = 0.0
        self._cpu = 0.0
        self._frames = 0

    def add(self, wall_s: float, cpu_s: float) -> None:
        self._wall += wall_s
        self._cpu += cpu_s
        self._frames += 1

    def _mean_ms(self, total: float) -> float:
        return total / self._frames * 1000 if self._frames else 0.0

    @property
    def wall_ms(self) -> float:
        return self._mean_ms(self._wall)

    @property
    def cpu_ms(self) -> float:
        return self._mean_ms(self._cpu)


class _SecondStats:
    """Aggregates frames into once-per-second RMS/peak/score lines.

    The score columns are stage 1; the WAKE/STOP marks are `process()`'s
    verdicts. On a two-stage config those are different instruments — stage 1
    runs deliberately low and the verifier swallows most of its triggers, so
    a score-derived mark would fire on almost any speech the app rejects.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._has_stop = bool(config.wakeword.stop_model)
        self._two_stage = bool(config.wakeword.stage2.model)
        self._buf: list[np.ndarray] = []
        self._samples = 0
        self._frames = 0
        self._row_start = time.monotonic()
        self._best = 0.0
        self._best_stop = 0.0
        self._verdicts: set[str] = set()
        self._verifier_score: float | None = None  # last verdict this second
        self._rejected = False
        self._audible = False
        self._cost = _Cost()
        self._run_frames = 0
        self._run_start = self._row_start
        self.t = 0
        self.peak_score = 0.0
        self.peak_stop_score = 0.0
        self.wake_events = 0
        self.stop_events = 0
        self.stage2_rejections = 0
        self.cost = _Cost()  # the whole run, not just the current second

    @property
    def fps(self) -> float:
        """Frames actually delivered per second, over the whole run."""
        elapsed = time.monotonic() - self._run_start
        return self._run_frames / elapsed if elapsed else 0.0

    def header(self) -> str:
        cols = [f"{'sec':>4}", f"{'rms':>7}", f"{'peak':>7}", f"{'wake_score':>10}"]
        if self._has_stop:
            cols.append(f"{'stop_score':>10}")
        if self._two_stage:
            cols.append(f"{'verifier':>8}")
        cols += [f"{'fps':>7}", f"{'ms/frame':>8}", f"{'cpu_ms':>7}"]
        return " ".join(cols)

    def add(
        self, frame: np.ndarray, detector: WakewordProtocol, verdict: str | None,
        audible: bool, wall_s: float, cpu_s: float,
    ) -> bool:
        """Accumulate one frame; True when a full second was printed."""
        self._buf.append(frame)
        self._samples += len(frame)
        self._frames += 1
        self._run_frames += 1
        self._best = max(self._best, detector.score(WAKE))
        if self._has_stop:
            self._best_stop = max(self._best_stop, detector.score(STOP))
        if verdict is not None:
            self._verdicts.add(verdict)
            self.wake_events += verdict == WAKE
            self.stop_events += verdict == STOP
        # last_verifier_score persists past its verdict frame, so only read
        # it on the frame a verdict lands: an accept (WAKE) or a rejection
        if verdict == WAKE or detector.last_rejection is not None:
            self._verifier_score = detector.last_verifier_score
        if detector.last_rejection is not None:
            self._rejected = True
            self.stage2_rejections += 1
        self._audible = self._audible or audible
        self._cost.add(wall_s, cpu_s)
        self.cost.add(wall_s, cpu_s)
        # a row per elapsed second, not per second of *audio*: pacing on the
        # sample count would stretch the run under-delivery instead of showing
        # it, which is exactly the failure this probe exists to catch
        elapsed = time.monotonic() - self._row_start
        if elapsed < 1.0:
            return False
        self._print_row(self._frames / elapsed)
        self.peak_score = max(self.peak_score, self._best)
        self.peak_stop_score = max(self.peak_stop_score, self._best_stop)
        self._buf.clear()
        self._samples = 0
        self._frames = 0
        self._row_start = time.monotonic()
        self._best = 0.0
        self._best_stop = 0.0
        self._verdicts = set()
        self._verifier_score = None
        self._rejected = False
        self._audible = False
        self._cost = _Cost()
        self.t += 1
        return True

    def _print_row(self, fps: float) -> None:
        pcm = np.concatenate(self._buf)
        # marked from the verdicts, not the score: process() already applied
        # patience, the state-dependent threshold (speaking=True while our own
        # output is audible) and the stage-2 verifier
        if STOP in self._verdicts:
            mark = "  <-- STOP"
        elif WAKE in self._verdicts:
            mark = "  <-- WAKE"
        elif self._rejected:
            mark = "  <-- rejected by stage 2"
        else:
            mark = ""
        if self._audible:
            mark += " (playback)"
        cells = [f"{self.t:>4}", f"{rms(pcm):>7}", f"{int(np.abs(pcm).max()):>7}",
                 f"{self._best:>10.3f}"]
        if self._has_stop:
            cells.append(f"{self._best_stop:>10.3f}")
        if self._two_stage:
            cells.append(f"{self._verifier_score:>8.3f}"
                         if self._verifier_score is not None else f"{'-':>8}")
        cells += [f"{fps:>7.1f}", f"{self._cost.wall_ms:>8.1f}",
                  f"{self._cost.cpu_ms:>7.1f}"]
        print(" ".join(cells) + mark)


async def _capture_loop(
    source, sink, detector: WakewordProtocol, stats: _SecondStats,
    captured: list[np.ndarray], on_second: "Callable[[int], None]",
) -> None:
    """Print per-second stats into `stats` until RUN_S seconds have elapsed."""
    async for frame in source.frames():
        captured.append(frame)
        audible = sink.is_playing
        wall = time.perf_counter()
        cpu = time.process_time()
        verdict = detector.process(frame, speaking=audible)
        wall = time.perf_counter() - wall
        cpu = time.process_time() - cpu
        if stats.add(frame, detector, verdict, audible, wall, cpu):
            on_second(stats.t)
            if stats.t >= RUN_S:
                break


async def _probe(config: Config) -> int:
    from .audio.io import audio_io, verify_links

    if not _print_sources(config):
        return 1
    detector = build_detector(config)
    earcon, earcon_rate = _load_earcon(config)
    timed_out = False

    async with audio_io(config.audio) as (source, sink):
        print(f"listening for {RUN_S}s — say the wakeword "
              f"({config.wakeword.model}) a few times")
        stats = _SecondStats(config)
        print(stats.header())
        captured: list[np.ndarray] = []
        side_tasks = [asyncio.create_task(verify_links(source.target, sink.target))]

        def on_second(t: int) -> None:
            if t in EARCON_AT_S:
                side_tasks.append(asyncio.create_task(
                    _play_earcon(sink, earcon, earcon_rate, f"t={t}")
                ))

        try:
            await asyncio.wait_for(
                _capture_loop(source, sink, detector, stats, captured, on_second),
                timeout=RUN_S + STALL_GRACE_S,
            )
        except asyncio.TimeoutError:
            timed_out = True
            print("TIMED OUT waiting for audio frames — capture stream is stalled/unlinked")
        finally:
            capture = source.stats()  # read before audio_io closes the source
            for task in side_tasks:
                task.cancel()

    if captured:
        from .audio.wav import write_wav

        wakeword = config.wakeword
        write_wav(DUMP_WAV, np.concatenate(captured), config.audio.sample_rate)
        print(f"\npeak wake score: {stats.peak_score:.3f} "
              f"(threshold {wakeword.threshold}, {wakeword.threshold_speaking} during playback)")
        if wakeword.stop_model:
            print(f"peak stop score: {stats.peak_stop_score:.3f} "
                  f"(threshold {wakeword.stop_threshold}, "
                  f"{wakeword.effective_stop_threshold_speaking} during playback)")
        verdicts = f"wake events: {stats.wake_events}"
        if wakeword.stop_model:
            verdicts += f", stop events: {stats.stop_events}"
        if wakeword.stage2.model:
            verdicts += f", rejected by stage 2: {stats.stage2_rejections}"
        print(f"{verdicts}  (process() verdicts — what the app would act on)")
        expected_fps = config.audio.sample_rate / config.audio.frame_samples
        short = "  <-- capture is under-delivering" if stats.fps < 0.8 * expected_fps else ""
        print(f"capture rate: {stats.fps:.1f} of {expected_fps:.1f} frames/s"
              f"{short}")
        print(f"{wakeword.engine} cost: {stats.cost.wall_ms:.1f} ms wall, "
              f"{stats.cost.cpu_ms:.1f} ms process CPU per frame "
              f"(budget {config.audio.frame_ms} ms)")
        print(f"capture accounting — {capture.describe()}")
        print(f"captured audio written to {DUMP_WAV} — play it back to hear what the app hears")
    # scriptable: a stalled/unlinked capture must fail the way --check does
    return 1 if timed_out else 0


def probe_mic(config: Config) -> int:
    return asyncio.run(_probe(config))
