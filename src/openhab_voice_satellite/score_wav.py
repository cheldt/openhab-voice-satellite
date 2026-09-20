"""`--score-wav`: replay recorded audio through the detector and grade the knobs.

The field question after a false accept is never "what did it score" but
"what would have kept this from firing without also losing the real
wakeword". Answering that on the satellite costs a config edit, a restart,
and then waiting for the next false positive — which is not reproducible on
demand. So a directory of $OVS_DUMP_WAKE dumps (the audio behind each
detection) and a directory of deliberate wakeword recordings are replayed
here through the same `build_detector(config)` the app runs, and every
(threshold, patience) pair is scored against both in one pass.

The would-fire rule is `EdgeTrigger` itself rather than a copy of it. The
hysteresis and the patience window are precisely what is being tuned, so a
reimplementation would grade the tuning against a rule the app does not run —
and that is the reason the decision lives apart from the engines at all.

Two limits the output states rather than hides: the app calls
`detector.reset()` after each detection (on livekit a 2 s mute) while this
replay only re-arms through the normal hysteresis, so counts above one per
file are upper bounds; and `threshold_speaking` is not graded, because
whether our own output was audible on a given frame is not something a WAV
carries. Every dump worth grading is an idle false accept anyway.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .audio.wav import read_wav_mono, rms
from .config import Config
from .wakeword import WAKE, EdgeTrigger, WakewordProtocol, build_detector

log = logging.getLogger(__name__)

# the range worth tabulating: below 0.5 almost nothing is rejected, above
# 0.95 almost nothing is accepted. WakewordConfig allows patience up to 10,
# but past 4 (320 ms at hop 1) the added latency is audible.
TRIAL_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)
TRIAL_PATIENCES = (1, 2, 3, 4)


@dataclass
class Replay:
    """One file's scored evaluations, in order."""

    path: Path
    frames: int
    # int16 RMS of the whole file: openwakeword does no input normalisation
    # and neither does livekit's frontend, so a file that reads far below the
    # ~1000-5000 of ordinary speech scores low for a reason no knob fixes
    level: int
    scores: list[float]
    # seconds into the file at which each evaluation landed
    times: list[float]

    @property
    def peak(self) -> float:
        return max(self.scores) if self.scores else 0.0


def would_fire(scores: list[float], threshold: float, patience: int) -> int:
    """Detections this evaluation sequence yields at (threshold, patience).

    Evaluations only: the live detector advances its trigger on scored frames
    and on nothing else, so padding the skipped frames back in here would
    change the answer for every patience above 1.
    """
    trigger = EdgeTrigger()
    return sum(trigger.feed(score, threshold, patience) for score in scores)


def _evaluation_interval_s(config: Config) -> float:
    """Wall time between two scored frames, for the trace's t column."""
    hop = config.wakeword.livekit.hop_frames if config.wakeword.engine == "livekit" else 1
    return config.audio.frame_ms * hop / 1000


def replay_file(detector: WakewordProtocol, config: Config, path: Path) -> Replay:
    """Push one WAV through `detector` at `audio.frame_ms`, from cold.

    The detector is reset first so file N starts as cold as file 1 — on a
    window-scoring engine that also re-primes it, which is the difference
    between grading a file and grading it plus the tail of the previous one.

    `scored_last_frame` is what selects the evaluations rather than watching
    for a changed score: a skipped frame leaves the previous score in place,
    so two equal evaluations and one skip are indistinguishable by value.
    """
    pcm, rate = read_wav_mono(path)
    if rate != config.audio.sample_rate:
        raise ValueError(
            f"{rate} Hz, but the detector runs at {config.audio.sample_rate} Hz — "
            "resample it offline; resampling here would change the very scores "
            "being graded"
        )
    detector.reset()
    frame_samples = config.audio.frame_samples
    interval_s = _evaluation_interval_s(config)
    scores: list[float] = []
    times: list[float] = []
    frames = 0
    # a trailing partial frame is dropped: an engine fed a short frame
    # mis-aligns its embedding grid for everything after it
    for start in range(0, len(pcm) - frame_samples + 1, frame_samples):
        detector.process(pcm[start:start + frame_samples])
        frames += 1
        if getattr(detector, "scored_last_frame", True):
            scores.append(detector.score(WAKE))
            times.append(start / config.audio.sample_rate + interval_s)
    return Replay(
        path=path, frames=frames, level=rms(pcm), scores=scores, times=times
    )


def _print_trace(replay: Replay, config: Config) -> None:
    print(f"{'t':>7} {'score':>8}")
    for at, score in zip(replay.times, replay.scores):
        mark = "  *" if score >= config.wakeword.threshold else ""
        print(f"{at:>7.2f} {score:>8.3f}{mark}")
    print(f"  peak {replay.peak:.3f} over {len(replay.scores)} evaluations")


def _print_unscored(replay: Replay, config: Config) -> None:
    """Say why a file produced nothing, without hardcoding an engine's number."""
    duration_s = replay.frames * config.audio.frame_ms / 1000
    print(
        f"  no evaluation: {duration_s:.2f}s ({replay.frames} frames) — engine "
        f"{config.wakeword.engine!r} scored none of them. A window-scoring engine "
        "stays muted until its context window fills (livekit: 2s), so a clip has "
        "to carry that much audio *before* the wakeword. Re-cut it from a longer "
        "recording, or collect dumps with OVS_DUMP_WAKE, which keeps 5s."
    )


def _print_grid(replays: list[Replay], config: Config) -> None:
    scored = [r for r in replays if r.scores]
    if not scored:
        return
    # detections/files rather than detections alone: over a positive corpus
    # the number that matters is how many recordings still fire at all
    # (recall), and a single file firing three times hides two that went
    # silent. Over a false-positive corpus the total is the one to read.
    print(
        f"\nwould-fire detections/files per (threshold, patience), "
        f"{len(scored)} file(s):"
    )
    print("  thr \\ pat" + "".join(f"{p:>9}" for p in TRIAL_PATIENCES))
    for threshold in TRIAL_THRESHOLDS:
        cells = ""
        for patience in TRIAL_PATIENCES:
            counts = [would_fire(r.scores, threshold, patience) for r in scored]
            cells += f"{sum(counts):>6}/{sum(1 for n in counts if n):<3}"
        print(f"      {threshold:.2f}{cells}")
    live_threshold = config.wakeword.threshold
    live_patience = config.wakeword.patience
    per_file = [would_fire(r.scores, live_threshold, live_patience) for r in scored]
    print(
        f"  configured (threshold {live_threshold:.2f}, patience {live_patience}): "
        f"{sum(per_file)} detection(s), {sum(1 for n in per_file if n)} of "
        f"{len(scored)} file(s) firing"
    )
    print(
        "  (counts above 1 per file are upper bounds: the app resets the detector\n"
        "   after every detection, this replay only re-arms below threshold/2)"
    )


def score_wav(config: Config, paths: list[Path]) -> int:
    """Replay every path and print the traces plus one shared grid. Exit code.

    A bad file is reported and skipped rather than aborting the run — a corpus
    of field dumps will contain one — but it still sets the exit code.
    """
    detector = build_detector(config)
    replays: list[Replay] = []
    failed = False
    for path in paths:
        try:
            replay = replay_file(detector, config, path)
        except (OSError, ValueError) as exc:
            print(f"{path}: {exc}")
            failed = True
            continue
        pcm_s = replay.frames * config.audio.frame_ms / 1000
        print(
            f"\n{path} ({pcm_s:.2f}s, {replay.frames} frames, rms {replay.level}, "
            f"{len(replay.scores)} evaluations)"
        )
        if not replay.scores:
            _print_unscored(replay, config)
            failed = True
            continue
        _print_trace(replay, config)
        replays.append(replay)
    _print_grid(replays, config)
    return 1 if failed or not replays else 0
