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

Both corpora go in one pass (`--positives`) because the cell worth choosing is
the one that silences the false accepts *and* still fires on every wakeword,
and that is a comparison between two tables at the same coordinates — which is
not something anyone reads correctly off two tables printed apart.

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
# 0.95 almost nothing is accepted. The step is 0.05 and not 0.10 because the
# operating points that matter fall between the round numbers — livekit
# publishes 0.08 false accepts per hour for its own pretrained head at a
# threshold of 0.68, and a grid offering only 0.60 or 0.70 cannot show it.
# WakewordConfig allows patience up to 10, but past 4 (320 ms at hop 1) the
# added latency is audible.
TRIAL_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
TRIAL_PATIENCES = (1, 2, 3, 4)

# printed under both grids: a count this replay cannot reproduce exactly
UPPER_BOUND_NOTE = (
    "  (counts above 1 per file are upper bounds: the app resets the detector\n"
    "   after every detection, this replay only re-arms below threshold/2)"
)


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


def _cell(replays: list[Replay], threshold: float, patience: int) -> tuple[int, int]:
    """(detections, files firing at least once) for one corpus at one cell."""
    counts = [would_fire(r.scores, threshold, patience) for r in replays]
    return sum(counts), sum(1 for n in counts if n)


def _header(left: int, right: int) -> str:
    """Patience labels sitting over each cell's left-hand number.

    That is the number a reader scans down a column — detections in one grid,
    false accepts in the other — so the label belongs above it rather than
    above the cell as a whole, which is what put it over the slash.

    `left` and `right` are the cell's own two field widths, passed so the
    header cannot drift from the row it labels: a one-character mismatch
    compounds per cell, and by the fourth column the grid reads off by one.
    """
    return " thr \\ pat" + "".join(
        f"{p:>{left}}{'':<{right + 1}}" for p in TRIAL_PATIENCES
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
    print(_header(6, 4))
    for threshold in TRIAL_THRESHOLDS:
        cells = ""
        for patience in TRIAL_PATIENCES:
            detections, firing = _cell(scored, threshold, patience)
            cells += f"{detections:>6}/{firing:<4}"
        print(f"      {threshold:.2f}{cells}")
    live_threshold = config.wakeword.threshold
    live_patience = config.wakeword.patience
    detections, firing = _cell(scored, live_threshold, live_patience)
    print(
        f"  configured (threshold {live_threshold:.2f}, patience {live_patience}): "
        f"{detections} detection(s), {firing} of {len(scored)} file(s) firing"
    )
    print(UPPER_BOUND_NOTE)


def _print_joined_grid(
    false_replays: list[Replay], positive_replays: list[Replay], config: Config
) -> None:
    """One table reading the false accepts against the recall they cost.

    The two numbers in a cell are deliberately not the same statistic. A
    false-accept corpus is counted by detections, because two firings inside
    one dump are two interruptions; a wakeword corpus is counted by files,
    because a recording that fires twice still only had to be spoken once,
    and counting detections there would let one stuttering recording cover
    for another that went silent.
    """
    false_scored = [r for r in false_replays if r.scores]
    true_scored = [r for r in positive_replays if r.scores]
    total = len(true_scored)
    print(
        f"\nwould-fire per (threshold, patience): false accepts / wakewords "
        f"firing (of {total}), over {len(false_scored)} false and {total} true file(s):"
    )
    print(_header(7, 5))
    # lowest false accepts among the cells that still catch every wakeword.
    # Scanning thresholds and patience ascending with a strict `<` keeps the
    # first such cell, which is the least aggressive one — the tie worth
    # taking, since every unseen wakeword is judged by the same bar.
    best: tuple[int, float, int] | None = None
    for threshold in TRIAL_THRESHOLDS:
        cells = ""
        for patience in TRIAL_PATIENCES:
            accepts, _ = _cell(false_scored, threshold, patience)
            _, caught = _cell(true_scored, threshold, patience)
            cells += f"{accepts:>7}/{caught:<5}"
            if caught == total and (best is None or accepts < best[0]):
                best = (accepts, threshold, patience)
        print(f"      {threshold:.2f}{cells}")

    if best is None:
        print(
            "  no cell catches every wakeword: nowhere in the grid does a bar "
            "hold all\n   of them, so this corpus cannot choose a threshold — "
            "read the traces above\n   for the recordings the model simply "
            "scored low, and fix the corpus or the model"
        )
    else:
        accepts, threshold, patience = best
        print(
            f"  best: threshold {threshold:.2f}, patience {patience} — "
            f"{accepts} false accept(s), all {total} wakeword(s) still firing"
        )
        speaking = config.wakeword.threshold_speaking
        if threshold > speaking:
            print(
                f"  note: wakeword.threshold_speaking is {speaking:.2f}, below "
                "that — raise it\n   too, or echo from our own playback is "
                "judged at a lower bar than the room"
            )
    live_threshold = config.wakeword.threshold
    live_patience = config.wakeword.patience
    accepts, _ = _cell(false_scored, live_threshold, live_patience)
    _, caught = _cell(true_scored, live_threshold, live_patience)
    print(
        f"  configured (threshold {live_threshold:.2f}, patience {live_patience}): "
        f"{accepts} false accept(s), {caught} of {total} wakeword(s) firing"
    )
    print(UPPER_BOUND_NOTE)


def _replay_corpus(
    detector: WakewordProtocol, config: Config, paths: list[Path], label: str = ""
) -> tuple[list[Replay], bool]:
    """Replay and trace every path; the scored replays and whether one failed.

    A bad file is reported and skipped rather than aborting the run — a corpus
    of field dumps will contain one — but it still sets the exit code.
    """
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
            f"\n{label}{path} ({pcm_s:.2f}s, {replay.frames} frames, "
            f"rms {replay.level}, {len(replay.scores)} evaluations)"
        )
        if not replay.scores:
            _print_unscored(replay, config)
            failed = True
            continue
        _print_trace(replay, config)
        replays.append(replay)
    return replays, failed


def score_wav(
    config: Config, paths: list[Path], positives: list[Path] | None = None
) -> int:
    """Replay every path and print the traces plus one grid. Exit code.

    With `positives` — deliberate wakeword recordings — the grid reads the
    false accepts in `paths` against the recall raising the bar would cost.
    Without them the one corpus is graded on its own, which answers "what
    would have stopped this firing" but not "what does that cost".
    """
    detector = build_detector(config)
    replays, failed = _replay_corpus(detector, config, paths)
    if not positives:
        _print_grid(replays, config)
        return 1 if failed or not replays else 0

    # the detector is reset per file either way, so the two corpora do not
    # have to be replayed by separate detectors to stay independent
    true_replays, true_failed = _replay_corpus(
        detector, config, positives, "wakeword: "
    )
    failed = failed or true_failed
    if replays and true_replays:
        _print_joined_grid(replays, true_replays, config)
    else:
        # one side scored nothing; a joined grid would read as a clean sweep
        # at every threshold rather than as the missing corpus it is
        print("\nonly one corpus scored; grading it alone")
        _print_grid(replays or true_replays, config)
    return 1 if failed or not replays or not true_replays else 0
