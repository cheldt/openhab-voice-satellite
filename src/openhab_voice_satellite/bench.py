"""Offline wakeword evaluation (--score-wav): score WAVs as the app would.

The running app is a bad place to judge a wakeword model. Detections arrive
minutes apart, the audio that caused them is gone, and the only knobs anyone
can turn are a threshold and a patience count that cannot separate a model
whose false accepts outscore its true ones. This scores recorded audio through
`build_detector` — the same engine, the same frame size, the same decision
rule — so a candidate model can be judged before it reaches a device.

Three modes, increasing in strength:

  --score-wav FILE|DIR...        per-file score distribution and a
                                 threshold x patience detection sweep
  --compare MODEL                a second model over the same frames, so two
                                 candidates are judged on identical audio
  --positives DIR --negatives DIR  recall and false accepts per hour, which is
                                 the only honest gate for promoting a model

Scores come from the engine; the detection counts come from `EdgeTrigger`, the
same object `BaseWakewordDetector` runs, so the sweep cannot drift from what
the app would have done with the same audio.

One thing the sweep structurally cannot see: a second stage. `score()` reports
the stage-1 score, so on a config with a verifier the sweep counts stage-1
triggers, most of which the verifier then swallows — and such a config runs
stage 1 deliberately low *because* the verifier is the real gate. Every
report therefore also carries a `live` count, which is what `process()` itself
returned, and a `--compare` across that asymmetry refuses to name a winner.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .config import SAMPLE_RATE, Config
from .wakeword import WAKE, EdgeTrigger, build_detector

# the sweep axes: thresholds worth considering, and the patience values that
# stay inside the latency budget (each frame is audio.frame_ms of delay).
# The tail past 0.99 is not padding: a sigmoid head saturates, and a model
# whose separation lives between 0.99 and 0.9999 would otherwise be reported
# as having no clean operating point — a grid artifact wearing a verdict.
# The low end matters with a second stage: a verifier lets stage 1 run far
# below any single-stage operating point (v6 two-stage sat at 0.10).
SWEEP_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99,
                    0.995, 0.999, 0.9995, 0.9999)
SWEEP_PATIENCE = (1, 2, 3)

WAV_SUFFIXES = (".wav",)

# Silence appended after every file before the detector is read.
#
# A detector may defer a verdict past the frame that triggered it: with a
# stage-2 verifier configured it holds each WAKE for `stage2.delay_ms` so the
# verifier can score a finished phrase. Live that costs nothing, because the mic
# keeps supplying frames. A file just ends, and the pending verdict dies with
# it: on 1 s wakeword clips that read as 42 % recall for a detector measured at
# 98 %. A quiet room after the phrase is what the flush imitates. It is the
# longest deferral the config permits (Stage2Config.delay_ms caps at 1000).
FLUSH_MS = 1000


class FrameScores(NamedTuple):
    """One WAV's per-frame wake scores, plus what the app would have emitted.

    `scores` is what the sweep replays. `live` counts the WAKEs `process()`
    actually returned — on an engine with a stage-2 verifier that is a
    different, smaller number than any sweep cell, because `score()` reports
    the stage-1 score and never sees the verifier's verdict.
    """

    scores: np.ndarray
    live: int


def _expand(paths: list[Path]) -> list[Path]:
    """Files as given, directories as their sorted *.wav contents."""
    out: list[Path] = []
    for path in paths:
        if path.is_dir():
            out.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in WAV_SUFFIXES))
        else:
            out.append(path)
    return out


def _override(config: Config, engine: str | None, model: str | None) -> Config:
    """A copy of `config` with the wakeword engine/model swapped.

    Re-validated rather than `model_copy`d so the cross-field rules still
    apply — an override that a hand-written config would be rejected for must
    be rejected here too, rather than failing silently at runtime.
    """
    if engine is None and model is None:
        return config
    data = config.model_dump()
    if engine is not None:
        data["wakeword"]["engine"] = engine
    if model is not None:
        data["wakeword"]["model"] = model
    return Config.model_validate(data)


def score_file(config: Config, path: Path) -> FrameScores:
    """Per-frame wake scores for one WAV, at the configured frame size."""
    from .audio.wav import read_wav_mono

    pcm, rate = read_wav_mono(path)
    if rate != SAMPLE_RATE:
        raise ValueError(
            f"{path}: {rate} Hz, but the detector only runs at {SAMPLE_RATE} Hz — "
            f"resample first (resampling here would change what is measured)"
        )
    detector = build_detector(config)
    n = config.audio.frame_samples
    scores = []
    live = 0
    for i in range(0, len(pcm) - n + 1, n):
        # the return value and the score are different measurements: the score
        # is stage 1, the return value is the whole detector
        if detector.process(pcm[i:i + n]) == WAKE:
            live += 1
        scores.append(detector.score(WAKE))
    # the flush feeds `live` only: these frames are not audio the corpus
    # contained, so they must not reach the sweep, the percentiles, or the
    # frame count the false-accepts-per-hour figure divides by
    silence = np.zeros(n, dtype=pcm.dtype)
    for _ in range(FLUSH_MS // config.audio.frame_ms):
        if detector.process(silence) == WAKE:
            live += 1
    return FrameScores(np.asarray(scores, dtype=np.float64), live)


def count_detections(scores: np.ndarray, threshold: float, patience: int) -> int:
    """How many events the app's decision rule would emit for this score run."""
    trigger = EdgeTrigger()
    return sum(bool(trigger.feed(s, threshold, patience)) for s in scores)


def _label(config: Config) -> str:
    wakeword = config.wakeword
    return f"{wakeword.engine}:{Path(wakeword.model).name}"


def _is_two_stage(config: Config) -> bool:
    """Whether this config runs a second stage the sweep cannot model.

    `--compare` carries the second stage onto both columns on purpose, which
    is what makes a two-stage comparison possible at all.
    """
    return bool(config.wakeword.stage2.model)


# -- reporting ----------------------------------------------------------


def _report_files(
    configs: list[Config], paths: list[Path], frame_ms: int
) -> tuple[dict[Path, list[FrameScores]], set[Path]]:
    """Per-file score distributions, one column group per config.

    Returns the scores plus the files that could not be read at all. A wrong
    sample rate or a corrupt container is a property of the file, not of the
    config, so one failure is recorded on the first column and skipped on the
    rest — otherwise `--compare`'s per-config columns would shift against each
    other. Reporting and skipping beats propagating: a single 44.1 kHz phone
    recording in a corpus of fifty used to abort the whole run mid-table with
    a traceback, losing the sweep, the gate and the live lines for everything.
    """
    scored: dict[Path, list[FrameScores]] = {}
    failed: set[Path] = set()
    for config in configs:
        print(f"\n=== {_label(config)} ===")
        print(f"{'file':<34} {'dur':>6} {'frames':>7} "
              f"{'max':>7} {'p50':>7} {'p90':>7} {'p99':>7} {'fires':>6} {'live':>6}")
        for path in paths:
            if path in failed:
                continue
            try:
                result = score_file(config, path)
            except (ValueError, wave.Error) as exc:
                failed.add(path)
                scored.pop(path, None)
                print(f"{path.name[:34]:<34} SKIPPED: {exc}")
                continue
            scored.setdefault(path, []).append(result)
            scores = result.scores
            if not len(scores):
                print(f"{path.name[:34]:<34} {'(shorter than one frame)':>50}")
                continue
            fires = count_detections(
                scores, config.wakeword.threshold, config.wakeword.patience
            )
            print(
                f"{path.name[:34]:<34} {len(scores) * frame_ms / 1000:>5.1f}s "
                f"{len(scores):>7} {np.max(scores):>7.3f} "
                f"{np.percentile(scores, 50):>7.3f} "
                f"{np.percentile(scores, 90):>7.3f} "
                f"{np.percentile(scores, 99):>7.3f} {fires:>6} {result.live:>6}"
            )
        wakeword = config.wakeword
        print(f"  (fires = sweep detections at threshold {wakeword.threshold}, "
              f"patience {wakeword.patience}; live = what process() returned)")
        if _is_two_stage(config):
            print("  (stage-2 verifier configured: fires counts stage-1 triggers, "
                  "live counts what survived the verifier)")
    return scored, failed


def _report_sweep(config: Config, paths: list[Path],
                  scored: dict[Path, list[FrameScores]], column: int) -> None:
    """Total detections across all files for each threshold x patience cell."""
    print(f"\n=== detection sweep — {_label(config)} (total over {len(paths)} file(s)) ===")
    print(f"{'threshold':>9} " + "".join(f"{'pat=' + str(p):>8}" for p in SWEEP_PATIENCE))
    for threshold in SWEEP_THRESHOLDS:
        cells = [
            sum(count_detections(scored[p][column].scores, threshold, patience)
                for p in paths)
            for patience in SWEEP_PATIENCE
        ]
        print(f"{threshold:>9.4g} " + "".join(f"{c:>8}" for c in cells))


def _report_separation(
    positives: list[Path], negatives: list[Path],
    scored: dict[Path, list[FrameScores]], column: int,
) -> None:
    """The weakest positive against the strongest negative.

    A saturated sigmoid puts both at 0.99-something, where the sweep grid
    reads as a wall of identical cells; these two numbers say whether any
    threshold could separate the corpora at all.
    """
    peaks = {
        name: [
            np.max(scored[p][column].scores)
            for p in group
            if len(scored[p][column].scores)
        ]
        for name, group in (("pos", positives), ("neg", negatives))
    }
    if not peaks["pos"] or not peaks["neg"]:
        return
    weakest, strongest = min(peaks["pos"]), max(peaks["neg"])
    margin = "separable" if weakest > strongest else "OVERLAPPING"
    print(f"  weakest positive {weakest:.6g}, strongest negative {strongest:.6g}"
          f"  [{margin}]")


def _report_live(
    config: Config, positives: list[Path], negatives: list[Path],
    scored: dict[Path, list[FrameScores]], column: int, negative_hours: float,
) -> None:
    """What the app itself would have emitted, verifier and all.

    The sweep above replays `EdgeTrigger` over stage-1 scores, which is the
    whole detector on a single-stage config and only half of it once a
    verifier is configured. This line is the other measurement.
    """
    hits = sum(scored[p][column].live > 0 for p in positives)
    recall = hits / len(positives) if positives else 0.0
    false = sum(scored[p][column].live for p in negatives)
    per_hour = false / negative_hours if negative_hours else float("nan")
    stage = "two-stage" if _is_two_stage(config) else "single-stage"
    print(f"  live ({stage}, at the configured threshold "
          f"{config.wakeword.threshold} / patience {config.wakeword.patience}): "
          f"recall {recall:.0%}, {per_hour:.1f} false/h")


def _report_gate(
    configs: list[Config],
    positives: list[Path],
    negatives: list[Path],
    scored: dict[Path, list[FrameScores]],
    frame_ms: int,
) -> None:
    """Recall vs false accepts per hour — the promotion gate for a model."""
    negative_hours = sum(
        len(scored[p][0].scores) * frame_ms / 1000 for p in negatives
    ) / 3600 or float("nan")
    for column, config in enumerate(configs):
        print(f"\n=== gate — {_label(config)} "
              f"({len(positives)} positive(s), {negative_hours * 3600:.0f}s of negatives) ===")
        print(f"{'threshold':>9} {'patience':>9} {'recall':>8} {'false/h':>9}")
        best = None
        for threshold in SWEEP_THRESHOLDS:
            for patience in SWEEP_PATIENCE:
                hits = sum(
                    count_detections(scored[p][column].scores, threshold, patience) > 0
                    for p in positives
                )
                recall = hits / len(positives) if positives else 0.0
                false = sum(
                    count_detections(scored[p][column].scores, threshold, patience)
                    for p in negatives
                )
                per_hour = false / negative_hours if negative_hours else float("nan")
                print(f"{threshold:>9.4g} {patience:>9} {recall:>7.0%} {per_hour:>9.1f}")
                # a cell is only clean if it also *fires*: a threshold high
                # enough to reject everything scores zero false accepts, and
                # reporting that as a pass is how a deaf model gets promoted.
                # Ties break towards the highest threshold and longest patience
                # — same measured result, most margin against the audio this
                # corpus does not happen to contain
                key = (recall > 0 and false == 0, recall, -per_hour, threshold, patience)
                if best is None or key > best[0]:
                    best = (key, threshold, patience, recall, per_hour)
        if best:
            _, threshold, patience, recall, per_hour = best
            verdict = "clean" if best[0][0] else "NO CLEAN OPERATING POINT"
            print(f"  best: threshold {threshold:.4g}, patience {patience} — "
                  f"recall {recall:.0%}, {per_hour:.1f} false/h  [{verdict}]")
        _report_separation(positives, negatives, scored, column)
        _report_live(config, positives, negatives, scored, column, negative_hours)


# -- entry point --------------------------------------------------------


class _InputError(Exception):
    """Bad corpus arguments; the message is what the user sees."""


def _collect_files(
    paths: list[Path], positives: Path | None, negatives: Path | None
) -> tuple[list[Path], list[Path], list[Path]]:
    """Expand every group and refuse anything that cannot be scored."""
    positive_files = _expand([positives]) if positives else []
    negative_files = _expand([negatives]) if negatives else []
    # a directory that exists but holds no WAVs used to fall through to the
    # generic "the gate needs both flags" message — naming a cause the user
    # had already satisfied — and still exit 0, so a promotion script reading
    # exit 0 as "evaluation ran" proceeded with no recall measured at all
    for flag, given, found in (
        ("--positives", positives, positive_files),
        ("--negatives", negatives, negative_files),
    ):
        if given and not found:
            raise _InputError(f"no WAV files under {given} (given as {flag})")
    # deduped, order-preserving: a path reachable through more than one group
    # (--score-wav corpus/ --positives corpus/positives) would otherwise be
    # scored once per appearance, shifting the per-config columns in `scored`
    # so that --compare reads config 0's second result as config 1's first
    files = list(dict.fromkeys(_expand(paths) + positive_files + negative_files))
    missing = [p for p in files if not p.is_file()]
    if missing:
        raise _InputError("\n".join(f"not a file: {p}" for p in missing))
    if not files:
        raise _InputError("nothing to score — pass WAV files or directories")
    return files, positive_files, negative_files


def score_wavs(
    config: Config,
    paths: list[Path],
    model: str | None = None,
    engine: str | None = None,
    compare: str | None = None,
    positives: Path | None = None,
    negatives: Path | None = None,
) -> int:
    """Run the requested evaluation; returns a process exit code."""
    configs = [_override(config, engine, model)]
    if compare:
        # an "openwakeword:" prefix is still accepted: --compare took
        # ENGINE:MODEL while there was more than one engine to name
        prefix, sep, rest = compare.partition(":")
        other_model = rest if sep and prefix == "openwakeword" else compare
        if not other_model:
            print("--compare wants a model, e.g. "
                  "models/wakeword/shodan_listen.onnx")
            return 2
        configs.append(_override(config, None, other_model))

    try:
        files, positive_files, negative_files = _collect_files(
            paths, positives, negatives
        )
    except _InputError as exc:
        print(exc)
        return 2

    frame_ms = config.audio.frame_ms
    scored, failed = _report_files(configs, files, frame_ms)
    # unreadable files are out of every group, so the sweep, the gate and the
    # false-accepts-per-hour denominator all describe what was actually scored
    files = [p for p in files if p not in failed]
    positive_files = [p for p in positive_files if p not in failed]
    negative_files = [p for p in negative_files if p not in failed]
    for column, cfg in enumerate(configs):
        _report_sweep(cfg, files, scored, column)
    if positive_files and negative_files:
        _report_gate(configs, positive_files, negative_files, scored, frame_ms)
    elif positives or negatives:
        print("\n(the gate needs both --positives and --negatives; "
              "showing the sweep only)")
    _warn_on_asymmetry(configs)
    if failed:
        # exit non-zero even though the report is complete: a corpus that lost
        # files did not measure what the caller asked for
        print(f"\n{len(failed)} file(s) could not be scored and were skipped")
        return 2
    return 0


def _warn_on_asymmetry(configs: list[Config]) -> None:
    """Refuse a head-to-head verdict the sweep is not entitled to give.

    With a second stage configured, every `best:` line above describes stage 1
    alone — deliberately detuned, because the verifier is the real gate. Two
    such lines side by side rank two stage 1s, which is not the question
    `--compare` was asked.
    """
    if len(configs) < 2 or not any(_is_two_stage(c) for c in configs):
        return
    labels = " and ".join(_label(c) for c in configs)
    print(f"\nNOT COMPARABLE from the sweep: {labels} run a stage-2 verifier the "
          f"sweep cannot model, so every `best:` line above is stage 1 only. "
          f"Compare the live lines, or unset wakeword.stage2.model for a "
          f"stage-1 against stage-1 run.")
