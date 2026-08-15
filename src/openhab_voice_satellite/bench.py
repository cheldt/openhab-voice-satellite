"""Offline wakeword evaluation (--score-wav): score WAVs as the app would.

The running app is a bad place to judge a wakeword model. Detections arrive
minutes apart, the audio that caused them is gone, and the only knobs anyone
can turn are a threshold and a patience count that cannot separate a model
whose false accepts outscore its true ones. This scores recorded audio through
`build_detector` — the same engines, the same frame size, the same decision
rule — so a candidate model can be judged before it reaches a device.

Three modes, increasing in strength:

  --score-wav FILE|DIR...        per-file score distribution and a
                                 threshold x patience detection sweep
  --compare ENGINE:MODEL         a second model over the same frames, so two
                                 engines are compared on identical audio
  --positives DIR --negatives DIR  recall and false accepts per hour, which is
                                 the only honest gate for promoting a model

Scores come from the engine; the detection counts come from `EdgeTrigger`, the
same object `BaseWakewordDetector` runs, so the sweep cannot drift from what
the app would have done with the same audio.

One thing the sweep structurally cannot see: a second stage. `score()` reports
the stage-1 score, so on a violawake config with a verifier the sweep counts
stage-1 triggers, most of which the verifier then swallows — and that config
runs stage 1 deliberately low *because* the verifier is the real gate. Every
report therefore also carries a `live` count, which is what `process()` itself
returned, and a `--compare` across that asymmetry refuses to name a winner.
"""

from __future__ import annotations

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
SWEEP_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99,
                    0.995, 0.999, 0.9995, 0.9999)
SWEEP_PATIENCE = (1, 2, 3)

WAV_SUFFIXES = (".wav",)


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
    apply — notably violawake's frame-size requirement, which otherwise fails
    silently at runtime by scoring 0.0 forever.
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

    The engine check is not redundant: `--compare` swaps engine and model but
    carries the rest of the config across, so a `viola.verifier` block survives
    onto an engine that has never heard of it and ignores it entirely.
    """
    wakeword = config.wakeword
    return wakeword.engine == "violawake" and bool(wakeword.viola.verifier.model)


# -- reporting ----------------------------------------------------------


def _report_files(
    configs: list[Config], paths: list[Path], frame_ms: int
) -> dict[Path, list[FrameScores]]:
    """Per-file score distributions, one column group per config."""
    scored: dict[Path, list[FrameScores]] = {}
    for config in configs:
        print(f"\n=== {_label(config)} ===")
        print(f"{'file':<34} {'dur':>6} {'frames':>7} "
              f"{'max':>7} {'p50':>7} {'p90':>7} {'p99':>7} {'fires':>6} {'live':>6}")
        for path in paths:
            result = score_file(config, path)
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
                f"{len(scores):>7} {scores.max():>7.3f} "
                f"{np.percentile(scores, 50):>7.3f} {np.percentile(scores, 90):>7.3f} "
                f"{np.percentile(scores, 99):>7.3f} {fires:>6} {result.live:>6}"
            )
        wakeword = config.wakeword
        print(f"  (fires = sweep detections at threshold {wakeword.threshold}, "
              f"patience {wakeword.patience}; live = what process() returned)")
        if _is_two_stage(config):
            print("  (stage-2 verifier configured: fires counts stage-1 triggers, "
                  "live counts what survived the verifier)")
    return scored


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
        name: [scored[p][column].scores.max() for p in group if len(scored[p][column].scores)]
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
    whole detector on a single-stage engine and only half of it on violawake
    with a verifier configured. This line is the other measurement.
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
        other_engine, _, other_model = compare.partition(":")
        if not other_model:
            print("--compare wants ENGINE:MODEL, e.g. "
                  "openwakeword:models/wakeword/shodan_listen.onnx")
            return 2
        configs.append(_override(config, other_engine, other_model))

    positive_files = _expand([positives]) if positives else []
    negative_files = _expand([negatives]) if negatives else []
    files = _expand(paths) + positive_files + negative_files
    missing = [p for p in files if not p.is_file()]
    if missing:
        for path in missing:
            print(f"not a file: {path}")
        return 2
    if not files:
        print("nothing to score — pass WAV files or directories")
        return 2

    frame_ms = config.audio.frame_ms
    scored = _report_files(configs, files, frame_ms)
    for column, cfg in enumerate(configs):
        _report_sweep(cfg, files, scored, column)
    if positive_files and negative_files:
        _report_gate(configs, positive_files, negative_files, scored, frame_ms)
    elif positives or negatives:
        print("\n(the gate needs both --positives and --negatives; "
              "showing the sweep only)")
    _warn_on_asymmetry(configs)
    return 0


def _warn_on_asymmetry(configs: list[Config]) -> None:
    """Refuse a head-to-head verdict when the two sides are not the same shape.

    A two-stage config's sweep rows are its stage 1 only, which is deliberately
    detuned because the verifier is the real gate. Ranking those rows against a
    single-stage model's rows flatters the single-stage model by roughly the
    verifier's rejection rate.
    """
    if len(configs) < 2 or len({_is_two_stage(c) for c in configs}) < 2:
        return
    two = ", ".join(_label(c) for c in configs if _is_two_stage(c))
    one = ", ".join(_label(c) for c in configs if not _is_two_stage(c))
    print(f"\nNOT COMPARABLE: {two} runs a stage-2 verifier the sweep cannot model; "
          f"{one} does not. Compare the live lines instead, or unset "
          f"wakeword.viola.verifier.model for a stage-1 against stage-1 run.")
