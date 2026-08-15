"""Offline evaluator: chunking, the re-validating override, and the gate."""

from __future__ import annotations

import numpy as np
import pytest

from openhab_voice_satellite.audio.wav import write_wav
from openhab_voice_satellite.bench import (
    FLUSH_MS,
    _override,
    count_detections,
    score_file,
    score_wavs,
)
from openhab_voice_satellite.config import SAMPLE_RATE
from openhab_voice_satellite.wakeword import EdgeTrigger
from .wakeword_stubs import (
    StubModel,
    StubWakeDetector,
    install_openwakeword,
    install_violawake,
    make_config,
    make_detector,
    reset_stub_state,
)


@pytest.fixture(autouse=True)
def _clean_stubs():
    reset_stub_state()
    yield
    reset_stub_state()


def _wav(path, seconds=1.0):
    write_wav(path, np.zeros(int(SAMPLE_RATE * seconds), dtype=np.int16), SAMPLE_RATE)
    return path


# score_file appends FLUSH_MS of silence after every file so a deferred verdict
# can land; at 80 ms frames that is 12 more calls into the scripted engine
FLUSH_FRAMES = FLUSH_MS // 80


def _script(*per_file):
    """One 1 s file's worth of scores per argument, each padded for the flush.

    Files are scored in order, from one shared script, so a test that sizes its
    script to the real frames alone has the next file reading the previous
    file's flush.
    """
    out = []
    for scores in per_file:
        out += list(scores) + [0.0] * FLUSH_FRAMES
    return out


def _viola_config(monkeypatch, scripts, **kwargs):
    """A Config whose violawake engine pops `scripts` one score per frame."""
    install_violawake(monkeypatch)
    StubWakeDetector.scripts = {k: list(v) for k, v in scripts.items()}
    from openhab_voice_satellite import wakeword_viola

    monkeypatch.setattr(wakeword_viola, "patch_onnx_threads", lambda: True)
    return make_config("violawake", model="wake", **kwargs)


VERIFIER = {"verifier": {"model": "v.onnx", "mel_basis": "m.npy",
                         "threshold": 0.5, "delay_ms": 300}}


def _two_stage_config(monkeypatch, scripts, verdict, **kwargs):
    """`_viola_config` with the stage-2 verifier stubbed to a constant verdict.

    Stubbed at the same seam `test_wakeword_verifier.py` uses: the verifier's
    scoring quality belongs to the violawakeword repo, and what matters here is
    only that a swallowed WAKE never reaches the `live` count.
    """
    from openhab_voice_satellite import wakeword_viola

    monkeypatch.setattr(
        wakeword_viola.ViolaWakeDetector, "_build_verifier",
        staticmethod(lambda config: ("session", "frontend")),
    )
    monkeypatch.setattr(
        wakeword_viola.ViolaWakeDetector, "_verify", lambda self: verdict
    )
    return _viola_config(monkeypatch, scripts, viola=VERIFIER, **kwargs)


# -- the decision replay ------------------------------------------------


def test_count_detections_matches_the_detector_it_replays(monkeypatch):
    # the sweep is only trustworthy if it reproduces the live rule exactly
    scores = [0.9, 0.95, 0.1, 0.05, 0.9, 0.9]
    detector = make_detector("violawake", monkeypatch, {"wake": list(scores)},
                             model="wake", threshold=0.8)
    frame = np.zeros(1280, dtype=np.int16)
    live = sum(detector.process(frame) == "wake" for _ in scores)
    assert count_detections(np.asarray(scores), 0.8, 1) == live == 2


def test_count_detections_honours_patience_and_rearming():
    scores = np.asarray([0.9, 0.2, 0.9, 0.9])
    # patience 1: two crossings, each separated by a fall below threshold/2
    assert count_detections(scores, 0.8, 1) == 2
    # patience 2: the lone spike is rejected, only the held pair fires
    assert count_detections(scores, 0.8, 2) == 1
    # a score that never falls below half the threshold cannot fire twice
    assert count_detections(np.asarray([0.9, 0.5, 0.9]), 0.8, 1) == 1


def test_count_detections_leaves_no_state_between_calls():
    scores = np.asarray([0.9, 0.9])
    assert count_detections(scores, 0.8, 1) == count_detections(scores, 0.8, 1) == 1


def test_a_gated_frame_does_not_extend_a_patience_window():
    # NaN is a frame the detector never scored. Recording the previous score
    # instead would satisfy patience 3 from a single real crossing, which is
    # how a gated run reads as a better model than it is
    scores = np.array([0.9, np.nan, np.nan, np.nan])
    assert count_detections(scores, 0.8, 3) == 0
    assert count_detections(scores, 0.8, 1) == 1


def test_a_gap_does_not_break_a_patience_window():
    # the other half of the same rule: _scores() -> None leaves the history
    # untouched live, so the sweep must not treat a gap as a reset either
    assert count_detections(np.array([0.9, np.nan, 0.9]), 0.8, 2) == 1


def test_edge_trigger_feed_is_observe_plus_fired():
    a, b = EdgeTrigger(), EdgeTrigger()
    for score in (0.9, 0.1, 0.9):
        a.observe(score)
        assert a.fired(0.8, 1) == b.feed(score, 0.8, 1)


# -- scoring files ------------------------------------------------------


def test_score_file_chunks_at_the_configured_frame_size(tmp_path, monkeypatch):
    config = _viola_config(monkeypatch, {"wake": [0.4] * 20})
    result = score_file(config, _wav(tmp_path / "a.wav", seconds=1.0))
    # 1 s at 80 ms frames, and the trailing partial frame is dropped
    assert len(result.scores) == 12
    assert StubWakeDetector.instances[-1].frames[0].shape == (1280,)


def test_score_file_rejects_a_rate_the_detector_cannot_run(tmp_path, monkeypatch):
    config = _viola_config(monkeypatch, {"wake": []})
    path = tmp_path / "8k.wav"
    write_wav(path, np.zeros(8000, dtype=np.int16), 8000)
    # resampling here would silently change the thing being measured
    with pytest.raises(ValueError, match="8000 Hz"):
        score_file(config, path)


def test_score_file_returns_empty_for_audio_shorter_than_a_frame(tmp_path, monkeypatch):
    config = _viola_config(monkeypatch, {"wake": []})
    path = tmp_path / "tiny.wav"
    write_wav(path, np.zeros(100, dtype=np.int16), SAMPLE_RATE)
    assert len(score_file(config, path).scores) == 0


# -- the override -------------------------------------------------------


def test_override_revalidates_instead_of_copying(tmp_path, monkeypatch):
    # frame_ms 30 is not a multiple of violawake's 20 ms unit: switching engine
    # via --engine has to hit the same validator a config file would
    config = make_config("openwakeword", frame_ms=30)
    with pytest.raises(ValueError, match="frame_ms"):
        score_wavs(config, [_wav(tmp_path / "a.wav")], engine="violawake",
                   model="wake.onnx")


def test_switching_engine_drops_the_previous_engine_s_block(monkeypatch):
    # the headline A/B is a two-stage violawake config against a single-stage
    # engine. Carrying viola.verifier onto that engine is both meaningless and
    # a hard config error, so --compare would fail on the one config it exists
    # to compare against.
    config = make_config(
        "violawake", model="w.onnx",
        viola={"verifier": {"model": "v.onnx", "mel_basis": "m.npy"}},
    )
    swapped = _override(config, "wakeforge", "some_dir")
    assert swapped.wakeword.viola.verifier.model is None
    # a model-only override is not an engine change and must keep it
    assert _override(config, None, "other.onnx").wakeword.viola.verifier.model


def test_compare_needs_engine_and_model(tmp_path, monkeypatch):
    config = _viola_config(monkeypatch, {"wake": [0.1] * 40})
    assert score_wavs(config, [_wav(tmp_path / "a.wav")], compare="openwakeword") == 2


# -- end to end ---------------------------------------------------------


def test_score_wavs_reports_and_succeeds(tmp_path, monkeypatch, capsys):
    config = _viola_config(monkeypatch, {"wake": [0.95] * 40}, threshold=0.9)
    assert score_wavs(config, [_wav(tmp_path / "hit.wav")]) == 0
    out = capsys.readouterr().out
    assert "violawake:wake" in out
    assert "detection sweep" in out


def test_score_wavs_rejects_missing_paths(tmp_path):
    config = make_config("openwakeword")
    assert score_wavs(config, [tmp_path / "nope.wav"]) == 2


def test_score_wavs_needs_something_to_score(tmp_path):
    config = make_config("openwakeword")
    assert score_wavs(config, []) == 2


def test_gate_names_a_clean_operating_point(tmp_path, monkeypatch, capsys):
    positives = tmp_path / "pos"
    negatives = tmp_path / "neg"
    positives.mkdir()
    negatives.mkdir()
    _wav(positives / "p.wav")
    _wav(negatives / "n.wav")
    # a separable model: the positive file scores 0.95, the negative 0.10
    config = _viola_config(monkeypatch, {"wake": _script([0.95] * 12, [0.10] * 12)})
    assert score_wavs(config, [], positives=positives, negatives=negatives) == 0
    out = capsys.readouterr().out
    # every cell from 0.30 to 0.95 is clean at 100% recall; the pick has to be
    # the one with the most headroom, not the first one scanned
    assert "best: threshold 0.95, patience 3 — recall 100%, 0.0 false/h  [clean]" in out


def test_gate_says_so_when_nothing_separates(tmp_path, monkeypatch, capsys):
    positives = tmp_path / "pos"
    negatives = tmp_path / "neg"
    positives.mkdir()
    negatives.mkdir()
    _wav(positives / "p.wav")
    _wav(negatives / "n.wav")
    # the shipped failure: the negative outscores the positive, as
    # shodan_listen_viola_wake.onnx does on real speech
    config = _viola_config(monkeypatch, {"wake": _script([0.32] * 12, [0.85] * 12)})
    assert score_wavs(config, [], positives=positives, negatives=negatives) == 0
    assert "NO CLEAN OPERATING POINT" in capsys.readouterr().out


def test_gate_refuses_to_pass_a_model_that_never_fires(tmp_path, monkeypatch, capsys):
    positives = tmp_path / "pos"
    negatives = tmp_path / "neg"
    positives.mkdir()
    negatives.mkdir()
    _wav(positives / "p.wav")
    _wav(negatives / "n.wav")
    # scores below every swept threshold: zero false accepts, but zero recall.
    # "no false accepts" must not read as a pass when nothing fires at all
    config = _viola_config(monkeypatch, {"wake": _script([0.05] * 12, [0.05] * 12)})
    assert score_wavs(config, [], positives=positives, negatives=negatives) == 0
    out = capsys.readouterr().out
    assert "NO CLEAN OPERATING POINT" in out
    assert "recall 0%" in out


def test_gate_needs_both_sides(tmp_path, monkeypatch, capsys):
    positives = tmp_path / "pos"
    positives.mkdir()
    _wav(positives / "p.wav")
    config = _viola_config(monkeypatch, {"wake": [0.1] * 40})
    assert score_wavs(config, [], positives=positives) == 0
    assert "needs both --positives and --negatives" in capsys.readouterr().out


def test_gate_finds_the_operating_point_of_a_saturated_model(
    tmp_path, monkeypatch, capsys
):
    positives = tmp_path / "pos"
    negatives = tmp_path / "neg"
    positives.mkdir()
    negatives.mkdir()
    _wav(positives / "p.wav")
    _wav(negatives / "n.wav")
    # a sigmoid head that saturates: separable, but only above 0.99, where a
    # grid ending at 0.99 would report NO CLEAN OPERATING POINT for a model
    # that is in fact perfectly usable
    config = _viola_config(monkeypatch, {"wake": _script([0.9999] * 12, [0.999] * 12)})
    assert score_wavs(config, [], positives=positives, negatives=negatives) == 0
    out = capsys.readouterr().out
    assert "[clean]" in out
    assert "weakest positive 0.9999, strongest negative 0.999  [separable]" in out


def test_separation_line_names_an_overlap(tmp_path, monkeypatch, capsys):
    positives = tmp_path / "pos"
    negatives = tmp_path / "neg"
    positives.mkdir()
    negatives.mkdir()
    _wav(positives / "p.wav")
    _wav(negatives / "n.wav")
    config = _viola_config(monkeypatch, {"wake": _script([0.32] * 12, [0.85] * 12)})
    score_wavs(config, [], positives=positives, negatives=negatives)
    assert "[OVERLAPPING]" in capsys.readouterr().out


# -- the second stage the sweep cannot see ------------------------------


def test_a_deferred_wake_is_flushed_out_before_the_file_is_scored(
    tmp_path, monkeypatch
):
    # violawake holds each WAKE for viola.verifier.delay_ms so the verifier can
    # score a finished phrase. A wakeword clip ends right after the phrase, so
    # without a flush the verdict dies with the file: measured 42% recall on a
    # detector whose real figure is 98%.
    config = _two_stage_config(
        monkeypatch, {"wake": [0.9] * 200}, verdict=1.0, threshold=0.4
    )
    # 1 s of audio: stage 1 fires inside it, the verifier resolves after it
    result = score_file(config, _wav(tmp_path / "wake.wav", seconds=1.0))
    assert result.live >= 1
    # and the synthetic frames stay out of everything the corpus is measured by
    assert len(result.scores) == 12


def test_live_counts_what_process_returned_not_what_the_sweep_swept(
    tmp_path, monkeypatch, capsys
):
    positives = tmp_path / "pos"
    negatives = tmp_path / "neg"
    positives.mkdir()
    negatives.mkdir()
    _wav(positives / "p.wav")
    _wav(negatives / "n.wav")
    # stage 1 fires on both files; the verifier rejects everything. The sweep
    # sees only stage 1, so a config judged on it alone looks far worse than
    # the detector the app actually runs.
    config = _two_stage_config(
        monkeypatch, {"wake": [0.95] * 24}, verdict=0.0, threshold=0.4
    )
    assert score_wavs(config, [], positives=positives, negatives=negatives) == 0
    out = capsys.readouterr().out
    assert "live (two-stage, at the configured threshold 0.4" in out
    assert "recall 0%, 0.0 false/h" in out


def test_a_gated_run_says_how_much_it_skipped(tmp_path, monkeypatch, capsys):
    # real Silero over real digital silence: after the hangover the gate shuts
    # and most of the file is never scored. Without the line this prints, a
    # gated run just looks like a model with fewer false accepts.
    config = _viola_config(
        monkeypatch, {"wake": [0.1] * 80},
        vad_gate={"enabled": True, "hangover_ms": 700},
    )
    assert score_wavs(config, [_wav(tmp_path / "quiet.wav", seconds=4.0)]) == 0
    out = capsys.readouterr().out
    assert "vad gate:" in out and "suppressed)" in out


def test_compare_refuses_a_verdict_across_a_verifier_asymmetry(
    tmp_path, monkeypatch, capsys
):
    # the whole point of --compare: a two-stage violawake config against a
    # single-stage engine. Both columns are stage-1 sweeps, but only one of
    # them is the whole detector.
    config = _two_stage_config(monkeypatch, {"wake": [0.5] * 40}, verdict=1.0)
    install_openwakeword(monkeypatch)
    StubModel.scripts = {"other": [0.5] * 40}
    assert score_wavs(
        config, [_wav(tmp_path / "a.wav")], compare="openwakeword:other.onnx"
    ) == 0
    out = capsys.readouterr().out
    assert "NOT COMPARABLE" in out
    assert "stage-2 verifier the sweep cannot model" in out
