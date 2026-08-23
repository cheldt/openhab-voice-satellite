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
    install_openwakeword,
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


def _scripted_config(monkeypatch, scripts, **kwargs):
    """A Config whose stubbed engine pops `scripts` one score per frame."""
    install_openwakeword(monkeypatch)
    StubModel.scripts = {k: list(v) for k, v in scripts.items()}
    return make_config("openwakeword", model="wake", **kwargs)


STAGE2 = {"model": "v.onnx", "mel_basis": "m.npy",
          "threshold": 0.5, "delay_ms": 300}


def _two_stage_config(monkeypatch, scripts, verdict, **kwargs):
    """`_scripted_config` with the stage-2 verifier stubbed to a fixed verdict.

    Stubbed at the same seam `test_wakeword_verifier.py` uses: the verifier's
    scoring quality belongs to the training pipeline, and what matters here is
    only that a swallowed WAKE never reaches the `live` count.
    """
    from openhab_voice_satellite import wakeword

    monkeypatch.setattr(
        wakeword.BaseWakewordDetector, "_build_verifier",
        staticmethod(lambda config: ("session", "frontend")),
    )
    monkeypatch.setattr(
        wakeword.BaseWakewordDetector, "_verify", lambda self: verdict
    )
    return _scripted_config(monkeypatch, scripts, stage2=STAGE2, **kwargs)


# -- the decision replay ------------------------------------------------


def test_count_detections_matches_the_detector_it_replays(monkeypatch):
    # the sweep is only trustworthy if it reproduces the live rule exactly
    scores = [0.9, 0.95, 0.1, 0.05, 0.9, 0.9]
    detector = make_detector("openwakeword", monkeypatch, {"wake": list(scores)},
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


def test_edge_trigger_feed_is_observe_plus_fired():
    a, b = EdgeTrigger(), EdgeTrigger()
    for score in (0.9, 0.1, 0.9):
        a.observe(score)
        assert a.fired(0.8, 1) == b.feed(score, 0.8, 1)


# -- scoring files ------------------------------------------------------


def test_score_file_chunks_at_the_configured_frame_size(tmp_path, monkeypatch):
    config = _scripted_config(monkeypatch, {"wake": [0.4] * 20})
    result = score_file(config, _wav(tmp_path / "a.wav", seconds=1.0))
    # 1 s at 80 ms frames, and the trailing partial frame is dropped
    assert len(result.scores) == 12
    assert StubModel.instances[-1].frames[0].shape == (1280,)


def test_score_file_rejects_a_rate_the_detector_cannot_run(tmp_path, monkeypatch):
    config = _scripted_config(monkeypatch, {"wake": []})
    path = tmp_path / "8k.wav"
    write_wav(path, np.zeros(8000, dtype=np.int16), 8000)
    # resampling here would silently change the thing being measured
    with pytest.raises(ValueError, match="8000 Hz"):
        score_file(config, path)


def test_score_file_returns_empty_for_audio_shorter_than_a_frame(tmp_path, monkeypatch):
    config = _scripted_config(monkeypatch, {"wake": []})
    path = tmp_path / "tiny.wav"
    write_wav(path, np.zeros(100, dtype=np.int16), SAMPLE_RATE)
    assert len(score_file(config, path).scores) == 0


# -- the override -------------------------------------------------------


def test_override_revalidates_instead_of_copying(tmp_path, monkeypatch):
    # a model_copy would take any value at all; the override has to hit the
    # same validators a config file would, so an unknown engine is rejected
    config = make_config("openwakeword")
    with pytest.raises(ValueError, match="engine"):
        score_wavs(config, [_wav(tmp_path / "a.wav")], engine="nosuchengine",
                   model="wake.onnx")


def test_swapping_the_model_keeps_the_second_stage(monkeypatch):
    # comparing two two-stage systems is the whole point of --compare, so the
    # verifier has to survive a model swap
    config = make_config(
        model="w.onnx", stage2={"model": "v.onnx", "mel_basis": "m.npy"},
    )
    swapped = _override(config, None, "other.onnx")
    assert swapped.wakeword.model == "other.onnx"
    assert swapped.wakeword.stage2.model == "v.onnx"


def test_compare_still_accepts_the_old_engine_prefix(tmp_path, monkeypatch):
    # --compare took ENGINE:MODEL while there was more than one engine to name
    config = _scripted_config(monkeypatch, {"wake": [0.1] * 40})
    StubModel.scripts["other.onnx"] = [0.1] * 40
    assert score_wavs(
        config, [_wav(tmp_path / "a.wav")], compare="openwakeword:other.onnx"
    ) == 0


# -- end to end ---------------------------------------------------------


def test_score_wavs_reports_and_succeeds(tmp_path, monkeypatch, capsys):
    config = _scripted_config(monkeypatch, {"wake": [0.95] * 40}, threshold=0.9)
    assert score_wavs(config, [_wav(tmp_path / "hit.wav")]) == 0
    out = capsys.readouterr().out
    assert "openwakeword:wake" in out
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
    config = _scripted_config(monkeypatch, {"wake": _script([0.95] * 12, [0.10] * 12)})
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
    # the failure that made this tool necessary: the negative outscores the
    # positive, which no threshold or patience count can rescue
    config = _scripted_config(monkeypatch, {"wake": _script([0.32] * 12, [0.85] * 12)})
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
    config = _scripted_config(monkeypatch, {"wake": _script([0.05] * 12, [0.05] * 12)})
    assert score_wavs(config, [], positives=positives, negatives=negatives) == 0
    out = capsys.readouterr().out
    assert "NO CLEAN OPERATING POINT" in out
    assert "recall 0%" in out


def test_compare_columns_survive_a_path_reached_through_two_groups(
    tmp_path, monkeypatch, capsys
):
    # `--score-wav corpus/ --positives corpus/pos --negatives corpus/neg`
    # reaches every gated file twice. Without dedup each appearance is scored
    # per config, so scored[p][1] is config 0's *second* result and every
    # model-B report silently describes model A.
    corpus = tmp_path / "corpus"
    positives = corpus / "pos"
    negatives = corpus / "neg"
    positives.mkdir(parents=True)
    negatives.mkdir()
    _wav(positives / "p.wav")
    _wav(negatives / "n.wav")
    # scripts sized for each file scored exactly once per config: a duplicate
    # would drain them early and read 0.0 where model B's scores belong
    config = _scripted_config(monkeypatch, {"wake": _script([0.1] * 12, [0.95] * 12)})
    StubModel.scripts["other"] = _script([0.1] * 12, [0.2] * 12)
    assert score_wavs(
        config, [corpus], compare="other", positives=positives, negatives=negatives
    ) == 0
    out = capsys.readouterr().out
    assert "weakest positive 0.95, strongest negative 0.1" in out
    assert "weakest positive 0.2, strongest negative 0.1" in out


def test_gate_needs_both_sides(tmp_path, monkeypatch, capsys):
    positives = tmp_path / "pos"
    positives.mkdir()
    _wav(positives / "p.wav")
    config = _scripted_config(monkeypatch, {"wake": [0.1] * 40})
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
    config = _scripted_config(monkeypatch, {"wake": _script([0.9999] * 12, [0.999] * 12)})
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
    config = _scripted_config(monkeypatch, {"wake": _script([0.32] * 12, [0.85] * 12)})
    score_wavs(config, [], positives=positives, negatives=negatives)
    assert "[OVERLAPPING]" in capsys.readouterr().out


# -- the second stage the sweep cannot see ------------------------------


def test_a_deferred_wake_is_flushed_out_before_the_file_is_scored(
    tmp_path, monkeypatch
):
    # a two-stage detector holds each WAKE for stage2.delay_ms so the verifier
    # can score a finished phrase. A wakeword clip ends right after the phrase,
    # so without a flush the verdict dies with the file: measured 42% recall on
    # a detector whose real figure is 98%.
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


def test_compare_refuses_a_sweep_verdict_when_a_second_stage_is_configured(
    tmp_path, monkeypatch, capsys
):
    # --compare carries the second stage onto both columns. Every best: line is
    # then stage 1 alone, and ranking two stage 1s is not the question
    # --compare was asked.
    config = _two_stage_config(monkeypatch, {"wake": [0.5] * 40}, verdict=1.0)
    install_openwakeword(monkeypatch)
    StubModel.scripts = {"other": [0.5] * 40}
    assert score_wavs(
        config, [_wav(tmp_path / "a.wav")], compare="openwakeword:other.onnx"
    ) == 0
    out = capsys.readouterr().out
    assert "NOT COMPARABLE from the sweep" in out
    assert "Compare the live lines" in out
