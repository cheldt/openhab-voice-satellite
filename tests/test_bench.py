"""Offline evaluator: chunking, the re-validating override, and the gate."""

from __future__ import annotations

import numpy as np
import pytest

from openhab_voice_satellite.audio.wav import write_wav
from openhab_voice_satellite.bench import (
    count_detections,
    score_file,
    score_wavs,
)
from openhab_voice_satellite.config import SAMPLE_RATE
from openhab_voice_satellite.wakeword import EdgeTrigger
from .wakeword_stubs import (
    StubWakeDetector,
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


def _viola_config(monkeypatch, scripts, **kwargs):
    """A Config whose violawake engine pops `scripts` one score per frame."""
    install_violawake(monkeypatch)
    StubWakeDetector.scripts = {k: list(v) for k, v in scripts.items()}
    from openhab_voice_satellite import wakeword_viola

    monkeypatch.setattr(wakeword_viola, "patch_onnx_threads", lambda: True)
    return make_config("violawake", model="wake", **kwargs)


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


def test_edge_trigger_feed_is_observe_plus_fired():
    a, b = EdgeTrigger(), EdgeTrigger()
    for score in (0.9, 0.1, 0.9):
        a.observe(score)
        assert a.fired(0.8, 1) == b.feed(score, 0.8, 1)


# -- scoring files ------------------------------------------------------


def test_score_file_chunks_at_the_configured_frame_size(tmp_path, monkeypatch):
    config = _viola_config(monkeypatch, {"wake": [0.4] * 20})
    scores = score_file(config, _wav(tmp_path / "a.wav", seconds=1.0))
    # 1 s at 80 ms frames, and the trailing partial frame is dropped
    assert len(scores) == 12
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
    assert len(score_file(config, path)) == 0


# -- the override -------------------------------------------------------


def test_override_revalidates_instead_of_copying(tmp_path, monkeypatch):
    # frame_ms 30 is not a multiple of violawake's 20 ms unit: switching engine
    # via --engine has to hit the same validator a config file would
    config = make_config("openwakeword", frame_ms=30)
    with pytest.raises(ValueError, match="frame_ms"):
        score_wavs(config, [_wav(tmp_path / "a.wav")], engine="violawake",
                   model="wake.onnx")


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
    config = _viola_config(monkeypatch, {"wake": [0.95] * 12 + [0.10] * 12})
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
    config = _viola_config(monkeypatch, {"wake": [0.32] * 12 + [0.85] * 12})
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
    config = _viola_config(monkeypatch, {"wake": [0.05] * 24})
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
