"""--score-wav: the offline replay and the tuning grid it prints.

The grid exists to answer "what would have stopped this firing", so the one
property worth pinning hardest is that it answers with the app's own rule and
not a lookalike: `test_would_fire_agrees_with_a_live_detector` fails the
moment anyone re-implements the decision inside the replay.
"""

from __future__ import annotations

import numpy as np
import pytest

from openhab_voice_satellite.audio.wav import write_wav
from openhab_voice_satellite.score_wav import score_wav, would_fire

from .wakeword_stubs import (
    LIVEKIT_MODEL,
    StubLivekitModel,
    StubModel,
    install_livekit,
    install_openwakeword,
    make_config,
    make_detector,
    reset_stub_state,
)

FRAME = 1280
RATE = 16000


@pytest.fixture(autouse=True)
def _stubs():
    reset_stub_state()
    yield
    reset_stub_state()


def _wav(tmp_path, name, samples, rate=RATE):
    path = tmp_path / name
    write_wav(path, np.zeros(samples, dtype=np.int16), rate)
    return path


# -- the rule itself, no engine involved --------------------------------


def test_would_fire_counts_edge_triggered_detections():
    assert would_fire([0.9, 0.2, 0.9], 0.5, 1) == 2  # dips below 0.25, re-arms
    assert would_fire([0.9, 0.6, 0.9], 0.5, 1) == 1  # 0.6 never re-arms it


def test_would_fire_honours_patience():
    assert would_fire([0.9, 0.9], 0.5, 2) == 1
    assert would_fire([0.9, 0.1, 0.9], 0.5, 2) == 0


def test_would_fire_agrees_with_a_live_detector(monkeypatch):
    """The grid has to grade against the rule the app runs, not a copy.

    A copy would drift, and a tuning table computed from a drifted copy is
    worse than no table — it would be wrong in the direction nobody checks.
    """
    scores = [0.1, 0.8, 0.9, 0.2, 0.05, 0.75, 0.75, 0.9]
    detector = make_detector(
        "openwakeword", monkeypatch, {"wake": list(scores)},
        model="wake", threshold=0.7, patience=2,
    )
    live = sum(
        detector.process(np.zeros(FRAME, dtype=np.int16)) == "wake" for _ in scores
    )
    assert live == would_fire(scores, 0.7, 2)


# -- replay, on the engine that scores from its first frame --------------


def _openwakeword_config(**kwargs):
    return make_config("openwakeword", model="wake", **kwargs)


def test_the_trace_is_one_line_per_evaluation(tmp_path, capsys, monkeypatch):
    install_openwakeword(monkeypatch)
    StubModel.scripts = {"wake": [0.01, 0.02, 0.91, 0.30]}
    path = _wav(tmp_path, "dump.wav", 4 * FRAME)
    assert score_wav(_openwakeword_config(threshold=0.8), [path]) == 0
    out = capsys.readouterr().out
    assert "4 evaluations" in out
    assert "0.910  *" in out  # marked: at or above the configured threshold
    assert "peak 0.910" in out


def test_a_trailing_partial_frame_is_dropped(tmp_path, capsys, monkeypatch):
    # an engine fed a short frame mis-aligns its embedding grid for
    # everything after it, so the remainder is not scored at all
    install_openwakeword(monkeypatch)
    StubModel.scripts = {"wake": [0.1, 0.1, 0.1, 0.1, 0.1]}
    path = _wav(tmp_path, "ragged.wav", 4 * FRAME + 500)
    score_wav(_openwakeword_config(), [path])
    assert "4 frames, " in capsys.readouterr().out


def test_the_grid_shows_what_patience_would_have_prevented(tmp_path, capsys, monkeypatch):
    # the prod question in one table row: a lone spike above the bar
    install_openwakeword(monkeypatch)
    StubModel.scripts = {"wake": [0.02, 0.91, 0.05]}
    path = _wav(tmp_path, "spike.wav", 3 * FRAME)
    score_wav(_openwakeword_config(threshold=0.8), [path])
    row = next(
        line for line in capsys.readouterr().out.splitlines() if line.strip().startswith("0.90")
    )
    assert row.split()[1:] == ["1/1", "0/0", "0/0", "0/0"]


def test_every_file_starts_cold(tmp_path, capsys, monkeypatch):
    # file 2 must not ride file 1's context, or its trace grades audio that
    # was never in it
    install_openwakeword(monkeypatch)
    StubModel.scripts = {"wake": [0.1] * 8}
    paths = [_wav(tmp_path, "a.wav", 2 * FRAME), _wav(tmp_path, "b.wav", 2 * FRAME)]
    score_wav(_openwakeword_config(), paths)
    assert StubModel.instances[-1].reset_calls == len(paths)


def test_a_sample_rate_mismatch_is_refused_not_resampled(tmp_path, capsys, monkeypatch):
    install_openwakeword(monkeypatch)
    path = _wav(tmp_path, "8k.wav", 4 * FRAME, rate=8000)
    assert score_wav(_openwakeword_config(), [path]) == 1
    out = capsys.readouterr().out
    assert "8000 Hz" in out and "16000 Hz" in out


def test_an_unreadable_file_does_not_abort_the_corpus(tmp_path, capsys, monkeypatch):
    # a directory of field dumps will contain one; it must not cost the rest
    install_openwakeword(monkeypatch)
    StubModel.scripts = {"wake": [0.1] * 4}
    good = _wav(tmp_path, "good.wav", 4 * FRAME)
    assert score_wav(_openwakeword_config(), [tmp_path / "missing.wav", good]) == 1
    assert "good.wav" in capsys.readouterr().out


# -- priming, on the engine that really is muted -------------------------


def test_a_clip_too_short_to_prime_says_so(tmp_path, capsys, monkeypatch):
    install_livekit(monkeypatch)
    StubLivekitModel.scripts = {}
    path = _wav(tmp_path, "short.wav", RATE)  # 1 s: livekit needs 2
    config = make_config("livekit", model=LIVEKIT_MODEL)
    assert score_wav(config, [path]) == 1
    assert "no evaluation" in capsys.readouterr().out


def test_a_long_enough_clip_is_scored(tmp_path, capsys, monkeypatch):
    install_livekit(monkeypatch)
    StubLivekitModel.scripts = {}
    path = _wav(tmp_path, "long.wav", 3 * RATE)
    config = make_config("livekit", model=LIVEKIT_MODEL)
    assert score_wav(config, [path]) == 0
    assert "no evaluation" not in capsys.readouterr().out
