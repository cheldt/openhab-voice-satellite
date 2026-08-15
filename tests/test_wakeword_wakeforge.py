"""wakeforge specifics: the streaming feature cache and the startup contract.

The runner is exercised against fake onnxruntime sessions rather than a stub of
itself, because the caching and the scaling *are* the runner — a stub of it
would assert nothing. Model quality is the trainer's problem; what is pinned
here is that a frame becomes unit-scale audio, that the cache accumulates and
truncates at the window the head was trained on, and that a pair which does not
score probabilities is rejected at startup rather than in the field.
"""

from __future__ import annotations

import numpy as np
import pytest

from openhab_voice_satellite import wakeword_wakeforge
from openhab_voice_satellite.config import WakewordConfig
from openhab_voice_satellite.wakeword_wakeforge import (
    WINDOW_FRAMES,
    WakeforgeDetector,
    WakeforgeRunner,
)

from .wakeword_stubs import StubWakeforgeRunner, make_detector, reset_stub_state

FRAME = np.zeros(1280, dtype=np.int16)
HOP = 160          # samples per feature frame, wakeforge's documented 10 ms
DIM = 40           # feature width; arbitrary here, read off the arrays


class _FakeSession:
    """Either half of the pair, told apart by the filename it was given."""

    logits: list[float] = []

    def __init__(self, path, sess_options=None, providers=None):
        self.path = str(path)
        self.is_featurizer = "featurizer" in self.path
        self.feeds: list[np.ndarray] = []

    def get_inputs(self):
        return [type("I", (), {"name": "in"})]

    def get_outputs(self):
        return [type("O", (), {"name": "out"})]

    def run(self, output_names, feed):
        value = next(iter(feed.values()))
        self.feeds.append(value)
        if self.is_featurizer:
            return [np.zeros((1, value.shape[-1] // HOP, DIM), dtype=np.float32)]
        return [np.array([_FakeSession.logits.pop(0) if _FakeSession.logits else 0.0])]


@pytest.fixture
def runner_factory(tmp_path, monkeypatch):
    """A real WakeforgeRunner over fake sessions, with scripted head logits."""
    import onnxruntime as ort

    monkeypatch.setattr(ort, "InferenceSession", _FakeSession)

    def make(logits=(), featurizer="best_f1_featurizer.onnx", head="best_f1.onnx"):
        _FakeSession.logits = list(logits)
        for name in (featurizer, head):
            (tmp_path / name).write_bytes(b"")
        return WakeforgeRunner(str(tmp_path), featurizer, head)

    yield make
    _FakeSession.logits = []


# -- the runner ---------------------------------------------------------


def test_frames_reach_the_featurizer_as_unit_scale_audio(runner_factory):
    runner = runner_factory()
    runner.score(np.full(1280, 32767, dtype=np.int16))
    audio = runner._ext.feeds[0]
    assert audio.dtype == np.float32
    # /32768, matching what soundfile hands the trainer — /32767 would put a
    # full-scale sample at exactly 1.0 and quietly shift every feature
    assert audio.max() == pytest.approx(32767 / 32768)


def test_cache_accumulates_then_truncates_at_the_trained_window(runner_factory):
    runner = runner_factory()
    per_frame = 1280 // HOP  # 8 feature frames per 80 ms frame
    for i in range(1, 12):
        runner.score(FRAME)
        assert len(runner._cache) == min(i * per_frame, WINDOW_FRAMES)
    # once full it stays full: the head always scores the same span of audio
    assert len(runner._head.feeds[-1][0]) == WINDOW_FRAMES


def test_head_logit_becomes_a_probability(runner_factory):
    runner = runner_factory(logits=[0.0, 2.0, -2.0])
    assert runner.score(FRAME) == pytest.approx(0.5)
    assert runner.score(FRAME) == pytest.approx(0.880797, abs=1e-6)
    assert runner.score(FRAME) == pytest.approx(0.119203, abs=1e-6)


def test_a_saturated_logit_does_not_overflow(runner_factory):
    # np.exp(-x) on a large negative logit overflows; the score must simply
    # be 0.0, not a warning and a nan that poisons every percentile downstream
    runner = runner_factory(logits=[-1e4, 1e4])
    assert runner.score(FRAME) == 0.0
    assert runner.score(FRAME) == 1.0


def test_reset_drops_the_cache(runner_factory):
    runner = runner_factory()
    runner.score(FRAME)
    assert runner._cache is not None
    runner.reset()
    assert runner._cache is None


def test_probe_reports_the_feature_rate(runner_factory):
    fps, silence = runner_factory().probe()
    assert fps == pytest.approx(100.0)  # 16 kHz / hop 160
    assert silence == pytest.approx(0.5)


def test_missing_half_of_the_pair_names_the_directory(tmp_path, monkeypatch):
    import onnxruntime as ort

    monkeypatch.setattr(ort, "InferenceSession", _FakeSession)
    (tmp_path / "best_f1_featurizer.onnx").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="best_f1.onnx"):
        WakeforgeRunner(str(tmp_path), "best_f1_featurizer.onnx", "best_f1.onnx")


# -- the startup contract -----------------------------------------------


def test_a_model_that_scores_silence_is_rejected(runner_factory, monkeypatch):
    # a head exported with its own sigmoid: the second sigmoid maps [0,1] onto
    # [0.5, 0.73], so it fires on an empty room rather than never firing
    runner = runner_factory(logits=[3.0])
    monkeypatch.setattr(
        wakeword_wakeforge, "WakeforgeRunner", lambda *a, **k: runner
    )
    with pytest.raises(ValueError, match="scores digital silence"):
        WakeforgeDetector(WakewordConfig(engine="wakeforge", model="d", threshold=0.5))


def test_a_featurizer_that_produces_nothing_is_rejected(monkeypatch):
    class Deaf:
        def probe(self):
            return 0.0, 0.0

    monkeypatch.setattr(wakeword_wakeforge, "WakeforgeRunner", lambda *a, **k: Deaf())
    with pytest.raises(ValueError, match="produced no frames"):
        WakeforgeDetector(WakewordConfig(engine="wakeforge", model="d"))


# -- the detector -------------------------------------------------------


@pytest.fixture
def detector_factory(monkeypatch):
    reset_stub_state()

    def make(scripts, **config_kwargs):
        return make_detector("wakeforge", monkeypatch, scripts, **config_kwargs)

    yield make
    reset_stub_state()


def test_stop_model_gets_its_own_runner(detector_factory):
    detector_factory({"wake": [0.1], "stop": [0.1]}, model="wake", stop_model="stop")
    assert len(StubWakeforgeRunner.instances) == 2
    # each carries its own featurizer: no shared backbone to amortise
    assert {r.directory for r in StubWakeforgeRunner.instances} == {"wake", "stop"}


def test_reset_clears_every_runner(detector_factory):
    detector = detector_factory({"wake": [0.1], "stop": [0.1]},
                                model="wake", stop_model="stop")
    detector.reset()
    assert all(r.resets == 1 for r in StubWakeforgeRunner.instances)


def test_frames_reach_the_runner_unmutated(detector_factory):
    detector = detector_factory({"wake": [0.1]}, model="wake")
    frame = np.arange(1280, dtype=np.int16)
    detector.process(frame)
    assert np.array_equal(StubWakeforgeRunner.instances[0].frames[0], frame)
