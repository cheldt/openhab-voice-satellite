"""Opt-in: drive a real wakeforge model pair, not the stub.

Everything else in the suite either scripts the scores or fakes the sessions,
so nothing proves audio reaches a real featurizer at all — and this engine's
whole runtime is an assumption about two tensors and a hop. A pair whose
featurizer runs at a different rate, or whose head expects a different feature
width, would sail through every other test.

Model *quality* is not asserted here; that is the `--score-wav` gate's job.
What this pins down is the contract: probabilities out, scores that follow the
audio, a reset that actually empties the cache, and the ~100 fps feature rate
the 50-frame window is measured in.

    OVS_TEST_WAKEFORGE_MODEL=models/wakeword/wakeforge/my_wake \\
        pytest tests/test_wakeword_wakeforge_model.py
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from openhab_voice_satellite.config import (
    SAMPLE_RATE,
    AudioConfig,
    Config,
    WakewordConfig,
)
from openhab_voice_satellite.wakeword import WAKE, build_detector
from openhab_voice_satellite.wakeword_wakeforge import WINDOW_FRAMES, WakeforgeRunner

MODEL = os.environ.get("OVS_TEST_WAKEFORGE_MODEL")

pytestmark = pytest.mark.skipif(
    not MODEL,
    reason="set OVS_TEST_WAKEFORGE_MODEL to a real wakeforge model directory to run",
)

FRAME_MS = 80
FRAME = SAMPLE_RATE * FRAME_MS // 1000
FRAMES = 40  # past the 50-feature-frame window several times over


@pytest.fixture
def config():
    return Config(
        audio=AudioConfig(frame_ms=FRAME_MS),
        wakeword=WakewordConfig(engine="wakeforge", model=MODEL, threshold=0.9),
    )


@pytest.fixture
def detector(config):
    return build_detector(config)


def _run(detector, pcm: np.ndarray) -> list[float]:
    scores = []
    for i in range(0, len(pcm) - FRAME + 1, FRAME):
        detector.process(pcm[i:i + FRAME])
        scores.append(detector.score(WAKE))
    return scores


def _noise(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 2000, FRAME * FRAMES).astype(np.int16)


def test_real_model_produces_usable_scores(detector):
    scores = _run(detector, _noise(0))
    assert len(scores) == FRAMES
    assert all(0.0 <= s <= 1.0 for s in scores), "scores must be probabilities"
    assert all(np.isfinite(scores))


def test_real_model_actually_listens_to_the_audio(detector):
    silence = _run(detector, np.zeros(FRAME * FRAMES, dtype=np.int16))
    detector.reset()
    noise = _run(detector, _noise(1))
    assert silence != noise, "identical scores for silence and noise: audio is not reaching the model"
    assert len(set(noise)) > 1, "score never changed across 40 frames"


def test_reset_clears_the_feature_cache(detector):
    _run(detector, _noise(2))
    detector.reset()
    runner = detector._runners[WAKE]
    assert runner._cache is None
    _run(detector, _noise(3))
    assert len(runner._cache) == WINDOW_FRAMES


def test_the_featurizer_runs_at_the_documented_hop(config):
    # the 50-frame window is only "500 ms" at a 10 ms hop. A pair trained with
    # a different hop still loads and still scores, but every threshold judged
    # offline would then describe a different span of audio than the app sees.
    forge = config.wakeword.wakeforge
    fps, _ = WakeforgeRunner(MODEL, forge.featurizer, forge.head).probe()
    assert fps == pytest.approx(100.0, rel=0.05), (
        f"featurizer runs at {fps:.1f} fps, so the {WINDOW_FRAMES}-frame window "
        f"covers {WINDOW_FRAMES * 1000 / fps:.0f} ms, not 500 ms"
    )
