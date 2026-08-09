"""Opt-in: drive a real violawake model, not the stub.

Every other wakeword test scripts the scores, so nothing in the suite proves
that audio reaches a real `violawake_sdk.WakeDetector` at all — a detector
that returned a constant would pass all of them. That is how a model whose
false accepts outscore its true ones reached a device unnoticed.

Model *quality* is not asserted here; it depends on the corpus and belongs in
the `--score-wav` gate. What this pins down is the plumbing: real audio in,
varying scores out, and a reset that genuinely clears the temporal buffer.

    OVS_TEST_VIOLA_MODEL=models/wakeword/some_wake.onnx pytest tests/test_wakeword_viola_model.py
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from openhab_voice_satellite.config import SAMPLE_RATE, AudioConfig, Config, WakewordConfig
from openhab_voice_satellite.wakeword import WAKE, build_detector

MODEL = os.environ.get("OVS_TEST_VIOLA_MODEL")

pytestmark = pytest.mark.skipif(
    not MODEL, reason="set OVS_TEST_VIOLA_MODEL to a real violawake .onnx to run"
)

FRAME_MS = 80
FRAME = SAMPLE_RATE * FRAME_MS // 1000
FRAMES = 40  # comfortably past the 9-embedding temporal warm-up


@pytest.fixture
def detector():
    pytest.importorskip("violawake_sdk")
    return build_detector(
        Config(
            audio=AudioConfig(frame_ms=FRAME_MS),
            wakeword=WakewordConfig(engine="violawake", model=MODEL, threshold=0.9),
        )
    )


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
    # the regression this guards: a constant series means the frames are not
    # reaching the model (wrong dtype, wrong length, silently swallowed)
    silence = _run(detector, np.zeros(FRAME * FRAMES, dtype=np.int16))
    detector.reset()
    noise = _run(detector, _noise(1))
    assert silence != noise, "identical scores for silence and noise: audio is not reaching the model"
    assert len(set(noise)) > 1, "score never changed across 40 frames"


def test_reset_clears_the_temporal_buffer(detector):
    _run(detector, _noise(2))
    detector.reset()
    after = _run(detector, _noise(3))
    # a temporal head scores 0.0 until its embedding buffer refills, so a
    # non-zero first frame would mean reset() left state behind
    assert after[0] == 0.0
    assert any(s > 0.0 for s in after), "scores never resumed after reset"
