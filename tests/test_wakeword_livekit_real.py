"""The livekit engine against the real livekit-wakeword library.

Everything else about the engine is tested through a stub. These pin what a
stub cannot: that the library still has the constructor shape the engine
calls and the private frontend attributes it streams through, that the ORT
session wrapper actually catches every session the library builds, and that
streaming the frontend frame by frame reproduces predict()'s scores exactly.
Skips visibly (pytest -rs) when the extra is not installed.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import numpy as np
import pytest

wakeword = pytest.importorskip("livekit.wakeword", reason="livekit extra not installed")

from openhab_voice_satellite.config import LivekitConfig, WakewordConfig  # noqa: E402
from openhab_voice_satellite.livekit_ort import TESTED_VERSION  # noqa: E402

MODEL = Path(__file__).resolve().parent.parent / "models" / "wakeword" / "livekit" / "hey_livekit.onnx"


def test_load_model_still_takes_the_name_the_engine_passes():
    params = inspect.signature(wakeword.WakeWordModel.load_model).parameters
    assert "model_name" in params, "LivekitDetector passes model_name= to name its keys"
    assert [p for p in params if p != "self"][:2] == ["model_path", "model_name"]


def test_the_constructor_takes_no_required_arguments():
    # the engine builds WakeWordModel() bare and loads classifiers afterwards
    params = inspect.signature(wakeword.WakeWordModel.__init__).parameters
    required = [
        name for name, p in params.items()
        if name != "self" and p.default is inspect.Parameter.empty
    ]
    assert required == []


def test_the_frontend_seams_the_engine_streams_through_still_exist():
    """wakeword_livekit drives these three private attributes directly."""
    model = wakeword.WakeWordModel()
    assert callable(model._mel_frontend)
    assert callable(model._speech_embedding)
    assert isinstance(model._classifiers, dict)
    # and the constants it aligns its grid to
    from livekit.wakeword.inference import model as inference

    assert (inference.EMBEDDING_WINDOW, inference.EMBEDDING_STRIDE, inference.MIN_EMBEDDINGS) == (76, 8, 16)


@pytest.mark.skipif(not MODEL.exists(), reason=f"no local livekit model at {MODEL}")
def test_a_loaded_classifier_is_a_session_and_input_name():
    model = wakeword.WakeWordModel()
    model.load_model(str(MODEL), model_name="wake")
    session, input_name = model._classifiers["wake"]
    assert hasattr(session, "run")
    assert isinstance(input_name, str)


def test_installed_version_is_the_tested_one():
    """A different release is not wrong, but the wrapper's claims are unchecked.

    Bump livekit_ort.TESTED_VERSION after re-reading its three construction
    sites (inference/model.py, models/feature_extractor.py) — that is what the
    constant means.
    """
    from importlib import metadata

    assert metadata.version("livekit-wakeword") == TESTED_VERSION


@pytest.mark.skipif(not MODEL.exists(), reason=f"no local livekit model at {MODEL}")
def test_no_ort_thread_pool_escapes_the_wrapper():
    """The whole point of livekit_ort: eighteen sessions, zero extra OS threads.

    ORT builds its intra-op pool at session construction, so a session that
    slipped past the wrapper shows up as new kernel threads right here.
    """
    from openhab_voice_satellite.wakeword_livekit import LivekitDetector

    before = len(os.listdir("/proc/self/task"))
    detector = LivekitDetector(
        WakewordConfig(engine="livekit", model=str(MODEL), livekit=LivekitConfig(hop_frames=1)),
        80,
    )
    assert len(os.listdir("/proc/self/task")) == before

    frame = (np.random.default_rng(0).standard_normal(1280) * 100).astype(np.int16)
    for _ in range(26):  # fill the 2 s window and score once
        detector.process(frame)
    assert detector.scored_last_frame is True
    assert 0.0 <= detector.score() <= 1.0
    assert len(os.listdir("/proc/self/task")) == before  # nothing lazy either


@pytest.mark.skipif(not MODEL.exists(), reason=f"no local livekit model at {MODEL}")
def test_streaming_reproduces_predict_on_the_aligned_window():
    """Frame-by-frame streaming must score what predict() scores, not roughly.

    The frontend is deterministic per mel frame, so the only difference is
    where the grids sit: the engine's newest mel frame ends on the newest
    sample and its newest embedding window ends on that frame, while predict()
    frames its buffer from the start (128 samples behind) and drops its last
    mel frame (160 more). Hence the 288-sample offset — see the module
    docstring of wakeword_livekit.
    """
    from openhab_voice_satellite.wakeword_livekit import WINDOW_SAMPLES, LivekitDetector

    detector = LivekitDetector(
        WakewordConfig(engine="livekit", model=str(MODEL), livekit=LivekitConfig(hop_frames=1)),
        80,
    )
    reference = detector._model  # predict() on the very same sessions

    rng = np.random.default_rng(1)
    audio = (rng.standard_normal(16000 * 5) * 300).astype(np.int16)
    audio[40000:52000] = (rng.standard_normal(12000) * 8000).astype(np.int16)  # something loud
    OFFSET = 288

    compared = 0
    for start in range(0, len(audio) - 1280 - OFFSET, 1280):
        detector.process(audio[start:start + 1280])
        end = start + 1280
        if not detector.scored_last_frame or end + OFFSET < WINDOW_SAMPLES:
            continue
        expected = reference.predict(audio[end + OFFSET - WINDOW_SAMPLES:end + OFFSET])["wake"]
        assert detector.score() == pytest.approx(expected, abs=1e-5)
        compared += 1
    assert compared >= 30
