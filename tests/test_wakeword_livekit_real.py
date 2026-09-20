"""The livekit engine against the real livekit-wakeword library.

Everything else about the engine is tested through a stub. These pin the two
things a stub cannot: that the library still has the constructor shape the
engine calls, and that the ORT session wrapper actually catches every session
the library builds. Skips visibly (pytest -rs) when the extra is not installed.
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
