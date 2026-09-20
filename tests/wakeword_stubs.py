"""A stubbed wakeword engine, so the decision logic is testable without models.

The engine is driven by a dict of per-model score scripts, one score popped per
frame. Keys are the model names the config carries, which the shared tests set
to "wake"/"stop".

The parameterised fixtures that read `ENGINES` are how the decision layer
stays engine-neutral in fact and not only in intent: an engine contributes
scores, never a decision, so every engine listed here is held to the identical
assertions.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

from openhab_voice_satellite.config import AudioConfig, LivekitConfig, WakewordConfig

ENGINES = ("openwakeword", "livekit")

LIVEKIT_MODEL = "models/wakeword/livekit/hey_livekit.onnx"
LIVEKIT_STOP_MODEL = "models/wakeword/livekit/stop.onnx"


# -- openwakeword ------------------------------------------------------


class StubModel:
    """Scripted scores: each predict() pops the next score per model."""

    scripts: dict[str, list[float]] = {}

    outputs: dict[str, int] = {}

    instances: list["StubModel"] = []

    def __init__(self, wakeword_models, inference_framework, ncpu=1):
        self.models = {name: object() for name in wakeword_models}
        self.model_outputs = {m: self.outputs.get(m, 1) for m in self.models}
        self.prediction_buffer: dict[str, list[float]] = {m: [0.0] for m in self.models}
        self.custom_verifier_models: dict[str, object] = {}
        self.custom_verifier_threshold = 0.1
        self.reset_calls = 0
        self.frames: list[object] = []
        StubModel.instances.append(self)

    def predict(self, frame):
        self.frames.append(frame)
        out = {}
        for name in self.models:
            script = self.scripts.get(name, [])
            score = script.pop(0) if script else 0.0
            self.prediction_buffer[name].append(score)
            out[name] = score
        return out

    def reset(self):
        self.reset_calls += 1


def install_openwakeword(monkeypatch) -> None:
    module = types.ModuleType("openwakeword.model")
    module.Model = StubModel
    package = types.ModuleType("openwakeword")
    package.model = module
    monkeypatch.setitem(sys.modules, "openwakeword", package)
    monkeypatch.setitem(sys.modules, "openwakeword.model", module)


# -- livekit -----------------------------------------------------------


class StubLivekitModel:
    """Scripted scores: each predict() pops the next score per classifier.

    `idle_score` is deliberately not 0.0. The engine's startup probe reads an
    all-zero result as livekit's short-window sentinel and raises, so a stub
    answering 0.0 would fail construction for a reason unrelated to the test.
    """

    scripts: dict[str, list[float]] = {}
    idle_score: float = 0.01
    instances: list["StubLivekitModel"] = []

    def __init__(self, models=None):
        self.classifiers: list[str] = []
        self.windows: list[np.ndarray] = []
        import onnxruntime as ort

        # captured so a test can prove the session wrapper was active here
        # and restored afterwards
        self.ort_during_init = ort.InferenceSession
        StubLivekitModel.instances.append(self)
        for path in models or []:
            self.load_model(path)

    def load_model(self, model_path, model_name=None):
        self.classifiers.append(model_name or Path(model_path).stem)

    def predict(self, audio):
        self.windows.append(np.asarray(audio))
        out = {}
        for name in self.classifiers:
            script = self.scripts.get(name, [])
            out[name] = script.pop(0) if script else self.idle_score
        return out


def install_livekit(monkeypatch) -> None:
    module = types.ModuleType("livekit.wakeword")
    module.WakeWordModel = StubLivekitModel
    package = types.ModuleType("livekit")
    # the real distribution ships livekit/ as an implicit namespace package;
    # without a __path__ the import machinery keeps searching and can find it
    package.__path__ = []
    package.wakeword = module
    monkeypatch.setitem(sys.modules, "livekit", package)
    monkeypatch.setitem(sys.modules, "livekit.wakeword", module)


def reset_stub_state() -> None:
    StubModel.scripts = {}
    StubModel.outputs = {}
    StubModel.instances = []
    StubLivekitModel.scripts = {}
    StubLivekitModel.idle_score = 0.01
    StubLivekitModel.instances = []


# -- construction ------------------------------------------------------


def make_detector(engine: str, monkeypatch, scripts: dict[str, list[float]],
                  frame_ms: int = 80, **config_kwargs):
    """Build `engine`'s detector with the given per-model score scripts.

    Engine differences that are not about the decision are normalised away
    here, so the shared tests compare like with like; each one is covered on
    its own in the per-engine test module.
    """
    scripts = {k: list(v) for k, v in scripts.items()}
    if engine == "livekit":
        install_livekit(monkeypatch)
        # scripts are installed *after* construction: the engine probes the
        # loaded model a few times to validate its contract, and those calls
        # would otherwise eat the first entries of the test's script
        StubLivekitModel.scripts = {}
        # the shared tests script one score per process() call, so the hop has
        # to be one frame or they would desync
        config_kwargs.setdefault("livekit", LivekitConfig(hop_frames=1))
        config_kwargs["model"] = LIVEKIT_MODEL
        if config_kwargs.get("stop_model"):
            config_kwargs["stop_model"] = LIVEKIT_STOP_MODEL
        from openhab_voice_satellite.wakeword_livekit import LivekitDetector

        detector = LivekitDetector(
            WakewordConfig(engine=engine, **config_kwargs), frame_ms
        )
        StubLivekitModel.scripts = scripts
        prime(detector)
        return detector

    install_openwakeword(monkeypatch)
    StubModel.scripts = scripts
    from openhab_voice_satellite.wakeword_oww import OpenWakewordDetector

    return OpenWakewordDetector(
        WakewordConfig(engine=engine, **config_kwargs), frame_ms
    )


def prime(detector) -> None:
    """Fill a window-scoring engine's context so the next frame is scored.

    openwakeword scores from its first frame; livekit is muted until 2 s of
    audio has arrived, and reset() clears that audio again. Priming the ring
    directly rather than warming it with real frames keeps the score scripts
    intact — 25 process() calls would pop 25 entries off them.

    A no-op for engines that hold no window, so the shared tests can call it
    unconditionally wherever an engine would otherwise be mid-prime.
    """
    ring = getattr(detector, "_ring", None)
    if ring is None:
        return
    from openhab_voice_satellite.wakeword_livekit import WINDOW_SAMPLES

    ring.extend(np.zeros(WINDOW_SAMPLES, dtype=np.int16))


def make_config(engine: str = "openwakeword", frame_ms: int = 80, **wakeword_kwargs):
    """A full Config for the engine, so the cross-model validators run."""
    from openhab_voice_satellite.config import Config

    return Config(
        audio=AudioConfig(frame_ms=frame_ms),
        wakeword=WakewordConfig(engine=engine, **wakeword_kwargs),
    )
