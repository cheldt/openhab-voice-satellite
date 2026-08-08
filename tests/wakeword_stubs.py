"""Stubbed wakeword engines, so the decision logic is testable without models.

Both engines are driven the same way: a dict of per-model score scripts, one
score popped per frame. Keys are the model names the config carries, which the
shared tests set to "wake"/"stop" so one script shape serves both engines.
"""

from __future__ import annotations

import sys
import types

from openhab_voice_satellite.config import AudioConfig, WakewordConfig

ENGINES = ("openwakeword", "violawake")


# -- openwakeword ------------------------------------------------------


class StubModel:
    """Scripted scores: each predict() pops the next score per model."""

    scripts: dict[str, list[float]] = {}

    outputs: dict[str, int] = {}

    def __init__(self, wakeword_models, inference_framework, ncpu=1):
        self.models = {name: object() for name in wakeword_models}
        self.model_outputs = {m: self.outputs.get(m, 1) for m in self.models}
        self.prediction_buffer: dict[str, list[float]] = {m: [0.0] for m in self.models}
        self.custom_verifier_models: dict[str, object] = {}
        self.custom_verifier_threshold = 0.1
        self.reset_calls = 0

    def predict(self, frame):
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


# -- violawake ---------------------------------------------------------


class StubWakeDetector:
    """One instance per configured model, as the real SDK requires."""

    scripts: dict[str, list[float]] = {}
    instances: list["StubWakeDetector"] = []

    def __init__(self, model, threshold=0.5, cooldown_s=0.0,
                 backend="onnx", confirm_count=1):
        self.model = model
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self.backend = backend
        self.confirm_count = confirm_count
        self.resets = 0
        self.frames: list[object] = []
        StubWakeDetector.instances.append(self)

    def process(self, frame):
        self.frames.append(frame)
        script = StubWakeDetector.scripts.get(self.model, [])
        return script.pop(0) if script else 0.0

    def reset(self):
        self.resets += 1


class StubNoiseProfiler:
    adapted: list[float] = []
    instances: list["StubNoiseProfiler"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.updates: list[object] = []
        StubNoiseProfiler.instances.append(self)

    def update(self, pcm):
        self.updates.append(pcm)
        return self.adapted.pop(0) if self.adapted else 0.5


class StubPowerManager:
    decisions: list[bool] = []
    instances: list["StubPowerManager"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        StubPowerManager.instances.append(self)

    def should_process(self, pcm):
        return self.decisions.pop(0) if self.decisions else True


def install_violawake(monkeypatch) -> types.ModuleType:
    package = types.ModuleType("violawake_sdk")
    package.WakeDetector = StubWakeDetector
    package.NoiseProfiler = StubNoiseProfiler
    package.PowerManager = StubPowerManager
    monkeypatch.setitem(sys.modules, "violawake_sdk", package)
    return package


def reset_stub_state() -> None:
    StubModel.scripts = {}
    StubModel.outputs = {}
    StubWakeDetector.scripts = {}
    StubWakeDetector.instances = []
    StubNoiseProfiler.adapted = []
    StubNoiseProfiler.instances = []
    StubPowerManager.decisions = []
    StubPowerManager.instances = []


# -- construction ------------------------------------------------------


def make_detector(engine: str, monkeypatch, scripts: dict[str, list[float]],
                  frame_ms: int = 80, **config_kwargs):
    """Build `engine`'s detector with the given per-model score scripts."""
    scripts = {k: list(v) for k, v in scripts.items()}
    config = WakewordConfig(engine=engine, **config_kwargs)
    if engine == "openwakeword":
        install_openwakeword(monkeypatch)
        StubModel.scripts = scripts
        from openhab_voice_satellite.wakeword_oww import OpenWakewordDetector

        return OpenWakewordDetector(config)

    install_violawake(monkeypatch)
    StubWakeDetector.scripts = scripts
    from openhab_voice_satellite import wakeword_viola

    # the ORT patch has no stub seam to bind to; it is covered on its own
    monkeypatch.setattr(wakeword_viola, "patch_onnx_threads", lambda: True)
    return wakeword_viola.ViolaWakeDetector(config, frame_ms)


def make_config(engine: str, frame_ms: int = 80, **wakeword_kwargs):
    """A full Config for the engine, so the cross-model validators run."""
    from openhab_voice_satellite.config import Config

    return Config(
        audio=AudioConfig(frame_ms=frame_ms),
        wakeword=WakewordConfig(engine=engine, **wakeword_kwargs),
    )
