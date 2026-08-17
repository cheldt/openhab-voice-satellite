"""A stubbed wakeword engine, so the decision logic is testable without models.

The engine is driven by a dict of per-model score scripts, one score popped per
frame. Keys are the model names the config carries, which the shared tests set
to "wake"/"stop".

`ENGINES` is a tuple of one. The parameterised fixtures that read it are how the
decision layer stays engine-neutral in fact and not only in intent, so it is
kept as the seam a second engine would slot into rather than inlined away.
"""

from __future__ import annotations

import sys
import types

from openhab_voice_satellite.config import AudioConfig, WakewordConfig

ENGINES = ("openwakeword",)


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


def reset_stub_state() -> None:
    StubModel.scripts = {}
    StubModel.outputs = {}
    StubModel.instances = []


# -- construction ------------------------------------------------------


def make_detector(engine: str, monkeypatch, scripts: dict[str, list[float]],
                  frame_ms: int = 80, **config_kwargs):
    """Build `engine`'s detector with the given per-model score scripts."""
    install_openwakeword(monkeypatch)
    StubModel.scripts = {k: list(v) for k, v in scripts.items()}
    from openhab_voice_satellite.wakeword_oww import OpenWakewordDetector

    return OpenWakewordDetector(
        WakewordConfig(engine=engine, **config_kwargs), frame_ms
    )


def make_config(engine: str = "openwakeword", frame_ms: int = 80, **wakeword_kwargs):
    """A full Config for the engine, so the cross-model validators run."""
    from openhab_voice_satellite.config import Config

    return Config(
        audio=AudioConfig(frame_ms=frame_ms),
        wakeword=WakewordConfig(engine=engine, **wakeword_kwargs),
    )
