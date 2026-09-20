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


class StubMelFrontend:
    """Shape-faithful stand-in for livekit's MelSpectrogramFrontend.

    Emits the frame count the real melspectrogram.onnx would for the samples
    given — a 512 window at a hop of 160, no padding — so the engine's window
    arithmetic is exercised for real; the values are zeros.
    """

    def __init__(self) -> None:
        self.inputs: list[np.ndarray] = []

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio)
        self.inputs.append(audio)
        frames = max(0, (audio.shape[-1] - 512) // 160 + 1)
        return np.zeros((1, frames, 32), dtype=np.float32)


class StubSpeechEmbedding:
    """(batch, 76, 32) -> (batch, 96), counting the windows it was asked for."""

    def __init__(self) -> None:
        self.windows: list[np.ndarray] = []

    def __call__(self, mel_windows: np.ndarray) -> np.ndarray:
        mel_windows = np.asarray(mel_windows)
        assert mel_windows.shape[1:] == (76, 32), mel_windows.shape
        self.windows.extend(mel_windows)
        return np.zeros((mel_windows.shape[0], 96), dtype=np.float32)


class StubClassifierSession:
    """One classifier head: each run() pops the next scripted score."""

    def __init__(self, model: "StubLivekitModel", name: str) -> None:
        self._model = model
        self._name = name

    def run(self, _outputs, feed):
        (x,) = feed.values()
        self._model.head_inputs.append(np.asarray(x))
        script = self._model.scripts.get(self._name, [])
        score = script.pop(0) if script else self._model.idle_score
        return [np.array([[score]], dtype=np.float32)]


class StubLivekitModel:
    """livekit's WakeWordModel as the engine actually drives it.

    The engine never calls predict(); it streams through `_mel_frontend`,
    `_speech_embedding` and the `_classifiers` dict of (session, input_name),
    so those are what is stubbed. Scores are scripted per classifier name,
    one popped per head run.

    `idle_score` is deliberately not 0.0 so a test that needs a distinguishable
    quiet score has one; the engine no longer treats zeros as a sentinel.
    """

    scripts: dict[str, list[float]] = {}
    idle_score: float = 0.01
    instances: list["StubLivekitModel"] = []

    def __init__(self, models=None):
        self.classifiers: list[str] = []
        self.head_inputs: list[np.ndarray] = []
        self._mel_frontend = StubMelFrontend()
        self._speech_embedding = StubSpeechEmbedding()
        self._classifiers: dict[str, tuple[StubClassifierSession, str]] = {}
        import onnxruntime as ort

        # captured so a test can prove the session wrapper was active here
        # and restored afterwards
        self.ort_during_init = ort.InferenceSession
        StubLivekitModel.instances.append(self)
        for path in models or []:
            self.load_model(path)

    def load_model(self, model_path, model_name=None):
        name = model_name or Path(model_path).stem
        self.classifiers.append(name)
        self._classifiers[name] = (StubClassifierSession(self, name), "input")

    # what the tests read: the audio handed to the mel frontend, and the
    # embedding windows the engine asked for
    @property
    def mel_inputs(self) -> list[np.ndarray]:
        return self._mel_frontend.inputs

    @property
    def embedding_windows(self) -> list[np.ndarray]:
        return self._speech_embedding.windows

    def clear_calls(self) -> None:
        self.head_inputs.clear()
        self._mel_frontend.inputs.clear()
        self._speech_embedding.windows.clear()


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
    directly and rebuilding the frontend from it, rather than warming it with
    real frames, keeps the score scripts intact — 25 process() calls would
    pop 25 entries off them.

    A no-op for engines that hold no window, so the shared tests can call it
    unconditionally wherever an engine would otherwise be mid-prime.
    """
    warm = getattr(detector, "_prime_from_ring", None)
    if warm is None:
        return
    from openhab_voice_satellite.wakeword_livekit import WINDOW_SAMPLES

    detector._ring.extend(np.zeros(WINDOW_SAMPLES, dtype=np.int16))
    warm()


def make_config(engine: str = "openwakeword", frame_ms: int = 80, **wakeword_kwargs):
    """A full Config for the engine, so the cross-model validators run."""
    from openhab_voice_satellite.config import Config

    return Config(
        audio=AudioConfig(frame_ms=frame_ms),
        wakeword=WakewordConfig(engine=engine, **wakeword_kwargs),
    )
