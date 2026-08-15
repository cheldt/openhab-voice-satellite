"""wakeforge engine: an ONNX featurizer streaming into an ONNX classifier head.

Two files trained together by ww_trainer (TigreGotico/wakeforge): the
featurizer turns raw audio into frames of features, the head scores the newest
window of them. The whole runtime contract is four tensors and a sigmoid, so it
lives here rather than behind a dependency — the same call this repo already
made for verifier_mel.py, and for the same reasons. The published plugin also
cannot be installed the way openwakeword and violawake are: its package
__init__ imports ovos_plugin_manager at module scope, so `pip install --no-deps`
yields something that cannot be imported at all.

The contract, from wakeforge's docs/guides/inference.md and export.md:

    featurizer   [B, T] float32   ->  [B, T_frames, F]   hop 160 (10 ms)
    head         [B, T_frames, F] ->  [B] logit          caller applies sigmoid

Tensor names are read off the sessions rather than hardcoded, and the head
takes a dynamic time dimension. Upstream's own smoothing (threshold, patience,
debounce) is not used: BaseWakewordDetector decides, as it does for every
engine here.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from .config import SAMPLE_RATE, WakewordConfig
from .violawake_ort import _session_options
from .wakeword import STOP, WAKE, BaseWakewordDetector

log = logging.getLogger(__name__)

# how many feature frames the head scores. Upstream truncates its streaming
# cache to this, and it is a property of how the head was trained rather than
# a deployment knob — a config field here would only invite detuning a model
# away from the window it was selected on.
WINDOW_FRAMES = 50

# long enough to fill the window several times over, so the probe measures a
# steady-state rate rather than a warm-up one
PROBE_SECONDS = 1.0


def _sigmoid(x: float) -> float:
    """Logistic, computed on whichever side of zero cannot overflow."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class WakeforgeRunner:
    """One featurizer + head pair, streaming a feature cache across frames."""

    def __init__(self, directory: str, featurizer: str, head: str) -> None:
        import onnxruntime as ort

        base = Path(directory)
        paths = {"featurizer": base / featurizer, "head": base / head}
        missing = [str(p) for p in paths.values() if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f"wakeforge model directory {directory!r} is missing {missing}; "
                f"it must hold the featurizer and head that ww_trainer exported "
                f"together (override the filenames with wakeword.wakeforge.*)"
            )
        # options at the construction site, as the stage-2 verifier does.
        # single_threaded_sessions() exists to reach a construction site we do
        # not own; here it would bind nothing and report having bound nothing.
        options = _session_options(ort)
        self._ext, self._head = (
            ort.InferenceSession(
                str(paths[name]), sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            for name in ("featurizer", "head")
        )
        self._ext_in = self._ext.get_inputs()[0].name
        self._ext_out = self._ext.get_outputs()[0].name
        self._head_in = self._head.get_inputs()[0].name
        self._head_out = self._head.get_outputs()[0].name
        self._cache: np.ndarray | None = None
        self._warned = False

    def _features(self, audio: np.ndarray) -> np.ndarray:
        """Feature frames for one chunk of unit-scale audio, as [T_frames, F]."""
        out = self._ext.run([self._ext_out], {self._ext_in: audio[None, :]})[0]
        return np.asarray(out, dtype=np.float32)[0]

    def _head_score(self, window: np.ndarray) -> float:
        logit = self._head.run([self._head_out], {self._head_in: window[None]})[0]
        score = _sigmoid(float(np.asarray(logit).ravel()[0]))
        if not self._warned and not 0.0 <= score <= 1.0:
            # a head that already applies its own sigmoid, or one exported with
            # a different output convention. Reported, never clamped: clamping
            # would hide it and every threshold below would be meaningless
            log.warning(
                "wakeforge head produced %.3f, which is not a probability — "
                "thresholds are being read against something else", score
            )
            self._warned = True
        return score

    def probe(self) -> tuple[float, float]:
        """Feature rate in fps, and the score this model gives to silence.

        Both come from one pass over digital silence at construction. The rate
        is the contract everything else rests on; the silence score catches a
        head whose output convention does not match the sigmoid applied above,
        which presents as a detector that fires constantly rather than one that
        never fires.
        """
        silence = np.zeros(int(PROBE_SECONDS * SAMPLE_RATE), dtype=np.float32)
        features = self._features(silence)
        fps = len(features) / PROBE_SECONDS
        return fps, self._head_score(features[-WINDOW_FRAMES:])

    def score(self, frame: np.ndarray) -> float:
        audio = frame.astype(np.float32) / 32768.0
        features = self._features(audio)
        self._cache = (
            features if self._cache is None
            else np.concatenate((self._cache, features))
        )
        self._cache = self._cache[-WINDOW_FRAMES:]
        return self._head_score(self._cache)

    def reset(self) -> None:
        # the sessions are stateless; the cache is the whole of this engine's
        # memory, which makes reset free — unlike openwakeword's, which this
        # project had to patch to stop it re-embedding four seconds of noise
        self._cache = None


class WakeforgeDetector(BaseWakewordDetector):
    """Feeds int16 frames to one wakeforge model pair per key."""

    def __init__(self, config: WakewordConfig) -> None:
        super().__init__(config)
        directories = {WAKE: config.model}
        if config.stop_model:
            directories[STOP] = config.stop_model
        forge = config.wakeforge
        self._runners = {
            key: WakeforgeRunner(directory, forge.featurizer, forge.head)
            for key, directory in directories.items()
        }
        for key, runner in self._runners.items():
            self._check_contract(key, runner, config.threshold)
        if STOP in self._runners:
            # no shared backbone, unlike openwakeword: the stop pair runs its
            # own featurizer, so it roughly doubles the per-frame cost
            log.info("wakeword stop model runs its own featurizer (double cost)")

    @staticmethod
    def _check_contract(key: str, runner: WakeforgeRunner, threshold: float) -> None:
        """Prove at startup that this pair scores probabilities at a sane rate.

        Named away from `_check`, which is BaseWakewordDetector's decision hook.
        """
        fps, silence = runner.probe()
        log.info(
            "wakeword model loaded (wakeforge, %s): featurizer %.1f fps, "
            "%d-frame window ≈ %.0f ms, silence scores %.3f",
            key, fps, WINDOW_FRAMES, WINDOW_FRAMES * 1000 / fps if fps else 0.0,
            silence,
        )
        if not fps:
            raise ValueError(
                f"wakeforge {key} featurizer produced no frames for "
                f"{PROBE_SECONDS:.0f} s of audio — it cannot score anything"
            )
        if silence >= threshold:
            raise ValueError(
                f"wakeforge {key} model scores digital silence at {silence:.3f}, "
                f"at or above wakeword.threshold {threshold} — this detector "
                f"would fire on an empty room. A head exported with its own "
                f"sigmoid does exactly this."
            )

    def _scores(self, frame: np.ndarray) -> dict[str, float]:
        return {key: runner.score(frame) for key, runner in self._runners.items()}

    def _engine_reset(self) -> None:
        for runner in self._runners.values():
            runner.reset()
