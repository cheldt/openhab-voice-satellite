"""livekit-wakeword engine: a stateless classifier over a 2 s trailing window.

Where openWakeWord keeps a streaming frontend and folds one new 80 ms frame
into it per call, livekit's `predict()` is a pure function of the buffer handed
to it — it rebuilds the melspectrogram and all sixteen speech embeddings every
time (inference/model.py). Same frontend, same (16, 96) feature matrix, a
heavier conv-attention head on top.

Two consequences shape this module:

* It costs ~14x openWakeWord per evaluation (measured single-threaded on x86:
  13.5 ms against ~0.9 ms), so it does not run on every frame. See
  `WakewordConfig.livekit.hop_frames` and the skip contract in
  `BaseWakewordDetector._scores`.
* A stop model is nearly free. The mel and all sixteen embeddings are shared
  across every loaded classifier, so a second phrase costs one extra
  (1, 16, 96) -> (1, 1) run. openWakeWord runs a second model end to end.
"""

from __future__ import annotations

import logging

import numpy as np

from .config import SAMPLE_RATE, WakewordConfig
from .livekit_ort import single_threaded_sessions, warn_on_untested_version
from .wakeword import STOP, WAKE, BaseWakewordDetector

log = logging.getLogger(__name__)

# The classifier takes exactly 16 embeddings, built from 76-frame mel windows
# at a stride of 8, so it needs 196 mel frames. The bundled melspectrogram
# emits 196 frames at 31712 samples and 197 at 32000 — below that every score
# comes back as exactly 0.0 with nothing logged (inference/model.py returns a
# zero dict rather than raising). 2 s is the documented window, lands on a
# whole 80 ms frame, and leaves one mel frame of slack over the cliff.
#
# Longer would be waste, not safety: the extra embeddings are built and then
# dropped by the [-16:] slice inside predict(). This is not a config knob for
# the same reason the 16 is not one — it is a property of how the head was
# trained, and moving it only detunes the model away from its own window.
WINDOW_SAMPLES = 2 * SAMPLE_RATE

# noise levels for the startup probe, in int16 RMS. Digital silence drives the
# mel log to its floor, which is an input no model is trained on, so a probe
# on zeros proves nothing about a real room.
PROBE_LEVELS = (5.0, 30.0, 200.0)


class LivekitDetector(BaseWakewordDetector):
    """Scores a 2 s window out of the base's ring every `hop_frames` frames."""

    def __init__(self, config: WakewordConfig, frame_ms: int) -> None:
        from livekit.wakeword import WakeWordModel

        super().__init__(config, frame_ms)
        warn_on_untested_version()
        self._hop_frames = config.livekit.hop_frames
        self._countdown = 1

        # canonical key -> the file backing it. Unlike openwakeword, which
        # derives its keys from model basenames and so collapses two models
        # that share one, livekit's load_model takes the name as an argument
        # (inference/model.py) — so we hand it ours and predict() comes back
        # keyed the way _decide already wants. No positional mapping, and the
        # shared-basename failure that wakeword_oww._check_keys guards against
        # cannot arise here.
        self._model_keys = {WAKE: config.model}
        if config.stop_model:
            self._model_keys[STOP] = config.stop_model
        if config.stop_model == config.model:
            log.warning(
                "wakeword.stop_model is the same file as wakeword.model; both "
                "keys will always score identically and one classifier run per "
                "evaluation is wasted"
            )

        # every session this engine will own is built in here: the mel and
        # embedding frontends in the constructor, one per classifier in
        # load_model. Nothing builds a session later.
        with single_threaded_sessions():
            self._model = WakeWordModel()
            for key, path in self._model_keys.items():
                self._model.load_model(path, model_name=key)

        self._check_contract()
        log.info(
            "livekit wakeword models loaded: %s (hop %d frames)",
            {key: path for key, path in self._model_keys.items()},
            self._hop_frames,
        )

    def _check_contract(self) -> None:
        """Probe the loaded models once, so a bad one fails here and not later.

        Named away from `_check`, which is the base class's decision hook.

        Three failures this catches at construction that would otherwise
        surface in the monitor loop hours later, all of them silent: a key the
        engine does not actually emit, a head exported without its sigmoid (the
        thresholds below it would be meaningless), and a window too short for
        sixteen embeddings, whose zero dict is indistinguishable from a quiet
        room. A wrong-shaped classifier output raises from inside predict()
        here rather than on the first live evaluation.

        It also warms all eighteen sessions, so the first real evaluation is
        not a cold-start outlier.
        """
        rng = np.random.default_rng(0)
        all_zero = True
        for level in PROBE_LEVELS:
            noise = (rng.standard_normal(WINDOW_SAMPLES) * level).astype(np.int16)
            prediction = self._model.predict(noise)
            missing = set(self._model_keys) - set(prediction)
            if missing:
                raise ValueError(
                    f"livekit model did not score {sorted(missing)}; got "
                    f"{sorted(prediction)} — the classifiers were not loaded "
                    "under the names this engine reads"
                )
            for key in self._model_keys:
                score = prediction[key]
                if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise ValueError(
                        f"livekit model {self._model_keys[key]} scored {score} for "
                        f"{key!r}, which is not a probability — it was probably "
                        "exported without its sigmoid, and every threshold read "
                        "against it would be meaningless"
                    )
                all_zero = all_zero and score == 0.0
        if all_zero:
            raise ValueError(
                f"livekit model scored exactly 0.0 on every probe: "
                f"{WINDOW_SAMPLES} samples did not yield the sixteen embeddings "
                "the classifier needs, so predict() is returning its zero "
                "sentinel — the bundled melspectrogram's frame count has moved"
            )

    def _scores(self, frame: np.ndarray) -> dict[str, float] | None:
        """Score the trailing window, or None on a frame we skip.

        `frame` is unused: this engine reads a window, not a frame, and
        `process` has already put this frame into the ring.
        """
        if len(self._ring) < WINDOW_SAMPLES:
            # priming. Padding the short window instead would hand predict()
            # too few samples for sixteen embeddings, and its zero sentinel
            # would reach EdgeTrigger as a genuine 0.0 — low enough to re-arm
            # a trigger that a reset had just disarmed.
            return None
        self._countdown -= 1
        if self._countdown > 0:
            return None
        self._countdown = self._hop_frames

        # no copy: tail() can hand back a live view, but predict() converts
        # with astype and flattens, both of which copy, and this loop is
        # single-threaded. The base's public tail() copies because its callers
        # keep the array; this one is consumed before the next frame arrives.
        prediction = self._model.predict(self._ring.tail(WINDOW_SAMPLES))
        # rebuilt against our keys rather than returned as-is, so a key the
        # engine stops emitting fails here instead of as a KeyError inside the
        # decision layer
        return {key: float(prediction[key]) for key in self._model_keys}

    def _engine_reset(self) -> None:
        # the model holds no audio state — the window lives in the base's ring,
        # which reset() clears right after this. Re-priming then mutes the
        # detector for a full 2 s, which is within 40 ms of what openWakeWord's
        # own context window costs after a reset, so barge-in does not regress.
        self._countdown = 1
