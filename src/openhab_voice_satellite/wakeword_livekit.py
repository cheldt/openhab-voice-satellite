"""livekit-wakeword engine, driven as a streaming frontend rather than a window.

livekit's own `predict()` is a pure function of the 2 s buffer handed to it: it
rebuilds the melspectrogram and all sixteen speech embeddings on every call
(inference/model.py). Measured on the Pi 5, single-threaded, that is 65 ms —
60 ms of it the sixteen embedding runs — so it could only ever run on every
Nth frame, and at hop 4 it still ate a fifth of a core while the satellite sat
idle.

But the frontend underneath is openWakeWord's: mel frames every 160 samples
from a 512-sample window with no padding (so every frame depends only on its
own samples), and a 96-dim embedding over 76 mel frames at a stride of 8. One
80 ms mic frame is 1280 samples, which is exactly 8 mel frames, which is
exactly one embedding stride. Between two consecutive windows fifteen of the
sixteen embeddings are identical; predict() recomputes them anyway.

So this module keeps what openWakeWord keeps — the last 352 raw samples of
context (one mel window less one hop), the last 76 mel frames and the last 16
embeddings — and per frame runs mel over 1632 samples, one embedding, and the
classifier head. On the Pi that is ~4.4 ms a frame against 65 ms per
evaluation before: cheaper at `hop_frames: 1` than the old engine was at 4,
with no hop latency.

The seams this depends on are the library's private attributes
(`_mel_frontend`, `_speech_embedding`, `_classifiers`); `livekit_ort` pins the
release they were checked against, and `_check_contract` streams a probe
window through the very same code path at startup so a rename fails there
rather than in the monitor loop.

Alignment: predict() frames its buffer from the start, so its last mel frame
ends 128 samples short of the newest sample, and it lays embedding windows
from the start too (`range(0, n - 76 + 1, 8)`), which leaves that last mel
frame unused. This engine's newest mel frame ends on the newest sample and its
newest embedding window ends on that frame — 288 samples (18 ms) later than
predict() looks. The scores are otherwise bit-for-bit those of predict() on
the buffer ending 288 samples further on; test_wakeword_livekit_real pins it.

A stop model stays nearly free: the mel and embeddings are shared across every
loaded classifier, so a second phrase costs one extra (1, 16, 96) -> (1, 1)
run per evaluation.
"""

from __future__ import annotations

import logging

import numpy as np

from .config import SAMPLE_RATE, WakewordConfig
from .livekit_ort import TESTED_VERSION, single_threaded_sessions, warn_on_untested_version
from .wakeword import STOP, WAKE, BaseWakewordDetector

log = logging.getLogger(__name__)

# The bundled melspectrogram.onnx: a 512-sample window every 160 samples, no
# padding, 32 bins. Frame i covers samples [160 i, 160 i + 512), so a chunk
# prefixed with the previous 352 samples yields exactly len(chunk) / 160 new
# frames, the newest ending on the newest sample, continuing the grid of the
# frames before it — the trick openWakeWord's `_streaming_melspectrogram`
# uses (utils.py prepends `160 * 3` and lands 128 samples behind instead).
MEL_HOP = 160
MEL_WINDOW = 512
MEL_CONTEXT = MEL_WINDOW - MEL_HOP
MEL_BINS = 32

# Google's speech_embedding: 76 mel frames in, 96 floats out, one window every
# 8 frames; the classifier head takes exactly 16 of them. These are how the
# heads were trained and are not knobs. Read from the library where it exports
# them, so a release that moves them fails loudly in _check_contract.
try:
    from livekit.wakeword.inference.model import (
        EMBEDDING_STRIDE,
        EMBEDDING_WINDOW,
        MIN_EMBEDDINGS,
    )
except Exception:  # noqa: BLE001 - the extra may be missing or renamed
    EMBEDDING_WINDOW = 76
    EMBEDDING_STRIDE = 8
    MIN_EMBEDDINGS = 16
EMBEDDING_DIM = 96

# The window predict() documents, kept as the priming contract: with no
# context the first frame yields 5 mel frames and every later one 8, so 25
# frames (32000 samples) give 197 mel frames and exactly the sixteen
# embeddings the head needs — which is also how long the detector stays muted
# after a reset. tail() callers and tests read it too.
WINDOW_SAMPLES = 2 * SAMPLE_RATE

# noise levels for the startup probe, in int16 RMS. Digital silence drives the
# mel log to its floor, which is an input no model is trained on, so a probe
# on zeros proves nothing about a real room.
PROBE_LEVELS = (5.0, 30.0, 200.0)


class _StreamingFrontend:
    """Raw context, mel frames and embeddings for one stream of frames.

    Two of these exist per detector: the live one behind the monitor loop and
    a scratch one `_score_window` builds for the startup probe, so both walk
    identical code.
    """

    def __init__(self, mel, embed, frame_samples: int) -> None:
        self._mel_fn = mel
        self._embed_fn = embed
        self._rows_per_frame = frame_samples // MEL_HOP
        # enough mel history for every window that ends inside one frame
        self._keep_rows = EMBEDDING_WINDOW + self._rows_per_frame
        self.reset()

    def reset(self) -> None:
        self._context = np.zeros(0, dtype=np.int16)
        self._mel = np.zeros((0, MEL_BINS), dtype=np.float32)
        self.embeddings = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    @property
    def ready(self) -> bool:
        return len(self.embeddings) >= MIN_EMBEDDINGS

    def push(self, frame: np.ndarray) -> bool:
        """Fold one frame in; True once the head has its sixteen embeddings."""
        audio = np.concatenate((self._context, np.asarray(frame, dtype=np.int16)))
        self._context = audio[-MEL_CONTEXT:]
        rows = self._mel_fn(audio.astype(np.float32) / 32768.0)
        if rows.ndim == 3:  # (1, frames, 32) from the library's batch axis
            rows = rows[0]
        self._mel = np.concatenate((self._mel, rows))[-self._keep_rows:]

        # one window per embedding stride this frame completed, oldest first,
        # the newest ending on the newest mel frame. Every frame after the
        # first adds a whole number of strides, so the grid never drifts.
        total = len(self._mel)
        ends = range(total - self._rows_per_frame + EMBEDDING_STRIDE, total + 1, EMBEDDING_STRIDE)
        windows = [self._mel[end - EMBEDDING_WINDOW:end] for end in ends if end >= EMBEDDING_WINDOW]
        if windows:
            new = self._embed_fn(np.stack(windows))
            self.embeddings = np.concatenate((self.embeddings, new))[-MIN_EMBEDDINGS:]
        return self.ready


class LivekitDetector(BaseWakewordDetector):
    """Streams frames through livekit's frontend; runs the head every `hop_frames`."""

    def __init__(self, config: WakewordConfig, frame_ms: int) -> None:
        from livekit.wakeword import WakeWordModel

        super().__init__(config, frame_ms)
        warn_on_untested_version()
        self._hop_frames = config.livekit.hop_frames
        self._countdown = 1
        self._frame_samples = SAMPLE_RATE * frame_ms // 1000
        if self._frame_samples % (MEL_HOP * EMBEDDING_STRIDE):
            # the config validator already holds frame_ms to multiples of 80;
            # this is the engine's own reason for that rule
            raise ValueError(
                f"audio.frame_ms {frame_ms} is not a whole number of embedding "
                f"strides ({MEL_HOP * EMBEDDING_STRIDE} samples); the livekit "
                "engine cannot keep its embedding grid aligned"
            )

        # canonical key -> the file backing it. Unlike openwakeword, which
        # derives its keys from model basenames and so collapses two models
        # that share one, livekit's load_model takes the name as an argument
        # (inference/model.py) — so we hand it ours and the classifier dict
        # comes back keyed the way _decide already wants.
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

        mel, embed, self._classifiers = self._bind_seams(self._model)
        self._frontend = _StreamingFrontend(mel, embed, self._frame_samples)
        self._check_contract()
        log.info(
            "livekit wakeword models loaded: %s (streaming frontend, head every %d frame%s)",
            {key: path for key, path in self._model_keys.items()},
            self._hop_frames,
            "" if self._hop_frames == 1 else "s",
        )

    @staticmethod
    def _bind_seams(model):
        """The three private attributes this engine drives directly.

        Looked up once, all before use, so a release that renames one fails
        at construction with the version to re-check rather than with an
        AttributeError on the first frame.
        """
        try:
            mel = model._mel_frontend
            embed = model._speech_embedding
            classifiers = dict(model._classifiers)
            for session, _input_name in classifiers.values():
                if not hasattr(session, "run"):
                    raise TypeError("classifier entry is not (session, input_name)")
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"livekit-wakeword no longer exposes the frontend this engine "
                f"streams through ({exc}); re-check inference/model.py against "
                f"the {TESTED_VERSION} layout and update wakeword_livekit"
            ) from exc
        if not callable(mel) or not callable(embed):
            raise RuntimeError(
                "livekit-wakeword's _mel_frontend / _speech_embedding are not "
                f"callable; re-check models/feature_extractor.py against {TESTED_VERSION}"
            )
        return mel, embed, classifiers

    def _head(self, embeddings: np.ndarray) -> dict[str, float]:
        """Run every classifier over sixteen embeddings, keyed by our names."""
        x = embeddings[np.newaxis].astype(np.float32)  # (1, 16, 96)
        scores = {}
        for key in self._model_keys:
            session, input_name = self._classifiers[key]
            scores[key] = float(session.run(None, {input_name: x})[0][0, 0])
        return scores

    def _score_window(self, window: np.ndarray) -> dict[str, float]:
        """Score one buffer the way the live loop would, from a cold start."""
        scratch = _StreamingFrontend(
            self._frontend._mel_fn, self._frontend._embed_fn, self._frame_samples
        )
        for start in range(0, len(window), self._frame_samples):
            scratch.push(window[start:start + self._frame_samples])
        if not scratch.ready:
            raise ValueError(
                f"{len(window)} samples yielded {len(scratch.embeddings)} of the "
                f"{MIN_EMBEDDINGS} embeddings the classifier needs — the bundled "
                "melspectrogram's framing has moved and the engine's window "
                "arithmetic no longer matches it"
            )
        return self._head(scratch.embeddings)

    def _check_contract(self) -> None:
        """Probe the loaded models once, so a bad one fails here and not later.

        Named away from `_check`, which is the base class's decision hook.

        Catches, at construction and on the same path the monitor loop runs: a
        key the engine did not load, a head exported without its sigmoid (the
        thresholds below it would be meaningless), a frontend whose framing no
        longer yields sixteen embeddings from WINDOW_SAMPLES, and any renamed
        seam. It also warms every session, so the first real evaluation is
        not a cold-start outlier.
        """
        missing = set(self._model_keys) - set(self._classifiers)
        if missing:
            raise ValueError(
                f"livekit model did not score {sorted(missing)}; got "
                f"{sorted(self._classifiers)} — the classifiers were not loaded "
                "under the names this engine reads"
            )
        rng = np.random.default_rng(0)
        for level in PROBE_LEVELS:
            noise = (rng.standard_normal(WINDOW_SAMPLES) * level).astype(np.int16)
            prediction = self._score_window(noise)
            for key in self._model_keys:
                score = prediction[key]
                if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise ValueError(
                        f"livekit model {self._model_keys[key]} scored {score} for "
                        f"{key!r}, which is not a probability — it was probably "
                        "exported without its sigmoid, and every threshold read "
                        "against it would be meaningless"
                    )

    def _prime_from_ring(self) -> None:
        """Rebuild the frontend from whatever the ring holds.

        For callers that fill the ring behind the engine's back (tests prime a
        detector this way instead of feeding 25 frames); the live loop never
        needs it because `process` hands every frame to `_scores`.
        """
        self._frontend.reset()
        audio = self._ring.tail(WINDOW_SAMPLES)
        for start in range(0, len(audio), self._frame_samples):
            self._frontend.push(audio[start:start + self._frame_samples])
        self._countdown = 1

    def _scores(self, frame: np.ndarray) -> dict[str, float] | None:
        """Fold the frame in; score on every `hop_frames`-th frame once primed.

        `process` has already put this frame into the ring; the frontend keeps
        its own 352-sample context, so it works from the frame itself.
        """
        if not self._frontend.push(frame):
            # priming: fewer than sixteen embeddings exist yet. Scoring a
            # shorter sequence is not an option — the head takes exactly 16.
            return None
        self._countdown -= 1
        if self._countdown > 0:
            return None
        self._countdown = self._hop_frames
        return self._head(self._frontend.embeddings)

    def _engine_reset(self) -> None:
        # the ring is cleared by the base right after this. Re-priming mutes
        # the detector for 25 frames (2 s). That is longer than openWakeWord's
        # post-reset behaviour: it zeroes only its first 5 frames (400 ms) and
        # scores on noise-primed context after that, so it can catch a phrase
        # spoken inside the 2 s that this engine cannot. Nothing the app does
        # needs a detection that soon after a reset (the wake earcon and
        # recording start fill the gap), so barge-in does not regress;
        # re-waking within 2 s of a barge-in cancel is the one case this
        # engine misses.
        self._frontend.reset()
        self._countdown = 1
