"""Bound violawake's ONNX Runtime sessions to one thread.

violawake builds every session as `ort.InferenceSession(path, providers=...)`
with no SessionOptions (backends/onnx_backend.py), so ORT sizes its intra-op
pool to the core count and those workers spin-wait between runs. At our 80 ms
inference cadence they never park: the pool burns whole cores while the
satellite sits idle. This project already paid for that lesson once on the
openwakeword path, which is why that engine passes `ncpu=1`.

violawake exposes neither a session_options kwarg nor a backend registry, so
the only seam is its `OnnxBackend.load`. We replace it rather than wrap it,
because the stock implementation drops every kwarg but `providers`.

Fails open like the openwakeword patch: a version that has moved the seam
still runs, just hot. Drop this module once violawake accepts session options
of its own.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

_PATCH_FLAG = "_ovs_single_threaded"


def _bind_seams():
    """Look up every seam the patch relies on; raises before any assignment."""
    import onnxruntime as ort

    from violawake_sdk.backends import onnx_backend

    for name in ("OnnxBackend", "OnnxSession", "ModelLoadError"):
        if not hasattr(onnx_backend, name):
            raise RuntimeError(f"no {name}")
    if not callable(getattr(onnx_backend.OnnxBackend, "load", None)):
        raise RuntimeError("OnnxBackend has no load()")
    return ort, onnx_backend


def patch_onnx_threads() -> bool:
    """Make violawake's ONNX sessions single-threaded and non-spinning.

    Idempotent, and safe to call before the first detector is built. Returns
    True when the patch is in place (including from an earlier call).
    """
    try:
        ort, onnx_backend = _bind_seams()
    except Exception as exc:
        log.warning(
            "violawake ONNX thread patch skipped (%s); ORT will run one "
            "spin-waiting pool per core at the frame cadence — expect the "
            "satellite to burn cores while idle",
            exc,
        )
        return False

    backend = onnx_backend.OnnxBackend
    if getattr(backend, _PATCH_FLAG, False):
        return True

    session_cls = onnx_backend.OnnxSession
    model_load_error = onnx_backend.ModelLoadError

    def load(self, model_path, **kwargs):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # belt and braces: even a one-thread pool spins between runs by default
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        providers = kwargs.get("providers") or getattr(
            self, "_providers", None
        ) or ["CPUExecutionProvider"]
        try:
            session = ort.InferenceSession(
                str(model_path), sess_options=options, providers=providers
            )
        except Exception as exc:
            raise model_load_error(
                f"ONNX Runtime failed to load {model_path}: {exc}"
            ) from exc
        return session_cls(session)

    # assignments last; they cannot fail, so the patch is all-or-nothing
    backend.load = load
    setattr(backend, _PATCH_FLAG, True)
    log.debug("violawake ONNX sessions bound to a single non-spinning thread")
    return True
