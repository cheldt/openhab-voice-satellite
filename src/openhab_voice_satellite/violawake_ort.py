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

Two layers, because a seam that exists is not a seam that is used:

- `patch_onnx_threads()` replaces `OnnxBackend.load`, the construction site in
  the pinned version.
- `single_threaded_sessions()` additionally wraps `ort.InferenceSession` for
  the duration of detector construction, so a release that builds its sessions
  somewhere else is still covered. It counts what it bound and reports the OS
  thread delta across the block: a silent no-op — the failure this module
  exists to prevent — then shows up as a warning naming the leaked threads
  instead of as unexplained idle CPU in the field.

Fails open like the openwakeword patch: a version that has moved the seam
still runs, just hot. Drop this module once violawake accepts session options
of its own.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger(__name__)

_PATCH_FLAG = "_ovs_single_threaded"

# the seam is version-sensitive; deploy/install.md tells the user to pin this
TESTED_VERSION = "0.2.10"

_sessions_bound = 0


def sessions_bound() -> int:
    """How many ONNX sessions this module has forced options onto, ever."""
    return _sessions_bound


def _session_options(ort):
    """The options violawake never sets: one thread, no spinning."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # belt and braces: even a one-thread pool spins between runs by default
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return options


def _thread_count() -> int:
    """OS threads in this process, or -1 where that cannot be read.

    ORT worker threads are native, so `threading.active_count()` cannot see
    them; only the kernel's view counts here.
    """
    try:
        return len(os.listdir("/proc/self/task"))
    except OSError:
        return -1


def _warn_on_untested_version() -> None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("violawake")
    except PackageNotFoundError:
        return
    if installed != TESTED_VERSION:
        log.warning(
            "violawake %s installed, %s is the tested pin; the ONNX thread "
            "patch binds a seam that may have moved — watch the session/thread "
            "counts logged when the detector loads",
            installed,
            TESTED_VERSION,
        )


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

    _warn_on_untested_version()
    backend = onnx_backend.OnnxBackend
    if getattr(backend, _PATCH_FLAG, False):
        return True

    session_cls = onnx_backend.OnnxSession
    model_load_error = onnx_backend.ModelLoadError

    def load(self, model_path, **kwargs):
        global _sessions_bound

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        options = _session_options(ort)
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
        _sessions_bound += 1
        return session_cls(session)

    # assignments last; they cannot fail, so the patch is all-or-nothing
    backend.load = load
    setattr(backend, _PATCH_FLAG, True)
    log.debug("violawake ONNX sessions bound to a single non-spinning thread")
    return True


@contextmanager
def single_threaded_sessions():
    """Force our SessionOptions onto every ORT session built in this block.

    The seam patch above only covers the construction site the pinned version
    happens to use. This covers all of them, at the cost of being a global
    override — so it is scoped to detector construction and restored in
    `finally`, leaving faster-whisper, piper and every other ORT user in the
    process on their own settings.

    A caller that passes its own options keeps them; only the bare calls
    violawake makes are rewritten.
    """
    try:
        import onnxruntime as ort
    except Exception as exc:  # noqa: BLE001 - fail open, as the seam patch does
        log.warning("onnxruntime not importable (%s); sessions left unbound", exc)
        yield
        return

    original = ort.InferenceSession

    def wrapped(model_path, sess_options=None, **kwargs):
        global _sessions_bound

        if sess_options is None:
            sess_options = _session_options(ort)
            _sessions_bound += 1
        return original(model_path, sess_options=sess_options, **kwargs)

    threads_before = _thread_count()
    bound_before = _sessions_bound
    ort.InferenceSession = wrapped
    try:
        yield
    finally:
        ort.InferenceSession = original
        bound = _sessions_bound - bound_before
        leaked = _thread_count() - threads_before
        if threads_before < 0 or leaked <= 0:
            log.info("violawake bound %d ONNX sessions, no extra OS threads", bound)
        else:
            log.warning(
                "violawake bound %d ONNX sessions but the process gained %d OS "
                "threads; ORT sized a pool anyway and those workers spin between "
                "frames — expect the satellite to burn cores while idle",
                bound,
                leaked,
            )
