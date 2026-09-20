"""ONNX Runtime session options for livekit-wakeword.

livekit-wakeword builds every one of its sessions with a bare
`ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])` and
exposes no options argument — three sites, all of them private:
`models/feature_extractor.py` (the mel frontend and the speech embedding) and
`inference/model.py` (each classifier). So the only way to bound its threading
is to own the constructor for the duration of our own construction.

That is not a nicety. Measured here on x86 with onnxruntime 1.28, one
`predict()` over a 2 s window:

    single thread, no spinning    13.5 ms
    ORT defaults                 132.3 ms

Ten times slower, and the wall-clock figure understates it — the default pool
sizes itself to the core count and spin-waits between runs, so it also burns
whole cores while the satellite sits idle. This is the same pathology
`ncpu=1` buys off for openWakeWord (see wakeword_oww), but the stakes are
higher: livekit issues eighteen `session.run` calls per evaluation where
openWakeWord issues three, each one another chance to wake a spinning worker
that then contends with faster-whisper on a four-core Pi.

Fails open throughout: an unbindable seam logs and leaves stock behaviour.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from importlib import metadata

log = logging.getLogger(__name__)

# the release whose three construction sites this wrapper was checked against;
# keep in sync with the livekit-wakeword pin in deploy/install.md
TESTED_VERSION = "0.2.1"

_sessions_bound = 0


def _session_options(ort):
    """One thread, no spinning: the options every ORT session here gets.

    ORT's default sizes the intra-op pool to the core count and those workers
    spin-wait between runs. At an 80 ms inference cadence they never park, so
    the pool burns whole cores while the satellite sits idle.
    """
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


def warn_on_untested_version() -> None:
    """Warn when the installed release is not the one this was checked against.

    Read from the package metadata, never from `livekit.wakeword.__version__`:
    that attribute still says "0.1.0" in the 0.2.1 release and would silently
    pass any comparison.
    """
    try:
        installed = metadata.version("livekit-wakeword")
    except Exception as exc:  # noqa: BLE001 - diagnostics must not break startup
        log.warning("livekit-wakeword version unreadable (%s)", exc)
        return
    if installed != TESTED_VERSION:
        log.warning(
            "livekit-wakeword %s installed, %s tested: re-check that its ONNX "
            "sessions are still built without options (models/feature_extractor.py, "
            "inference/model.py) — an escaped session spins a thread pool per core",
            installed,
            TESTED_VERSION,
        )


@contextmanager
def single_threaded_sessions():
    """Force our SessionOptions onto every ORT session built in this block.

    A global override, so it is scoped to detector construction and restored
    in `finally`, leaving faster-whisper, piper and every other ORT user in
    the process on their own settings. A caller that passes its own options
    keeps them; only the bare calls livekit makes are rewritten.
    """
    try:
        import onnxruntime as ort
    except Exception as exc:  # noqa: BLE001 - fail open
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
            log.info("livekit bound %d ONNX sessions, no extra OS threads", bound)
        else:
            log.warning(
                "livekit bound %d ONNX sessions but the process gained %d OS "
                "threads; ORT sized a pool anyway and those workers spin between "
                "evaluations — expect the satellite to burn cores while idle",
                bound,
                leaked,
            )
