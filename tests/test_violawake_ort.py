"""The ORT thread patch: violawake ships no session options of its own."""

from __future__ import annotations

import sys
import types

import pytest

from openhab_voice_satellite import violawake_ort
from openhab_voice_satellite.violawake_ort import (
    patch_onnx_threads,
    sessions_bound,
    single_threaded_sessions,
)


class RecordingSession:
    calls: list[dict] = []

    def __init__(self, path, sess_options=None, providers=None):
        RecordingSession.calls.append(
            {"path": path, "options": sess_options, "providers": providers}
        )


def install_fake_backend(monkeypatch) -> types.ModuleType:
    """A stand-in for violawake_sdk.backends.onnx_backend with the real seams."""

    class ModelLoadError(Exception):
        pass

    class OnnxSession:
        def __init__(self, session):
            self.session = session

    class OnnxBackend:
        def __init__(self, providers=None):
            self._providers = providers or ["CPUExecutionProvider"]

        def load(self, model_path, **kwargs):
            raise AssertionError("stock load() ran: the patch did not take")

    module = types.ModuleType("violawake_sdk.backends.onnx_backend")
    module.OnnxBackend = OnnxBackend
    module.OnnxSession = OnnxSession
    module.ModelLoadError = ModelLoadError
    backends = types.ModuleType("violawake_sdk.backends")
    backends.onnx_backend = module
    package = types.ModuleType("violawake_sdk")
    package.backends = backends
    monkeypatch.setitem(sys.modules, "violawake_sdk", package)
    monkeypatch.setitem(sys.modules, "violawake_sdk.backends", backends)
    monkeypatch.setitem(sys.modules, "violawake_sdk.backends.onnx_backend", module)
    return module


@pytest.fixture
def backend(monkeypatch, tmp_path):
    import onnxruntime as ort

    RecordingSession.calls = []
    monkeypatch.setattr(ort, "InferenceSession", RecordingSession)
    module = install_fake_backend(monkeypatch)
    model = tmp_path / "model.onnx"
    model.write_bytes(b"")
    return module, model


def test_sessions_are_single_threaded_and_do_not_spin(backend):
    module, model = backend
    assert patch_onnx_threads() is True
    module.OnnxBackend().load(model)

    options = RecordingSession.calls[0]["options"]
    assert options is not None, "no SessionOptions passed — ORT would size to core count"
    assert options.intra_op_num_threads == 1
    assert options.inter_op_num_threads == 1


def test_providers_still_reach_the_session(backend):
    module, model = backend
    patch_onnx_threads()
    module.OnnxBackend(providers=["CPUExecutionProvider"]).load(model)
    assert RecordingSession.calls[0]["providers"] == ["CPUExecutionProvider"]
    # an explicit kwarg still wins, as it does upstream
    module.OnnxBackend().load(model, providers=["OtherProvider"])
    assert RecordingSession.calls[1]["providers"] == ["OtherProvider"]


def test_patch_is_idempotent(backend):
    module, model = backend
    assert patch_onnx_threads() is True
    once = module.OnnxBackend.load
    assert patch_onnx_threads() is True
    assert module.OnnxBackend.load is once  # not wrapped a second time


def test_missing_model_reports_the_path(backend):
    module, model = backend
    patch_onnx_threads()
    with pytest.raises(FileNotFoundError, match="absent.onnx"):
        module.OnnxBackend().load(model.parent / "absent.onnx")


def test_load_failure_becomes_the_sdk_error(backend, monkeypatch):
    import onnxruntime as ort

    module, model = backend
    patch_onnx_threads()
    monkeypatch.setattr(ort, "InferenceSession", _boom)
    with pytest.raises(module.ModelLoadError, match="bad graph"):
        module.OnnxBackend().load(model)


def test_a_bound_session_is_counted(backend):
    """"Patch installed" and "patch used" have to be separate facts."""
    module, model = backend
    patch_onnx_threads()
    before = sessions_bound()
    module.OnnxBackend().load(model)
    assert sessions_bound() == before + 1


def test_a_failed_load_is_not_counted(backend, monkeypatch):
    import onnxruntime as ort

    module, model = backend
    patch_onnx_threads()
    monkeypatch.setattr(ort, "InferenceSession", _boom)
    before = sessions_bound()
    with pytest.raises(module.ModelLoadError):
        module.OnnxBackend().load(model)
    assert sessions_bound() == before


def test_the_block_binds_sessions_built_off_the_seam(backend):
    """A release that stops using OnnxBackend.load must still be covered."""
    import onnxruntime as ort

    module, model = backend
    before = sessions_bound()
    with single_threaded_sessions():
        ort.InferenceSession(str(model))  # as stock violawake builds them

    options = RecordingSession.calls[0]["options"]
    assert options is not None
    assert options.intra_op_num_threads == 1
    assert options.inter_op_num_threads == 1
    assert sessions_bound() == before + 1


def test_the_block_leaves_explicit_options_alone(backend):
    import onnxruntime as ort

    module, model = backend
    sentinel = object()
    before = sessions_bound()
    with single_threaded_sessions():
        ort.InferenceSession(str(model), sess_options=sentinel)
    assert RecordingSession.calls[0]["options"] is sentinel
    assert sessions_bound() == before  # ours to count, not everyone's


def test_the_block_is_scoped_to_itself(backend):
    """faster-whisper and piper build ORT sessions too; leave them alone."""
    import onnxruntime as ort

    module, model = backend
    with single_threaded_sessions():
        pass
    assert ort.InferenceSession is RecordingSession
    ort.InferenceSession(str(model))
    assert RecordingSession.calls[0]["options"] is None


def test_the_block_restores_the_session_class_after_a_failure(backend):
    import onnxruntime as ort

    with pytest.raises(RuntimeError, match="model load blew up"):
        with single_threaded_sessions():
            raise RuntimeError("model load blew up")
    assert ort.InferenceSession is RecordingSession


def test_leaked_threads_are_reported(backend, monkeypatch, caplog):
    """The failure this module exists to prevent must not be silent."""
    counts = iter([12, 21])
    monkeypatch.setattr(violawake_ort, "_thread_count", lambda: next(counts))
    with caplog.at_level("WARNING"), single_threaded_sessions():
        pass
    assert "gained 9 OS threads" in caplog.text
    assert "burn cores" in caplog.text


def test_a_clean_block_reports_no_extra_threads(backend, monkeypatch, caplog):
    counts = iter([12, 12])
    monkeypatch.setattr(violawake_ort, "_thread_count", lambda: next(counts))
    with caplog.at_level("INFO"), single_threaded_sessions():
        pass
    assert "no extra OS threads" in caplog.text


def test_an_untested_version_warns(backend, monkeypatch, caplog):
    import importlib.metadata

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.3.0")
    with caplog.at_level("WARNING"):
        assert patch_onnx_threads() is True  # a warning, never a refusal
    assert "0.3.0" in caplog.text
    assert violawake_ort.TESTED_VERSION in caplog.text


def _boom(*args, **kwargs):
    raise RuntimeError("bad graph")


def test_fails_open_when_the_seam_moved(monkeypatch, caplog):
    # a violawake release that restructured its backends must not stop startup
    monkeypatch.setitem(sys.modules, "violawake_sdk", types.ModuleType("violawake_sdk"))
    monkeypatch.delitem(sys.modules, "violawake_sdk.backends", raising=False)
    monkeypatch.delitem(sys.modules, "violawake_sdk.backends.onnx_backend", raising=False)
    with caplog.at_level("WARNING"):
        assert patch_onnx_threads() is False
    assert "thread patch skipped" in caplog.text
    assert "burn cores" in caplog.text  # the warning has to state the cost
