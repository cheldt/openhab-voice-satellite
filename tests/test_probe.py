"""Field diagnostic: verdict-driven marks, gated columns, exit codes."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import numpy as np

from openhab_voice_satellite import probe
from openhab_voice_satellite.audio import io as audio_io_module
from openhab_voice_satellite.audio.gst_source import CaptureStats

from .fakes import BufferAudioSink, ScriptedDetector
from .wakeword_stubs import make_config

FRAME = 1280  # 80 ms at 16 kHz

STAGE2 = {"model": "v.onnx", "mel_basis": "m.npy"}


class FakeClock:
    """Deterministic clock behind the probe's per-second row pacing."""

    def __init__(self) -> None:
        self.t = 0.0

    def monotonic(self) -> float:
        return self.t

    perf_counter = monotonic
    process_time = monotonic


class TickingSource:
    """Yields silence frames, advancing the fake clock per frame.

    At 0.6 s per frame every second row completes a probe second, so a
    handful of frames covers several printed rows without real sleeping.
    """

    target = None

    def __init__(self, clock: FakeClock, frames: int, tick: float = 0.6) -> None:
        self._clock = clock
        self._n = frames
        self._tick = tick

    async def frames(self):
        for _ in range(self._n):
            self._clock.t += self._tick
            await asyncio.sleep(0)
            yield np.zeros(FRAME, dtype=np.int16)

    def stats(self) -> CaptureStats:
        return CaptureStats()


class StalledSource:
    """Never yields a frame — the stalled/unlinked capture stream."""

    target = None

    async def frames(self):
        await asyncio.sleep(3600)
        yield np.zeros(FRAME, dtype=np.int16)  # pragma: no cover

    def stats(self) -> CaptureStats:
        return CaptureStats()


def _wire(monkeypatch, tmp_path, detector, source, run_s: int = 3):
    """Point the probe at fakes: audio pair, detector, node listing, clock."""
    sink = BufferAudioSink()
    sink.target = None

    @asynccontextmanager
    async def fake_audio_io(audio_config):
        yield source, sink

    async def fake_verify_links(source_target, sink_target):
        pass

    monkeypatch.chdir(tmp_path)  # diagnose_capture.wav lands here
    monkeypatch.setattr(audio_io_module, "audio_io", fake_audio_io)
    monkeypatch.setattr(audio_io_module, "verify_links", fake_verify_links)
    monkeypatch.setattr(probe, "build_detector", lambda config: detector)
    monkeypatch.setattr(probe, "_print_sources", lambda config: True)
    monkeypatch.setattr(probe, "RUN_S", run_s)
    return sink


def test_marks_the_row_from_the_verdict_and_exits_zero(
    tmp_path, monkeypatch, capsys
):
    clock = FakeClock()
    monkeypatch.setattr(probe, "time", clock)
    detector = ScriptedDetector(
        detections={3: "wake"}, scores={i: 0.2 for i in range(10)}
    )
    _wire(monkeypatch, tmp_path, detector, TickingSource(clock, frames=10))
    assert probe.probe_mic(make_config()) == 0
    out = capsys.readouterr().out
    assert "<-- WAKE" in out
    assert "wake events: 1" in out
    # no stop model, no stage 2: neither column exists — score("stop") would
    # silently repeat the wake score
    assert "stop_score" not in out
    assert "verifier" not in out
    assert (tmp_path / "diagnose_capture.wav").exists()


def test_a_high_stage1_score_alone_no_longer_marks_wake(
    tmp_path, monkeypatch, capsys
):
    # the shipped two-stage config runs stage 1 deliberately low, so almost
    # any speech crosses it; a score-derived mark would say WAKE on rows the
    # running app rejected. The mark has to come from process()'s verdict.
    clock = FakeClock()
    monkeypatch.setattr(probe, "time", clock)
    detector = ScriptedDetector(
        scores={i: 0.95 for i in range(10)},
        rejections={4: 0.95},
        verifier_scores={4: 0.12},
    )
    _wire(monkeypatch, tmp_path, detector, TickingSource(clock, frames=10))
    assert probe.probe_mic(make_config(stage2=STAGE2)) == 0
    out = capsys.readouterr().out
    assert "<-- WAKE" not in out
    assert "<-- rejected by stage 2" in out
    assert "verifier" in out
    assert "0.120" in out  # the verdict's score, shown on its row
    assert "rejected by stage 2: 1" in out
    assert "wake events: 0" in out


def test_stop_column_and_events_only_with_a_stop_model(
    tmp_path, monkeypatch, capsys
):
    clock = FakeClock()
    monkeypatch.setattr(probe, "time", clock)
    detector = ScriptedDetector(
        detections={3: "stop"}, scores={i: 0.4 for i in range(10)}
    )
    _wire(monkeypatch, tmp_path, detector, TickingSource(clock, frames=10))
    assert probe.probe_mic(make_config(stop_model="stop")) == 0
    out = capsys.readouterr().out
    assert "stop_score" in out
    assert "<-- STOP" in out
    assert "stop events: 1" in out


def test_a_stalled_capture_exits_nonzero(tmp_path, monkeypatch, capsys):
    _wire(monkeypatch, tmp_path, ScriptedDetector(), StalledSource(), run_s=0)
    monkeypatch.setattr(probe, "STALL_GRACE_S", 0.2)
    assert probe.probe_mic(make_config()) == 1
    assert "TIMED OUT" in capsys.readouterr().out


def test_fails_cleanly_when_pipewire_nodes_cannot_be_listed(
    tmp_path, monkeypatch, capsys
):
    # same failure --list-devices already handles: no PipeWire / no GStreamer
    # pipewire plugin must print the install hint, not a raw traceback
    from openhab_voice_satellite.audio import gst_devices

    def boom():
        raise RuntimeError("failed to start GStreamer device monitor")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gst_devices, "list_audio_nodes", boom)
    assert probe.probe_mic(make_config()) == 1
    out = capsys.readouterr().out
    assert "cannot list PipeWire nodes" in out
    assert "deploy/install.md" in out
