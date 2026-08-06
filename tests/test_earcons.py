import wave

import numpy as np

from openhab_voice_satellite.audio.earcons import Earcons
from openhab_voice_satellite.config import EarconsConfig

from .fakes import BufferAudioSink


def _write_wav(path, samples=160, rate=16000):
    pcm = np.zeros(samples, dtype=np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())


def _make_earcons(tmp_path, sink, **config_kwargs) -> Earcons:
    (tmp_path / "sounds").mkdir(exist_ok=True)
    for name in ("wake", "ack", "error", "idle"):
        _write_wav(tmp_path / "sounds" / f"{name}.wav")
    return Earcons(EarconsConfig(**config_kwargs), sink, base_dir=tmp_path)


async def test_idle_earcon_loads_and_plays(tmp_path):
    sink = BufferAudioSink()
    earcons = _make_earcons(tmp_path, sink)

    await earcons.play("idle")

    assert len(sink.played) == 1
    pcm, rate = sink.played[0]
    assert rate == 16000
    assert pcm.dtype == np.int16


async def test_disabled_plays_nothing(tmp_path):
    sink = BufferAudioSink()
    earcons = _make_earcons(tmp_path, sink, enabled=False)

    await earcons.play("idle")

    assert sink.played == []


async def test_unknown_name_is_noop(tmp_path):
    sink = BufferAudioSink()
    earcons = _make_earcons(tmp_path, sink)

    await earcons.play("does-not-exist")

    assert sink.played == []


async def test_playback_failure_is_swallowed(tmp_path, caplog):
    # a dead speaker must not kill the pipeline
    class BrokenSink(BufferAudioSink):
        async def play(self, pcm, sample_rate):
            raise RuntimeError("playback pipeline error")

    earcons = _make_earcons(tmp_path, BrokenSink())

    await earcons.play("idle")  # must not raise

    assert "earcon playback failed" in caplog.text
