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


async def test_idle_earcon_loads_and_plays(tmp_path):
    (tmp_path / "sounds").mkdir()
    for name in ("wake", "ack", "error", "idle"):
        _write_wav(tmp_path / "sounds" / f"{name}.wav")
    sink = BufferAudioSink()
    earcons = Earcons(EarconsConfig(), sink, base_dir=tmp_path)

    await earcons.play("idle")

    assert len(sink.played) == 1
    pcm, rate = sink.played[0]
    assert rate == 16000
    assert pcm.dtype == np.int16


async def test_unknown_name_is_noop(tmp_path):
    sink = BufferAudioSink()
    earcons = Earcons(EarconsConfig(enabled=False), sink, base_dir=tmp_path)

    await earcons.play("idle")

    assert sink.played == []
