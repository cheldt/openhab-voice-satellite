import io
import wave

import numpy as np

from openhab_voice_satellite.audio.wav import pcm_to_wav_bytes, read_wav_mono, write_wav


def test_pcm_to_wav_roundtrip():
    pcm = np.arange(-100, 100, dtype=np.int16)
    data = pcm_to_wav_bytes(pcm, 16000)
    with wave.open(io.BytesIO(data), "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        decoded = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    assert np.array_equal(decoded, pcm)


def test_write_read_roundtrip(tmp_path):
    pcm = np.arange(-50, 50, dtype=np.int16)
    path = tmp_path / "roundtrip.wav"
    write_wav(path, pcm, 22050)
    decoded, rate = read_wav_mono(path)
    assert rate == 22050
    assert np.array_equal(decoded, pcm)


def test_read_wav_mono_downmixes_multichannel(tmp_path):
    left = np.arange(0, 100, dtype=np.int16)
    right = np.full(100, -1, dtype=np.int16)
    stereo = np.column_stack([left, right]).ravel()
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(stereo.tobytes())
    decoded, rate = read_wav_mono(path)
    assert rate == 16000
    assert np.array_equal(decoded, left)  # channel 0 only
