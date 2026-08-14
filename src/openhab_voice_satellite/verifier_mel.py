"""Mel-PCEN frontend for the violawake stage-2 verifier, numpy/scipy only.

The verifier model (violawakeword repo, scripts/train_verifier.py) was trained
on features from ``violawake_sdk.audio.compute_features``, which runs through
librosa. This box deliberately has no librosa or torch, so this module
replicates that pipeline byte-for-byte from numpy + scipy:

  float32 [-1,1] audio, 24000 samples (1.5 s @ 16 kHz)
  -> STFT: n_fft 512, hop 160, win 400 (hann, fftbins), center=True,
     constant (zero) padding — librosa 0.11 defaults
  -> power spectrogram |S|^2
  -> mel projection with a PRECOMPUTED filterbank matrix (shipped as .npy
     next to the verifier model; exported from librosa.filters.mel with
     sr=16000, n_fft=512, n_mels=40, fmin=60, fmax=7800, slaney norm).
     The matrix is data, not code — shipping it removes the one part of
     librosa that is genuinely fiddly to reimplement.
  -> PCEN exactly as librosa.pcen(mel * 2**31, gain=0.98, bias=2.0,
     power=0.5, time_constant=0.06, eps=1e-6): first-order IIR smoother via
     scipy.signal.lfilter seeded with lfilter_zi.

Output shape (40, 151). Parity against the SDK is asserted by
tests/test_verifier_mel.py over golden fixtures generated where librosa
exists (violawakeword repo, scripts/export_verifier_frontend.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.signal

SAMPLE_RATE = 16000
CLIP_SAMPLES = 24000
N_FFT = 512
HOP = 160
WIN = 400
PCEN_GAIN = 0.98
PCEN_BIAS = 2.0
PCEN_POWER = 0.5
PCEN_TIME_CONSTANT = 0.06
PCEN_EPS = 1e-6

_window = scipy.signal.get_window("hann", WIN, fftbins=True)
# librosa centers a short window inside the FFT frame
_pad_left = (N_FFT - WIN) // 2
_window_padded = np.pad(_window, (_pad_left, N_FFT - WIN - _pad_left))


def _stft_power(audio: np.ndarray) -> np.ndarray:
    """|STFT|^2 with librosa's center/constant-pad conventions: (257, T)."""
    y = np.pad(audio.astype(np.float64), (N_FFT // 2, N_FFT // 2))
    n_frames = 1 + (len(y) - N_FFT) // HOP
    frames = np.lib.stride_tricks.as_strided(
        y,
        shape=(n_frames, N_FFT),
        strides=(y.strides[0] * HOP, y.strides[0]),
    )
    spec = np.fft.rfft(frames * _window_padded, axis=1)
    return (spec.real**2 + spec.imag**2).T


def _pcen(S: np.ndarray) -> np.ndarray:
    """librosa.pcen with the SDK's parameters; S is mel power * 2**31."""
    t_frames = PCEN_TIME_CONSTANT * SAMPLE_RATE / HOP
    b = (np.sqrt(1 + 4 * t_frames**2) - 1) / (2 * t_frames**2)
    zi = np.empty((1, 1))
    zi[:] = scipy.signal.lfilter_zi([b], [1, b - 1])[:]
    S_smooth, _ = scipy.signal.lfilter([b], [1, b - 1], S, zi=zi, axis=-1)
    smooth = np.exp(-PCEN_GAIN * (np.log(PCEN_EPS) + np.log1p(S_smooth / PCEN_EPS)))
    return (PCEN_BIAS**PCEN_POWER) * np.expm1(
        PCEN_POWER * np.log1p(S * smooth / PCEN_BIAS)
    )


class MelPcenFrontend:
    """Callable turning a 1.5 s float32 clip into the verifier's (40, 151)."""

    def __init__(self, mel_basis_path: str | Path):
        self._mel_basis = np.load(mel_basis_path)
        if self._mel_basis.shape != (40, N_FFT // 2 + 1):
            raise ValueError(
                f"mel basis {mel_basis_path}: expected (40, {N_FFT // 2 + 1}), "
                f"got {self._mel_basis.shape}"
            )

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) != CLIP_SAMPLES:
            raise ValueError(f"expected {CLIP_SAMPLES} samples, got {len(audio)}")
        mel = self._mel_basis @ _stft_power(audio)
        return _pcen(mel * (2**31)).astype(np.float32)
