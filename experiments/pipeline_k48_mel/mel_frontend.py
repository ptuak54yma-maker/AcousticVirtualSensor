"""STFT -> Mel filterbank -> nonnegative dynamic-range compression."""

from __future__ import annotations

from typing import Literal

import librosa
import numpy as np
from scipy.signal import get_window

try:
    from . import experiment_config as config
except ImportError:  # also support direct execution from this folder
    import experiment_config as config

Compressor = Literal["log1p", "power", "linear"]


def make_mel_filterbank(
    sample_rate: int = config.SAMPLE_RATE,
    n_fft: int = config.N_FFT,
    n_mels: int = config.N_MELS,
    f_min: float = config.F_MIN,
    f_max: float = config.F_MAX,
) -> np.ndarray:
    """Return a nonnegative Slaney-normalized matrix shaped (n_mels, F)."""
    if not (0.0 <= f_min < f_max <= sample_rate / 2):
        raise ValueError("Require 0 <= f_min < f_max <= Nyquist.")
    bank = librosa.filters.mel(
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
        htk=config.MEL_HTK,
        norm=config.MEL_NORM,
        dtype=np.float32,
    )
    if not np.isfinite(bank).all() or np.any(bank < 0):
        raise ValueError("Mel filterbank must be finite and nonnegative.")
    return np.ascontiguousarray(bank, dtype=np.float32)


def stft_magnitude(audio: np.ndarray) -> np.ndarray:
    """Compute a centered=False magnitude STFT shaped (F, frames)."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono 1D audio, got shape {audio.shape}.")
    if audio.size < config.N_FFT:
        return np.empty((config.N_FFT // 2 + 1, 0), dtype=np.float32)

    # Match the current project's audio loader: remove the file-level DC bias.
    audio = audio - np.float32(np.mean(audio, dtype=np.float64))

    window = get_window(config.WINDOW, config.N_FFT, fftbins=True).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(audio, config.N_FFT)[::config.HOP_LENGTH]
    spec = np.fft.rfft(frames * window[None, :], n=config.N_FFT, axis=1)
    return np.ascontiguousarray(np.abs(spec).T, dtype=np.float32)


def mel_spectrogram(
    audio: np.ndarray,
    mel_bank: np.ndarray | None = None,
    mel_input: Literal["magnitude", "power"] = config.MEL_INPUT,
) -> np.ndarray:
    """Project magnitude or power STFT onto the Mel bank; output is nonnegative."""
    if mel_input not in ("magnitude", "power"):
        raise ValueError("mel_input must be 'magnitude' or 'power'.")
    magnitude = stft_magnitude(audio)
    linear_spec = magnitude if mel_input == "magnitude" else np.square(magnitude)
    bank = make_mel_filterbank() if mel_bank is None else np.asarray(mel_bank, dtype=np.float32)
    if bank.ndim != 2 or bank.shape[1] != linear_spec.shape[0]:
        raise ValueError(f"Mel bank shape {bank.shape} is incompatible with {linear_spec.shape}.")
    return np.ascontiguousarray(bank @ linear_spec, dtype=np.float32)


def compress_mel(
    mel: np.ndarray,
    reference: float,
    compressor: Compressor = config.COMPRESSOR,
    gamma: float = config.POWER_GAMMA,
) -> np.ndarray:
    """Compress a nonnegative Mel spectrogram while keeping it NMF-compatible."""
    mel = np.asarray(mel, dtype=np.float32)
    if mel.ndim != 2 or not np.isfinite(mel).all() or np.any(mel < 0):
        raise ValueError("Mel input must be a finite, nonnegative 2D matrix.")
    if compressor not in ("log1p", "power", "linear"):
        raise ValueError(f"Unsupported compressor: {compressor}")
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("reference must be finite and positive.")

    scaled = mel / np.float32(reference)
    if compressor == "log1p":
        result = np.log1p(scaled)
    elif compressor == "power":
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1].")
        result = np.power(scaled, np.float32(gamma))
    else:
        result = scaled
    return np.ascontiguousarray(result, dtype=np.float32)


def fit_reference(mel_specs: list[np.ndarray]) -> float:
    """Estimate one robust global scale from training data (median file P95)."""
    if not mel_specs:
        raise ValueError("At least one training Mel spectrogram is required.")
    p95 = [float(np.percentile(m[ m > 0 ], 95)) for m in mel_specs if np.any(m > 0)]
    if not p95:
        raise ValueError("Training Mel spectrograms contain no positive values.")
    reference = float(np.median(np.asarray(p95, dtype=np.float64)))
    return max(reference, float(np.finfo(np.float32).tiny))
