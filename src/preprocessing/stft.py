"""
===============================================================================
Short-Time Fourier Transform (STFT) Module
===============================================================================

Computes one-dimensional and batch Short-Time Fourier Transform (STFT).
Uses librosa/scipy backend strictly adhering to project configuration parameters:
- Sampling rate: 44.1 kHz
- N_FFT: 2048
- Window length: 2048
- Hop length: 1024
- Window: Hann
- Center: False (for exact sample alignment)

Does NOT perform log-compression or NMF factorization.
"""

import numpy as np
import scipy.signal
from typing import Optional

import config


def compute_stft(
    y: np.ndarray,
    n_fft: int = config.N_FFT,
    hop_length: int = config.HOP_LENGTH,
    win_length: int = config.WINDOW_LENGTH,
    window: str = config.WINDOW_FUNCTION,
    center: bool = config.CENTER_STFT
) -> np.ndarray:
    """
    Compute the Short-Time Fourier Transform (STFT) of a 1D audio signal.

    Parameters
    ----------
    y : np.ndarray
        1D float32 audio array.
    n_fft : int, optional
        FFT point size. Defaults to 2048.
    hop_length : int, optional
        Hop length in samples. Defaults to 1024.
    win_length : int, optional
        Physical window length in samples. Defaults to 2048.
    window : str, optional
        Window type ('hann'). Defaults to config.WINDOW_FUNCTION.
    center : bool, optional
        Whether to pad the signal so that frame t is centered. Defaults to False.

    Returns
    -------
    D : np.ndarray (complex64)
        Complex STFT matrix with shape (1 + n_fft // 2, T) = (1025, T).

    Raises
    ------
    ValueError
        If the input audio is shorter than n_fft when center=False.
    """
    if y.ndim != 1:
        raise ValueError(f"Expected 1D audio array, got shape {y.shape}")

    if not center and len(y) < n_fft:
        raise ValueError(
            f"Audio length ({len(y)} samples) is shorter than N_FFT ({n_fft} samples) "
            f"with center=False."
        )

    if center:
        y = np.pad(y, (n_fft // 2, n_fft // 2), mode="constant")

    if len(y) < n_fft:
        return np.empty((1 + n_fft // 2, 0), dtype=np.complex64)

    fft_window = scipy.signal.get_window(window, win_length, fftbins=True).astype(np.float32)
    if win_length < n_fft:
        left = (n_fft - win_length) // 2
        right = n_fft - win_length - left
        fft_window = np.pad(fft_window, (left, right), mode="constant")
    elif win_length > n_fft:
        raise ValueError(f"WINDOW_LENGTH ({win_length}) cannot exceed N_FFT ({n_fft}).")

    frames = np.lib.stride_tricks.sliding_window_view(y, n_fft)[::hop_length]
    windowed_frames = frames.astype(np.float32, copy=False) * fft_window
    stft_matrix = np.fft.rfft(windowed_frames, n=n_fft, axis=1).T

    return np.ascontiguousarray(stft_matrix, dtype=np.complex64)


def compute_frame_stft(
    frame: np.ndarray,
    n_fft: int = config.N_FFT,
    window: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute a single-frame Real FFT (RFFT) for real-time streaming buffers.
    Optimized for single-frame inference without overhead from full librosa STFT.

    Parameters
    ----------
    frame : np.ndarray
        1D audio array of length exactly equal to n_fft (2048 samples).
    n_fft : int, optional
        FFT size. Defaults to 2048.
    window : np.ndarray, optional
        Pre-computed Hann window array of length n_fft.

    Returns
    -------
    spectrum : np.ndarray (complex64)
        Single-column complex spectrum vector of shape (1025, 1).
    """
    if len(frame) != n_fft:
        raise ValueError(f"Expected frame length {n_fft}, got {len(frame)}")

    if window is None:
        window = scipy.signal.windows.hann(n_fft, sym=False).astype(np.float32)

    # Windowed signal
    windowed_frame = frame * window

    # Fast One-Dimensional Discrete Fourier Transform for Real Input
    fft_result = np.fft.rfft(windowed_frame, n=n_fft)

    return fft_result[:, np.newaxis]
