"""
===============================================================================
Unified Feature Extraction Module (src/preprocessing/features.py)
===============================================================================

Computes both 2D Log-Mel Spectrogram and 1D NMF Event Activation Envelope
from a SINGLE STFT computation pass to guarantee identical frame grids.
"""

from typing import Tuple, Dict, Any
from pathlib import Path
import numpy as np
import librosa

import config
from src.preprocessing.stft import compute_stft
from src.preprocessing.log_compression import log_compression
from src.nmf.nnls import solve_nnls_batch
from src.nmf.activation import compute_h_event_sum
from src.audio.framing import frame_to_time


def extract_features_from_audio(
    audio: np.ndarray,
    w_standard: np.ndarray,
    sr: int = config.SAMPLE_RATE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract Log-Mel and H_event_sum from a single STFT matrix.

    Parameters
    ----------
    audio : np.ndarray
        1D float32 audio waveform.
    w_standard : np.ndarray
        Fixed NMF basis matrix of shape (1025, 48).
    sr : int
        Audio sampling rate.

    Returns
    -------
    log_mel : np.ndarray
        2D array of shape (N_mels, T) containing log-magnitude Mel spectrogram.
    h_event_sum : np.ndarray
        1D array of shape (T,) containing the part impact activation envelope.
    frame_times : np.ndarray
        1D array of timestamps corresponding to each frame.
    """
    # 1. Thực hiện STFT duy nhất
    stft_matrix = compute_stft(audio)  # (1025, T)
    total_frames = stft_matrix.shape[1]

    # 2. Nhánh NMF -> H_event_sum
    v_matrix = log_compression(stft_matrix)
    h_matrix = solve_nnls_batch(v_matrix, w_standard)
    h_event_sum = compute_h_event_sum(h_matrix)  # (T,)

    # 3. Nhánh Mel Filterbank -> Log-Mel (từ cùng stft_matrix)
    magnitude = np.abs(stft_matrix)
    mel_basis = librosa.filters.mel(
        sr=sr,
        n_fft=config.N_FFT,
        n_mels=config.N_MELS,
        fmin=config.F_MIN,
        fmax=config.F_MAX
    )
    mel_spectrogram = np.dot(mel_basis, magnitude)
    log_mel = np.log1p(mel_spectrogram).astype(np.float32)  # (N_mels, T)

    # 4. Trục thời gian đồng bộ
    frame_times = frame_to_time(np.arange(total_frames), hop_length=config.HOP_LENGTH, sr=sr)

    return log_mel, h_event_sum.astype(np.float32), frame_times.astype(np.float32)


def save_train_features(
    output_path: Path,
    filename: str,
    log_mel: np.ndarray,
    h_event_sum: np.ndarray,
    frame_times: np.ndarray,
    true_count: int
):
    """Lưu trữ persistent các đặc trưng đã trích xuất ra file .npz."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output_path),
        filename=filename,
        log_mel=log_mel,
        h_event_sum=h_event_sum,
        frame_times=frame_times,
        true_count=true_count
    )