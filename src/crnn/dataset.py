"""
===============================================================================
Hybrid CRNN Dataset & Window Slicing Module (Log-Mel + H_event_sum)
===============================================================================

Handles multimodal sliding-window slicing and provides a PyTorch Dataset wrapper
that outputs both Log-Mel Spectrogram (2D) and H_event_sum (1D) per sample.
"""

from typing import Tuple, List, Union, Dict
import numpy as np
import torch
from torch.utils.data import Dataset

import config


def extract_hybrid_windows(
    log_mel_seq: np.ndarray,
    h_event_sum_seq: np.ndarray,
    label_seq: np.ndarray,
    window_length: int = config.SEQUENCE_LENGTH,
    hop_size: int = config.SEQUENCE_HOP
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Slice continuous Log-Mel Spectrogram, H_event_sum, and target labels into
    synchronized, fixed-length sliding windows.

    Parameters
    ----------
    log_mel_seq : np.ndarray
        2D Log-Mel Spectrogram of shape (N_mels, T), e.g., (128, T).
    h_event_sum_seq : np.ndarray
        1D or 2D activation envelope of shape (T,) or (1, T).
    label_seq : np.ndarray
        Target frame-level labels of shape (T,).
    window_length : int, optional
        Length of window L in frames. Defaults to config.SEQUENCE_LENGTH (128).
    hop_size : int, optional
        Step size between windows. Defaults to config.SEQUENCE_HOP (64).

    Returns
    -------
    X_mel : np.ndarray (float32)
        Array of Mel windows with shape (N_samples, 1, N_mels, window_length).
    X_he : np.ndarray (float32)
        Array of H_event_sum windows with shape (N_samples, 1, window_length).
    Y : np.ndarray (float32)
        Array of label windows with shape (N_samples, window_length).
    """
    total_frames = len(label_seq)

    # Validate and standardize H_event_sum to (1, T)
    if h_event_sum_seq.ndim == 1:
        h_event_sum_seq = h_event_sum_seq[np.newaxis, :]

    if log_mel_seq.shape[1] != total_frames or h_event_sum_seq.shape[1] != total_frames:
        raise ValueError(
            f"Temporal frame count mismatch: Mel ({log_mel_seq.shape[1]}), "
            f"H_event ({h_event_sum_seq.shape[1]}), Labels ({total_frames})"
        )

    # Zero-padding if file is shorter than one window
    if total_frames < window_length:
        pad_len = window_length - total_frames
        log_mel_seq = np.pad(log_mel_seq, ((0, 0), (0, pad_len)), mode="constant", constant_values=0)
        h_event_sum_seq = np.pad(h_event_sum_seq, ((0, 0), (0, pad_len)), mode="constant", constant_values=0)
        label_seq = np.pad(label_seq, (0, pad_len), mode="constant", constant_values=0)
        total_frames = window_length

    mel_list: List[np.ndarray] = []
    he_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    for start_idx in range(0, total_frames - window_length + 1, hop_size):
        end_idx = start_idx + window_length
        
        # Slicing
        mel_win = log_mel_seq[:, start_idx:end_idx]      # (N_mels, L)
        he_win = h_event_sum_seq[:, start_idx:end_idx]    # (1, L)
        y_win = label_seq[start_idx:end_idx]             # (L,)

        # Add channel dim for 2D Conv: (1, N_mels, L)
        mel_list.append(mel_win[np.newaxis, :, :])
        he_list.append(he_win)
        y_list.append(y_win)

    if not mel_list:
        return (
            np.empty((0, 1, config.N_MELS, window_length), dtype=np.float32),
            np.empty((0, 1, window_length), dtype=np.float32),
            np.empty((0, window_length), dtype=np.float32)
        )

    X_mel = np.stack(mel_list, axis=0).astype(np.float32)
    X_he = np.stack(he_list, axis=0).astype(np.float32)
    Y = np.stack(y_list, axis=0).astype(np.float32)

    return X_mel, X_he, Y


class HybridCRNNDataset(Dataset):
    """
    PyTorch Dataset returning both Log-Mel and H_event_sum features.
    """

    def __init__(
        self,
        x_mel: Union[np.ndarray, torch.Tensor],
        x_he: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor]
    ):
        self.x_mel = torch.from_numpy(x_mel).float() if isinstance(x_mel, np.ndarray) else x_mel.float()
        self.x_he = torch.from_numpy(x_he).float() if isinstance(x_he, np.ndarray) else x_he.float()
        self.labels = torch.from_numpy(labels).float() if isinstance(labels, np.ndarray) else labels.float()

        if not (len(self.x_mel) == len(self.x_he) == len(self.labels)):
            raise ValueError("Sample count mismatch among features and labels.")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_mel[idx], self.x_he[idx], self.labels[idx]