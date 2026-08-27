"""
===============================================================================
Audio Framing Utility Module
===============================================================================

Provides mathematical conversion functions between raw audio samples, 
physical time (seconds), and STFT/CRNN frame indices.

Strictly handles framing geometry. Does NOT perform FFT, NMF, or I/O operations.
"""

from typing import Tuple, Union
import numpy as np
import config


def samples_to_frames(
    num_samples: int,
    n_fft: int = config.N_FFT,
    hop_length: int = config.HOP_LENGTH,
    center: bool = config.CENTER_STFT
) -> int:
    """
    Calculate the total number of frames produced from a given number of audio samples.

    Parameters
    ----------
    num_samples : int
        Total length of the audio signal in discrete samples.
    n_fft : int, optional
        Analysis frame length (N_FFT).
    hop_length : int, optional
        Hop size between consecutive frames.
    center : bool, optional
        Whether the STFT frame is centered with zero-padding.

    Returns
    -------
    int
        Total number of computable frames T.
    """
    if num_samples <= 0:
        return 0

    if center:
        # Standard librosa centered padding: pad with n_fft // 2 on both sides
        return int(num_samples // hop_length) + 1
    else:
        # Exact non-centered slicing (no zero-padding at boundaries)
        if num_samples < n_fft:
            return 0
        return int((num_samples - n_fft) // hop_length) + 1


def frame_to_samples(
    frame_idx: Union[int, np.ndarray],
    n_fft: int = config.N_FFT,
    hop_length: int = config.HOP_LENGTH,
    center: bool = config.CENTER_STFT
) -> Tuple[Union[int, np.ndarray], Union[int, np.ndarray]]:
    """
    Get the start and end audio sample indices [start_sample, end_sample) 
    corresponding to a specific frame index.

    Parameters
    ----------
    frame_idx : int or np.ndarray
        Frame index (0-based) or an array of frame indices.
    n_fft : int, optional
        Analysis frame length.
    hop_length : int, optional
        Hop size between consecutive frames.
    center : bool, optional
        Whether STFT is centered.

    Returns
    -------
    start_sample : int or np.ndarray
        Starting sample index (inclusive).
    end_sample : int or np.ndarray
        Ending sample index (exclusive).
    """
    if center:
        start_sample = frame_idx * hop_length - (n_fft // 2)
        end_sample = start_sample + n_fft
    else:
        start_sample = frame_idx * hop_length
        end_sample = start_sample + n_fft

    return start_sample, end_sample


def frame_to_time(
    frame_idx: Union[int, np.ndarray],
    hop_length: int = config.HOP_LENGTH,
    sr: int = config.SAMPLE_RATE
) -> Union[float, np.ndarray]:
    """
    Convert a frame index to its physical timestamp (in seconds) at the frame's center.

    Parameters
    ----------
    frame_idx : int or np.ndarray
        Frame index (0-based).
    hop_length : int, optional
        Hop size between frames.
    sr : int, optional
        Sampling rate in Hz.

    Returns
    -------
    float or np.ndarray
        Timestamp in seconds.
    """
    # Time corresponding to the start of the hop window
    return (frame_idx * hop_length) / float(sr)


def time_to_frame(
    time_sec: Union[float, np.ndarray],
    hop_length: int = config.HOP_LENGTH,
    sr: int = config.SAMPLE_RATE
) -> Union[int, np.ndarray]:
    """
    Convert a physical timestamp (in seconds) to the nearest frame index.

    Parameters
    ----------
    time_sec : float or np.ndarray
        Time in seconds.
    hop_length : int, optional
        Hop size between frames.
    sr : int, optional
        Sampling rate in Hz.

    Returns
    -------
    int or np.ndarray
        Nearest frame index.
    """
    frame = np.round((np.asarray(time_sec) * sr) / hop_length)
    if isinstance(time_sec, (int, float)):
        return int(frame)
    return frame.astype(int)