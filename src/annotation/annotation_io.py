"""
===============================================================================
Annotation I/O Module
===============================================================================

Handles saving and loading frame-level binary/smoothed ground truth labels (*_labels.npz).
Ensures strict temporal alignment between frames, timestamps, and target labels.
"""

from pathlib import Path
from typing import Union, Dict, Any, Optional
import numpy as np
import config


def create_gaussian_label(
    peak_frames: np.ndarray,
    total_frames: int,
    sigma: float = config.ANNOTATION_GAUSSIAN_SIGMA
) -> np.ndarray:
    """
    Generate a 1D continuous label vector with Gaussian-smoothed impulses at peak locations.

    Parameters
    ----------
    peak_frames : np.ndarray
        Array of integer frame indices where part impacts occur.
    total_frames : int
        Total number of frames T.
    sigma : float, optional
        Standard deviation (radius in frames) of the Gaussian kernel.

    Returns
    -------
    y_smooth : np.ndarray (float32)
        1D array of shape (T,) with values in [0.0, 1.0].
    """
    y_smooth = np.zeros(total_frames, dtype=np.float32)
    radius = int(np.ceil(3.0 * sigma))

    for p in peak_frames:
        if 0 <= p < total_frames:
            start = max(0, p - radius)
            end = min(total_frames, p + radius + 1)
            t_range = np.arange(start, end)
            gaussian = np.exp(-0.5 * ((t_range - p) / sigma) ** 2)
            y_smooth[start:end] = np.maximum(y_smooth[start:end], gaussian)

    return y_smooth


def save_annotation(
    output_path: Union[str, Path],
    audio_filename: str,
    h_event_sum: np.ndarray,
    binary_labels: np.ndarray,
    frame_times: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Save annotation data and H_event_sum feature into an NPZ archive.

    Parameters
    ----------
    output_path : str or Path
        Target destination path ending with .npz.
    audio_filename : str
        Name of the source WAV file.
    h_event_sum : np.ndarray
        1D float32 array of shape (T,).
    binary_labels : np.ndarray
        1D binary array of shape (T,) with values in {0, 1}.
    frame_times : np.ndarray
        1D array of physical timestamps (seconds) for each frame.
    metadata : dict, optional
        Extra metadata (sample rate, n_fft, true count, etc.).

    Returns
    -------
    saved_path : Path
        Resolved path to the saved .npz file.
    """
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    if not (len(h_event_sum) == len(binary_labels) == len(frame_times)):
        raise ValueError(
            f"Array dimension mismatch: h_event_sum ({len(h_event_sum)}), "
            f"binary_labels ({len(binary_labels)}), frame_times ({len(frame_times)})"
        )

    meta_dict = metadata or {}

    np.savez_compressed(
        str(path),
        audio_filename=audio_filename,
        h_event_sum=h_event_sum.astype(np.float32),
        binary_labels=binary_labels.astype(np.uint8),
        frame_times=frame_times.astype(np.float32),
        **meta_dict
    )
    return path


def load_annotation(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load an existing annotation NPZ archive.

    Parameters
    ----------
    file_path : str or Path
        Path to the *_labels.npz file.

    Returns
    -------
    dict
        Dictionary containing all unpacked arrays and metadata.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file not found at: {path}")

    with np.load(str(path), allow_pickle=True) as data:
        return {key: data[key] for key in data.files}