"""
===============================================================================
Peak Detection Baseline Module
===============================================================================

Implements the baseline peak picking algorithm on continuous envelopes.
"""

from typing import Tuple, Optional
import numpy as np
from scipy.signal import find_peaks

import config


def detect_peaks_envelope(
    envelope: np.ndarray,
    prominence: Optional[float] = config.PEAK_PROMINENCE,
    distance: Optional[int] = config.PEAK_DISTANCE_FRAMES
) -> Tuple[np.ndarray, dict]:
    """
    Detect peaks from an activation envelope.

    Parameters
    ----------
    envelope : np.ndarray
        1D float32 array of shape (T,).
    prominence : float, optional
        Minimum prominence threshold.
    distance : int, optional
        Minimum distance between adjacent peaks.

    Returns
    -------
    peaks : np.ndarray
        Frame indices of valid detected peaks.
    properties : dict
        Peak properties dictionary from scipy.signal.find_peaks.
    """
    if envelope.ndim != 1:
        raise ValueError(f"Envelope must be 1D, got shape {envelope.shape}")

    peaks, properties = find_peaks(
        envelope,
        prominence=prominence,
        distance=distance
    )
    return peaks.astype(np.int64), properties