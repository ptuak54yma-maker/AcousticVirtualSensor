"""
===============================================================================
Peak Reference Detection Module
===============================================================================

Generates baseline reference peak locations on H_event_sum for annotation guidance.
Strictly used as an annotation assistant and baseline comparison.
"""

from typing import Tuple, Optional
import numpy as np
from scipy.signal import find_peaks
import config


def detect_reference_peaks(
    h_event_sum: np.ndarray,
    prominence: Optional[float] = config.PEAK_PROMINENCE,
    distance: Optional[int] = config.PEAK_DISTANCE_FRAMES
) -> Tuple[np.ndarray, dict]:
    """
    Detect peaks on the H_event_sum envelope using configured prominence and distance.

    Parameters
    ----------
    h_event_sum : np.ndarray
        1D event envelope array of shape (T,).
    prominence : float, optional
        Minimum peak prominence relative to baseline. Defaults to config.PEAK_PROMINENCE.
    distance : int, optional
        Minimum frame distance between adjacent peaks. Defaults to config.PEAK_DISTANCE_FRAMES.

    Returns
    -------
    peaks : np.ndarray
        1D array containing indices of detected peak frames.
    properties : dict
        Properties dictionary returned by scipy.signal.find_peaks (prominences, etc.).
    """
    if h_event_sum.ndim != 1:
        raise ValueError(f"Expected 1D envelope, got shape {h_event_sum.shape}")

    peaks, properties = find_peaks(
        h_event_sum,
        prominence=prominence,
        distance=distance
    )
    return peaks.astype(np.int64), properties