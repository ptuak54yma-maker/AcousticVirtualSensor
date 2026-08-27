"""
===============================================================================
Part Counting & Feed Rate Estimation Module
===============================================================================

Calculates total parts count and instantaneous feed rate from detected events.
"""

from typing import List, Dict, Any, Union
import numpy as np


def count_parts_from_events(events: Union[List[Dict[str, Any]], np.ndarray]) -> int:
    """Return total workpiece count from an event list or peak index array."""
    return len(events)


def compute_feed_rate(
    current_count: int,
    window_seconds: float
) -> float:
    """
    Calculate feed rate (workpieces per second).

    Parameters
    ----------
    current_count : int
        Number of detected parts within the sliding window.
    window_seconds : float
        Duration of the measurement window in seconds.

    Returns
    -------
    feed_rate : float
        Parts per second.
    """
    if window_seconds <= 0.0:
        return 0.0
    return float(current_count / window_seconds)