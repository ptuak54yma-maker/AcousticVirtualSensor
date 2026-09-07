"""
===============================================================================
Peak Reference Detection Module
===============================================================================

Generates baseline reference peak locations on H_event_sum using optimized P*, D*.
"""

import json
from typing import Tuple, Optional
import numpy as np
from scipy.signal import find_peaks
import config


def load_peak_parameters() -> Tuple[float, int]:
    """Tải P*, D* từ models/peak_detection/peak_params.json."""
    if config.PEAK_PARAMS_PATH.is_file():
        try:
            with open(config.PEAK_PARAMS_PATH, "r", encoding="utf-8") as f:
                params = json.load(f)
                return float(params["prominence"]), int(params["distance"])
        except Exception:
            pass
    return config.DEFAULT_PEAK_PROMINENCE, config.DEFAULT_PEAK_DISTANCE_FRAMES


def detect_reference_peaks(
    h_event_sum: np.ndarray,
    prominence: Optional[float] = None,
    distance: Optional[int] = None
) -> Tuple[np.ndarray, dict]:
    """
    Phát hiện đỉnh trên H_event_sum sử dụng P*, D* đã được tối ưu từ Train set.
    """
    if h_event_sum.ndim != 1:
        raise ValueError(f"Expected 1D envelope, got shape {h_event_sum.shape}")

    # Nếu không truyền thủ công, tự động load P*, D* tối ưu
    if prominence is None or distance is None:
        opt_p, opt_d = load_peak_parameters()
        prominence = prominence if prominence is not None else opt_p
        distance = distance if distance is not None else opt_d

    peaks, properties = find_peaks(
        h_event_sum,
        prominence=prominence,
        distance=distance
    )
    return peaks.astype(np.int64), properties