"""
===============================================================================
Peak Detection Baseline Module (src/event/peak_detection.py)
===============================================================================

Implements the baseline peak picking algorithm using trained parameters (P*, D*).
Falls back to config.DEFAULT_PEAK_PROMINENCE and config.DEFAULT_PEAK_DISTANCE_FRAMES
to avoid AttributeError.
"""

import json
from typing import Tuple, Optional
import numpy as np
from scipy.signal import find_peaks

import config


def load_trained_peak_params() -> Tuple[float, int]:
    """Nạp chính xác P*, D* từ models/peak_detection/peak_params.json hoặc fallback về config."""
    if hasattr(config, "PEAK_PARAMS_PATH") and config.PEAK_PARAMS_PATH.is_file():
        try:
            with open(config.PEAK_PARAMS_PATH, "r", encoding="utf-8") as f:
                params = json.load(f)
                return float(params["prominence"]), int(params["distance"])
        except Exception:
            pass

    return config.DEFAULT_PEAK_PROMINENCE, config.DEFAULT_PEAK_DISTANCE_FRAMES


def detect_peaks_envelope(
    envelope: np.ndarray,
    prominence: Optional[float] = None,
    distance: Optional[int] = None
) -> Tuple[np.ndarray, dict]:
    """
    Phát hiện đỉnh trên đường bao năng lượng sử dụng P*, D* đã huấn luyện từ trước.
    """
    if envelope.ndim != 1:
        raise ValueError(f"Envelope must be 1D, got shape {envelope.shape}")

    if prominence is None or distance is None:
        opt_p, opt_d = load_trained_peak_params()
        prominence = prominence if prominence is not None else opt_p
        distance = distance if distance is not None else opt_d

    peaks, properties = find_peaks(
        envelope,
        prominence=prominence,
        distance=distance
    )
    return peaks.astype(np.int64), properties