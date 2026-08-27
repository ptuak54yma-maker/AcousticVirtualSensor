"""
===============================================================================
Spectral Log-Compression Module
===============================================================================

Transforms linear magnitude or power spectrograms into log-compressed representations:
    V = log(1 + S)  (Log-Magnitude)
or
    V = log(1 + |X|^2)  (Log-Power)

Strictly mathematical transformation. Does NOT perform STFT, NMF, or I/O.
"""

from typing import Union
import numpy as np
import config


def log_compression(
    spectrogram: np.ndarray,
    spec_type: str = config.SPECTROGRAM_TYPE,
    epsilon: float = config.LOG_EPSILON
) -> np.ndarray:
    """
    Apply logarithmic dynamic range compression to a complex or real spectrogram.

    Parameters
    ----------
    spectrogram : np.ndarray
        Complex STFT matrix X or real-valued magnitude/power spectrogram S.
        Shape: (F, T) or (F, 1).
    spec_type : str, optional
        "log_magnitude" computes log(1 + |X|).
        "log_power" computes log(1 + |X|^2).
        Defaults to config.SPECTROGRAM_TYPE.
    epsilon : float, optional
        Offset inside the logarithm. Defaults to 1.0 (via np.log1p).

    Returns
    -------
    V : np.ndarray (float32)
        Non-negative log-compressed spectral matrix of identical spatial shape.
    """
    if np.iscomplexobj(spectrogram):
        magnitude = np.abs(spectrogram)
    else:
        magnitude = spectrogram

    if spec_type == "log_magnitude":
        # Paper definition: V = log(1 + |X|)
        if epsilon == 1.0:
            v = np.log1p(magnitude)
        else:
            v = np.log(magnitude + epsilon)
    elif spec_type == "log_power":
        # Alternative: V = log(1 + |X|^2)
        power = np.square(magnitude)
        if epsilon == 1.0:
            v = np.log1p(power)
        else:
            v = np.log(power + epsilon)
    else:
        raise ValueError(f"Unsupported spectrogram type: '{spec_type}'")

    return np.ascontiguousarray(v, dtype=np.float32)