"""
===============================================================================
NMF Activation Extraction Module
===============================================================================

Extracts part-impact event activation envelopes from the full activation matrix H:
    h_event(t) = sum_{k in K_event} H(k, t)

Strictly mathematical aggregation. Does NOT perform NNLS, peak detection, or CRNN inference.
"""

from typing import Union
import numpy as np
import config


def extract_event_activation(
    h_matrix: np.ndarray,
    n_event_components: int = config.EVENT_COMPONENTS
) -> np.ndarray:
    """
    Extract the sub-matrix H_active corresponding to the part-impact components.

    Parameters
    ----------
    h_matrix : np.ndarray
        Full activation matrix H of shape (K, T) or 1D vector of shape (K,).
    n_event_components : int, optional
        Number of initial rows corresponding to part events (default: 24).

    Returns
    -------
    h_event : np.ndarray
        Event activation matrix of shape (n_event_components, T) or (n_event_components,).
    """
    if h_matrix.ndim == 1:
        if len(h_matrix) < n_event_components:
            raise ValueError(
                f"Activation vector length ({len(h_matrix)}) is smaller than n_event_components ({n_event_components})"
            )
        return h_matrix[:n_event_components]
    elif h_matrix.ndim == 2:
        if h_matrix.shape[0] < n_event_components:
            raise ValueError(
                f"Activation matrix rows ({h_matrix.shape[0]}) are smaller than n_event_components ({n_event_components})"
            )
        return h_matrix[:n_event_components, :]
    else:
        raise ValueError(f"Unsupported activation array dimension: {h_matrix.ndim}")


def compute_h_event_sum(
    h_matrix: np.ndarray,
    n_event_components: int = config.EVENT_COMPONENTS
) -> Union[float, np.ndarray]:
    """
    Compute the 1D event activation envelope by summing across the event components:
        h_event_sum(t) = sum_{i=0}^{n_event_components-1} H(i, t)

    Parameters
    ----------
    h_matrix : np.ndarray
        Full activation matrix of shape (K, T) or 1D vector of shape (K,).
    n_event_components : int, optional
        Number of event components. Defaults to config.EVENT_COMPONENTS.

    Returns
    -------
    h_event_sum : float or np.ndarray (float32)
        1D activation envelope of shape (T,) or a scalar float if input is a single frame.
    """
    h_event = extract_event_activation(h_matrix, n_event_components=n_event_components)
    
    if h_event.ndim == 1:
        return float(np.sum(h_event, dtype=np.float32))
    
    envelope = np.sum(h_event, axis=0, dtype=np.float32)
    return np.ascontiguousarray(envelope, dtype=np.float32)