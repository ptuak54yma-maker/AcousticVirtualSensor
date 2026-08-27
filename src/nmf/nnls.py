"""
===============================================================================
Non-Negative Least Squares (NNLS) Solver Module
===============================================================================

Solves for activation matrix/vector H given fixed basis W and target V:
    min_{H >= 0} || V - W H ||_F^2

Provides both frame-by-frame and batch solvers via scipy.optimize.nnls.
"""

from typing import Tuple, Optional
import numpy as np
from scipy.optimize import nnls

import config


def solve_nnls_frame(
    v_frame: np.ndarray,
    w_matrix: np.ndarray,
    max_iter: Optional[int] = config.NNLS_MAX_ITER
) -> np.ndarray:
    """
    Solve NNLS for a single frame vector: min_{h >= 0} || v_frame - W * h ||_2^2.

    Parameters
    ----------
    v_frame : np.ndarray
        1D non-negative spectral column vector of shape (F,) or (F, 1).
    w_matrix : np.ndarray
        Fixed basis dictionary of shape (F, K).
    max_iter : int, optional
        Maximum iterations for the active set solver. Defaults to config.NNLS_MAX_ITER.

    Returns
    -------
    h : np.ndarray (float32)
        1D non-negative activation vector of shape (K,).
    """
    target = v_frame.squeeze()
    if target.ndim != 1:
        raise ValueError(f"v_frame must be 1D, got shape {v_frame.shape}")

    if max_iter is not None:
        h, _ = nnls(w_matrix, target, maxiter=max_iter)
    else:
        h, _ = nnls(w_matrix, target)

    return np.ascontiguousarray(h, dtype=np.float32)


def solve_nnls_batch(
    v_matrix: np.ndarray,
    w_matrix: np.ndarray,
    max_iter: Optional[int] = config.NNLS_MAX_ITER
) -> np.ndarray:
    """
    Solve NNLS column-by-column across a full spectrogram matrix V.

    Parameters
    ----------
    v_matrix : np.ndarray
        2D non-negative spectral matrix of shape (F, T).
    w_matrix : np.ndarray
        Fixed basis dictionary of shape (F, K).
    max_iter : int, optional
        Maximum iterations per frame. Defaults to config.NNLS_MAX_ITER.

    Returns
    -------
    H : np.ndarray (float32)
        2D non-negative activation matrix of shape (K, T).
    """
    if v_matrix.ndim != 2:
        raise ValueError(f"v_matrix must be 2D, got shape {v_matrix.shape}")

    num_bins, num_frames = v_matrix.shape
    num_components = w_matrix.shape[1]

    h_matrix = np.zeros((num_components, num_frames), dtype=np.float32)

    for t in range(num_frames):
        target = v_matrix[:, t]
        if max_iter is not None:
            h_col, _ = nnls(w_matrix, target, maxiter=max_iter)
        else:
            h_col, _ = nnls(w_matrix, target)
        h_matrix[:, t] = h_col

    return h_matrix