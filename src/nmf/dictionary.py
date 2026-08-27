"""
===============================================================================
NMF Dictionary Loader & Validator Module
===============================================================================

Handles loading, validating, and slicing the pre-trained fixed dictionary W_standard.
Validates matrix dimensions: (F, K) = (1025, 48).

Does NOT solve NNLS or perform event envelope summing.
"""

from pathlib import Path
from typing import Union, Tuple
import numpy as np
import config


def load_dictionary(
    dictionary_path: Union[str, Path] = config.W_STANDARD_PATH,
    expected_freq_bins: int = (config.N_FFT // 2) + 1,
    expected_components: int = config.TOTAL_COMPONENTS
) -> np.ndarray:
    """
    Load the fixed NMF dictionary W_standard and verify its shape and non-negativity.

    Parameters
    ----------
    dictionary_path : str or Path
        Path to the saved W_standard.npy file.
    expected_freq_bins : int, optional
        Expected number of frequency bins F (1025 for N_FFT=2048).
    expected_components : int, optional
        Expected total number of components K (48 = 24 Event + 12 Bowl + 12 Env).

    Returns
    -------
    W : np.ndarray (float32)
        Validated 2D non-negative basis matrix of shape (1025, 48).

    Raises
    ------
    FileNotFoundError
        If the .npy file does not exist at dictionary_path.
    ValueError
        If dimensions do not match or if W contains negative values / NaNs.
    """
    path = Path(dictionary_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"NMF Dictionary file not found at: {path}")

    w = np.load(str(path))

    # Validate shape
    if w.ndim != 2:
        raise ValueError(f"W_standard must be a 2D matrix, got ndim={w.ndim}")

    freq_bins, num_components = w.shape
    if freq_bins != expected_freq_bins:
        raise ValueError(
            f"Frequency bin mismatch in W. Expected {expected_freq_bins}, got {freq_bins}."
        )

    if num_components != expected_components:
        raise ValueError(
            f"Component count mismatch in W. Expected {expected_components}, got {num_components}."
        )

    # Validate non-negativity
    if np.isnan(w).any() or np.isinf(w).any():
        raise ValueError("W_standard contains NaN or Inf values.")

    if (w < 0.0).any():
        raise ValueError("W_standard violates non-negativity constraint (contains values < 0).")

    return np.ascontiguousarray(w, dtype=np.float32)


def get_sub_dictionaries(
    w: np.ndarray,
    n_event: int = config.EVENT_COMPONENTS,
    n_bowl: int = config.BOWL_COMPONENTS,
    n_env: int = config.ENV_COMPONENTS
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Slice W_standard into its three constituent source dictionaries:
    W_event, W_bowl, and W_env.

    Parameters
    ----------
    w : np.ndarray
        Full basis matrix (1025, 48).
    n_event : int
        Number of event/impact basis vectors (default: 24).
    n_bowl : int
        Number of bowl-drive basis vectors (default: 12).
    n_env : int
        Number of environmental basis vectors (default: 12).

    Returns
    -------
    w_event : np.ndarray
        Matrix of shape (1025, n_event).
    w_bowl : np.ndarray
        Matrix of shape (1025, n_bowl).
    w_env : np.ndarray
        Matrix of shape (1025, n_env).
    """
    idx_bowl = n_event + n_bowl
    w_event = w[:, :n_event]
    w_bowl = w[:, n_event:idx_bowl]
    w_env = w[:, idx_bowl:idx_bowl + n_env]

    return w_event, w_bowl, w_env