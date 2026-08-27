"""
===============================================================================
Audio Loader Module
===============================================================================

Provides pure audio ingestion functionality.
Strictly handles loading WAV files, validating channels/sample rates, 
converting to mono float32, and removing initial DC offset.

Does NOT execute STFT, NMF, or any neural model inference.
"""

from pathlib import Path
from typing import Tuple, Union, Dict, Any
import numpy as np
import soundfile as sf

import config


def load_audio(
    file_path: Union[str, Path],
    target_sr: int = config.SAMPLE_RATE,
    remove_dc: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file, convert it to single-channel (mono) float32,
    validate its sampling rate, and optionally remove DC bias.

    Parameters
    ----------
    file_path : str or Path
        Path to the target WAV audio file.
    target_sr : int, optional
        Expected sample rate in Hz. Defaults to config.SAMPLE_RATE (44100).
    remove_dc : bool, optional
        Whether to subtract the mean amplitude across the signal. Defaults to True.

    Returns
    -------
    audio : np.ndarray
        1D NumPy array with dtype float32 bounded in [-1.0, 1.0].
    sr : int
        Verified sampling rate of the audio file.

    Raises
    ------
    FileNotFoundError
        If the audio file does not exist.
    ValueError
        If the audio file's sampling rate does not match target_sr.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found at: {path}")

    # Load audio in native 32-bit floating point format [-1.0, 1.0]
    audio_data, sr = sf.read(str(path), dtype="float32", always_2d=False)

    # Validate sampling rate
    if sr != target_sr:
        raise ValueError(
            f"Sample rate mismatch for '{path.name}'. Expected {target_sr} Hz, got {sr} Hz."
        )

    # Convert multi-channel (stereo) to mono 1D array
    if audio_data.ndim == 2:
        audio_data = np.mean(audio_data, axis=1)
    elif audio_data.ndim > 2:
        raise ValueError(f"Unsupported audio channel dimension: {audio_data.ndim}")

    # Remove DC offset if requested
    if remove_dc:
        audio_data = audio_data - np.mean(audio_data)

    return np.ascontiguousarray(audio_data, dtype=np.float32), sr


def get_audio_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Retrieve audio metadata without loading the entire waveform into memory.

    Parameters
    ----------
    file_path : str or Path
        Path to the target WAV audio file.

    Returns
    -------
    dict
        Dictionary containing duration (seconds), frames, channels, and samplerate.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found at: {path}")

    info = sf.info(str(path))
    return {
        "filename": path.name,
        "samplerate": info.samplerate,
        "channels": info.channels,
        "duration_seconds": info.duration,
        "total_samples": info.frames,
        "format": info.format,
        "subtype": info.subtype
    }