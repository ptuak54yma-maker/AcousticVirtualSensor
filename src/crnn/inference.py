"""
===============================================================================
CRNN Inference Module
===============================================================================

Provides model inference wrappers for generating frame-wise event probabilities 
over continuous full-length recordings.
"""

from pathlib import Path
from typing import Union, Optional  # <-- Bổ sung Optional
import numpy as np
import torch

import config
from src.crnn.model import HybridCRNN


class CRNNPredictor:
    """
    Inference engine for the trained HybridCRNN model.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = config.CRNN_MODEL_PATH,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = HybridCRNN().to(self.device)
        self._load_weights(Path(model_path))
        self.model.eval()

    def _load_weights(self, path: Path):
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Trained CRNN model weights not found at: {path}")
        state_dict = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def predict_probabilities(
        self,
        log_mel: np.ndarray,
        h_event_sum: np.ndarray,
        window_len: int = config.SEQUENCE_LENGTH,
        hop_size: int = config.SEQUENCE_HOP
    ) -> np.ndarray:
        """
        Run sliding window inference and reconstruct a continuous probability sequence.

        Parameters
        ----------
        log_mel : np.ndarray
            Full Log-Mel Spectrogram of shape (N_mels, T).
        h_event_sum : np.ndarray
            Full H_event_sum array of shape (T,) or (1, T).

        Returns
        -------
        probs : np.ndarray (float32)
            1D array of predicted peak probabilities for each frame t in [1:T].
        """
        if h_event_sum.ndim == 1:
            h_event_sum = h_event_sum[np.newaxis, :]

        total_frames = log_mel.shape[1]
        prob_accum = np.zeros(total_frames, dtype=np.float32)
        count_accum = np.zeros(total_frames, dtype=np.float32)

        # Pad if signal is shorter than one window
        pad_len = 0
        if total_frames < window_len:
            pad_len = window_len - total_frames
            log_mel = np.pad(log_mel, ((0, 0), (0, pad_len)), mode="constant")
            h_event_sum = np.pad(h_event_sum, ((0, 0), (0, pad_len)), mode="constant")
            total_frames = window_len
            prob_accum = np.zeros(total_frames, dtype=np.float32)
            count_accum = np.zeros(total_frames, dtype=np.float32)

        for start_idx in range(0, total_frames - window_len + 1, hop_size):
            end_idx = start_idx + window_len
            
            mel_win = torch.from_numpy(log_mel[np.newaxis, np.newaxis, :, start_idx:end_idx]).float().to(self.device)
            he_win = torch.from_numpy(h_event_sum[np.newaxis, :, start_idx:end_idx]).float().to(self.device)

            p_win = self.model.predict_probability(mel_win, he_win).squeeze().cpu().numpy()

            prob_accum[start_idx:end_idx] += p_win
            count_accum[start_idx:end_idx] += 1.0

        # Non-zero division
        count_accum[count_accum == 0] = 1.0
        averaged_probs = prob_accum / count_accum

        if pad_len > 0:
            averaged_probs = averaged_probs[:-pad_len]

        return averaged_probs