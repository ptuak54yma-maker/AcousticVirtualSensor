"""
===============================================================================
Acoustic Virtual Sensor Core Engine (src/runtime/sensor.py)
===============================================================================

Unified Virtual Sensor class supporting both Offline Audio Files and 
Simulated/Real Microphone Streaming buffers.
Runs STFT -> Log-Spectrum -> NMF -> Hybrid CRNN -> Event Grouping -> Count/Rate.
"""

from typing import Dict, Any, Optional
from collections import deque
import numpy as np
import librosa

import config
from src.nmf.dictionary import load_dictionary
from src.nmf.nnls import solve_nnls_frame, solve_nnls_batch
from src.nmf.activation import compute_h_event_sum
from src.preprocessing.stft import compute_frame_stft, compute_stft
from src.preprocessing.log_compression import log_compression
from src.crnn.inference import CRNNPredictor
from src.event.binary_to_event import probability_to_binary, group_binary_events
from src.event.peak_detection import detect_peaks_envelope
from src.event.counting import count_parts_from_events, compute_feed_rate


class AcousticVirtualSensorCore:
    """
    Hardware-ready Acoustic Virtual Sensor processing core.
    """

    def __init__(
        self,
        mode: str = "crnn",  # "crnn" or "baseline_nmf"
        w_path=config.W_STANDARD_PATH,
        crnn_weights_path=config.CRNN_MODEL_PATH
    ):
        self.mode = mode
        self.w_matrix = load_dictionary(w_path)
        
        if self.mode == "crnn":
            self.crnn_predictor = CRNNPredictor(crnn_weights_path)
        else:
            self.crnn_predictor = None

        # Streaming Ring Buffers
        self.audio_buffer = np.empty(0, dtype=np.float32)
        self.max_frames = int(round(config.WINDOW_SECONDS * (config.SAMPLE_RATE / config.HOP_LENGTH)))  # ~129 frames
        self.he_buffer = deque(maxlen=self.max_frames)
        
        self.elapsed_samples = 0
        self.total_count = 0
        self.current_feed_rate = 0.0

    def reset(self):
        """Reset internal memory buffers."""
        self.audio_buffer = np.empty(0, dtype=np.float32)
        self.he_buffer.clear()
        self.elapsed_samples = 0
        self.total_count = 0
        self.current_feed_rate = 0.0

    def process_offline_file(self, audio_mono: np.ndarray) -> Dict[str, Any]:
        """
        Batch inference on a full audio recording.
        """
        stft_matrix = compute_stft(audio_mono)
        v_matrix = log_compression(stft_matrix)
        h_matrix = solve_nnls_batch(v_matrix, self.w_matrix)
        h_event_sum = compute_h_event_sum(h_matrix)

        if self.mode == "crnn":
            # Compute Mel Spectrogram
            magnitude = np.abs(stft_matrix)
            mel_basis = librosa.filters.mel(
                sr=config.SAMPLE_RATE,
                n_fft=config.N_FFT,
                n_mels=config.N_MELS,
                fmin=config.F_MIN,
                fmax=config.F_MAX
            )
            log_mel = np.log1p(np.dot(mel_basis, magnitude))

            probs = self.crnn_predictor.predict_probabilities(log_mel, h_event_sum)
            binary_seq = probability_to_binary(probs, threshold=config.CRNN_THRESHOLD)
            events = group_binary_events(binary_seq, probabilities=probs)
            rep_frames = np.array([e["representative_frame"] for e in events], dtype=np.int64)
            count = len(events)
        else:
            rep_frames, _ = detect_peaks_envelope(h_event_sum)
            count = len(rep_frames)
            probs = None
            binary_seq = None

        total_duration = len(audio_mono) / float(config.SAMPLE_RATE)
        feed_rate = count / max(1e-6, total_duration)

        return {
            "total_count": count,
            "feed_rate": feed_rate,
            "event_frames": rep_frames,
            "h_event_sum": h_event_sum,
            "probabilities": probs,
            "binary_seq": binary_seq
        }