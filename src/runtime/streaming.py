"""
===============================================================================
Streaming Runtime Engine (src/runtime/streaming.py)
===============================================================================

Provides continuous, chunk-by-chunk streaming inference using Ring Buffers.
Emulates real-time hardware ingestion (Microphone / I2S ADC).
"""

from typing import Dict, Any, Optional
from collections import deque
import numpy as np

import config
from src.nmf.activation import compute_h_event_sum
from src.nmf.nnls import solve_nnls_frame
from src.preprocessing.stft import compute_frame_stft
from src.preprocessing.log_compression import log_compression
from src.runtime.sensor import AcousticVirtualSensorCore


class StreamingSensorEngine:
    """
    Real-time streaming engine with Audio Buffer and H_event Buffer.
    """

    def __init__(self, mode: str = "baseline_nmf"):
        self.sensor_core = AcousticVirtualSensorCore(mode=mode)
        
        # Audio Buffer to accumulate samples until frame-ready (N_FFT = 2048)
        self.audio_buffer = np.empty(0, dtype=np.float32)
        
        # Ring buffer for sliding window peak counting (e.g. 3.0s ~ 129 frames)
        self.max_he_frames = int(round(config.WINDOW_SECONDS * (config.SAMPLE_RATE / config.HOP_LENGTH)))
        self.he_buffer = deque(maxlen=self.max_he_frames)
        
        self.elapsed_samples = 0
        self.last_count = 0
        self.last_feed_rate = 0.0

    def reset(self):
        """Reset internal buffers and states."""
        self.audio_buffer = np.empty(0, dtype=np.float32)
        self.he_buffer.clear()
        self.elapsed_samples = 0
        self.last_count = 0
        self.last_feed_rate = 0.0
        self.sensor_core.reset()

    def process_chunk(self, chunk: np.ndarray) -> Dict[str, Any]:
        """
        Process an incoming audio chunk (e.g., 1024 samples).
        """
        # Append incoming chunk
        self.audio_buffer = np.concatenate((self.audio_buffer, chunk.astype(np.float32)))
        self.elapsed_samples += len(chunk)

        # Process frames when enough samples are available (Overlap-Add mechanism)
        while len(self.audio_buffer) >= config.N_FFT:
            frame = self.audio_buffer[:config.N_FFT].copy()
            self.audio_buffer = self.audio_buffer[config.HOP_LENGTH:]

            # Remove DC offset
            frame = frame - np.mean(frame)

            # STFT -> Log -> NNLS -> H_event_sum
            spectrum = compute_frame_stft(frame, n_fft=config.N_FFT)
            v_frame = log_compression(spectrum)
            h_frame = solve_nnls_frame(v_frame, self.sensor_core.w_matrix)
            he_val = compute_h_event_sum(h_frame)

            self.he_buffer.append(he_val)

        # Detect peaks within sliding window
        if len(self.he_buffer) >= self.max_he_frames:
            he_array = np.asarray(self.he_buffer, dtype=np.float32)
            from src.event.peak_detection import detect_peaks_envelope
            peaks, _ = detect_peaks_envelope(he_array)
            self.last_count = len(peaks)
            self.last_feed_rate = self.last_count / float(config.WINDOW_SECONDS)

        return {
            "timestamp": self.elapsed_samples / float(config.SAMPLE_RATE),
            "count": self.last_count,
            "feed_rate": self.last_feed_rate,
            "latest_he": self.he_buffer[-1] if len(self.he_buffer) > 0 else 0.0
        }