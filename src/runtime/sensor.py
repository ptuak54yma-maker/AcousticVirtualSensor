"""
===============================================================================
Acoustic Virtual Sensor Core Engine (src/runtime/sensor.py)
===============================================================================

Primary virtual sensor core running NMF + Peak Detection pipeline.
Supports offline audio files and streaming buffers without PyTorch dependencies.
"""

from typing import Dict, Any, Optional
from collections import deque
import numpy as np

import config
from src.nmf.dictionary import load_dictionary
from src.nmf.nnls import solve_nnls_frame, solve_nnls_batch
from src.nmf.activation import compute_h_event_sum
from src.preprocessing.stft import compute_frame_stft, compute_stft
from src.preprocessing.log_compression import log_compression
from src.event.peak_detection import detect_peaks_envelope
from src.event.counting import count_parts_from_events, compute_feed_rate


class AcousticVirtualSensorCore:
    """
    Hardware-ready Acoustic Virtual Sensor processing core using NMF + Peak Detection.
    """

    def __init__(
        self,
        mode: str = "baseline_nmf",
        w_path=config.W_STANDARD_PATH,
        peak_params_path=getattr(config, "PEAK_PARAMS_PATH", None)
    ):
        self.mode = mode
        self.w_matrix = load_dictionary(w_path)
        self.peak_params_path = peak_params_path

        # Streaming Ring Buffers (dùng khi chạy real-time/streaming)
        self.audio_buffer = np.empty(0, dtype=np.float32)
        self.max_frames = int(round(config.WINDOW_SECONDS * (config.SAMPLE_RATE / config.HOP_LENGTH)))
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
        Thực hiện inference offline trên toàn bộ file âm thanh:
        STFT -> Log-Magnitude -> NNLS (W_standard) -> H_event_sum -> Peak Detection.
        """
        # 1. Phổ STFT và nén Log-Magnitude
        stft_matrix = compute_stft(audio_mono)
        v_matrix = log_compression(stft_matrix)

        # 2. Phân rã ma trận kích hoạt qua NNLS
        h_matrix = solve_nnls_batch(v_matrix, self.w_matrix)

        # 3. Tổng hợp đường bao 24 thành phần sự kiện phôi
        h_event_sum = compute_h_event_sum(h_matrix)

        # 4. Nhận diện đỉnh sự kiện va đập (Peak Picking)
        rep_frames, properties = detect_peaks_envelope(h_event_sum)
        count = count_parts_from_events(rep_frames)

        # 5. Tính toán tốc độ cấp phôi (Feed rate)
        total_duration = len(audio_mono) / float(config.SAMPLE_RATE)
        feed_rate = compute_feed_rate(count, total_duration)

        return {
            "total_count": count,
            "feed_rate": feed_rate,
            "event_frames": rep_frames,
            "h_event_sum": h_event_sum,
            "peak_properties": properties,
            "duration_sec": total_duration
        }