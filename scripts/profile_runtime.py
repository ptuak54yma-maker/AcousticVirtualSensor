"""
===============================================================================
Script: Runtime & Latency Profiling (scripts/profile_runtime.py)
===============================================================================

Profiles frame-by-frame computational latency of the primary pipeline:
    T_total = T_STFT + T_LOG + T_NNLS + T_HE_SUM + T_PEAK

Compares against real-time budget T_budget = Hop / fs ≈ 23.22 ms.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import config
from src.audio.loader import load_audio
from src.preprocessing.stft import compute_frame_stft
from src.preprocessing.log_compression import log_compression
from src.nmf.dictionary import load_dictionary
from src.nmf.nnls import solve_nnls_frame
from src.nmf.activation import compute_h_event_sum
from src.event.peak_detection import detect_peaks_envelope


def profile_pipeline():
    # 1. Chọn file âm thanh thử nghiệm
    test_files = sorted(list(config.TEST_AUDIO_DIR.glob("*.wav")))
    if not test_files:
        test_files = sorted(list(config.TRAIN_AUDIO_DIR.glob("*.wav")))
    if not test_files:
        print("[!] Không tìm thấy file WAV nào để profiling.")
        return

    wav_path = test_files[0]
    print(f"[*] Bắt đầu Profiling trên file: {wav_path.name}")
    audio, sr = load_audio(wav_path, target_sr=config.SAMPLE_RATE)

    # 2. Nạp từ điển NMF
    w_matrix = load_dictionary(config.W_STANDARD_PATH)

    hop_length = config.HOP_LENGTH
    n_fft = config.N_FFT
    frame_budget_ms = (hop_length / sr) * 1000.0

    total_samples = len(audio)
    num_frames = (total_samples - n_fft) // hop_length + 1

    latencies_stft = []
    latencies_log = []
    latencies_nnls = []
    latencies_hesum = []
    latencies_total = []

    print(f"[*] Tổng số frames cần xử lý: {num_frames} (Budget mỗi frame: {frame_budget_ms:.2f} ms)")

    # Giả lập buffer đẩy từng frame để đo độ trễ thực tế
    h_event_stream = []

    for i in range(num_frames):
        start_idx = i * hop_length
        frame_audio = audio[start_idx : start_idx + n_fft]

        t0 = time.perf_counter()

        # Step 1: STFT
        t_start = time.perf_counter()
        stft_frame = compute_frame_stft(frame_audio)
        t_stft = (time.perf_counter() - t_start) * 1000.0

        # Step 2: Log-magnitude
        t_start = time.perf_counter()
        v_frame = log_compression(stft_frame)
        t_log = (time.perf_counter() - t_start) * 1000.0

        # Step 3: NNLS
        t_start = time.perf_counter()
        h_frame = solve_nnls_frame(v_frame, w_matrix)
        t_nnls = (time.perf_counter() - t_start) * 1000.0

        # Step 4: H_event_sum
        t_start = time.perf_counter()
        he_val = compute_h_event_sum(h_frame)
        t_hesum = (time.perf_counter() - t_start) * 1000.0

        t_frame_total = (time.perf_counter() - t0) * 1000.0

        h_event_stream.append(he_val)

        latencies_stft.append(t_stft)
        latencies_log.append(t_log)
        latencies_nnls.append(t_nnls)
        latencies_hesum.append(t_hesum)
        latencies_total.append(t_frame_total)

    # Đo khối phát hiện đỉnh offline trên toàn bộ chuỗi H_event_sum đã trích xuất
    t_start = time.perf_counter()
    peaks, _ = detect_peaks_envelope(np.array(h_event_stream, dtype=np.float32))
    t_peak_total = (time.perf_counter() - t_start) * 1000.0
    t_peak_per_frame = t_peak_total / num_frames

    # Tính toán các chỉ số thống kê
    def get_stats(arr):
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(np.max(arr))
        }

    s_stft = get_stats(latencies_stft)
    s_log = get_stats(latencies_log)
    s_nnls = get_stats(latencies_nnls)
    s_hesum = get_stats(latencies_hesum)
    s_total = get_stats(latencies_total)

    rtf = s_total["mean"] / frame_budget_ms

    print("\n" + "=" * 70)
    print("KẾT QUẢ PROFILING ĐỘ TRỄ THEO TỪNG FRAME (NMF + PEAK DETECTION)")
    print("=" * 70)
    print(f"Ngân sách thời gian thực (Budget T_frame) : {frame_budget_ms:.2f} ms")
    print(f"Thời gian xử lý trung bình (T_total mean): {s_total['mean']:.3f} ms")
    print(f"Real-Time Factor (RTF)                  : {rtf:.4f} ({'ĐẠT CHUẨN REAL-TIME' if rtf < 1.0 else 'QUÁ TẢI'})")
    print("-" * 70)
    print(f"{'Khối chức năng':<20} | {'Mean (ms)':<10} | {'Median (ms)':<11} | {'P95 (ms)':<9} | {'Max (ms)':<9}")
    print("-" * 70)
    print(f"{'1. STFT Frame':<20} | {s_stft['mean']:<10.3f} | {s_stft['median']:<11.3f} | {s_stft['p95']:<9.3f} | {s_stft['max']:<9.3f}")
    print(f"{'2. Log-Compression':<20} | {s_log['mean']:<10.3f} | {s_log['median']:<11.3f} | {s_log['p95']:<9.3f} | {s_log['max']:<9.3f}")
    print(f"{'3. NNLS (Bottleneck)':<20} | {s_nnls['mean']:<10.3f} | {s_nnls['median']:<11.3f} | {s_nnls['p95']:<9.3f} | {s_nnls['max']:<9.3f}")
    print(f"{'4. H_event_sum':<20} | {s_hesum['mean']:<10.3f} | {s_hesum['median']:<11.3f} | {s_hesum['p95']:<9.3f} | {s_hesum['max']:<9.3f}")
    print(f"{'5. Peak Detection':<20} | {t_peak_per_frame:<10.3f} | {'-':<11} | {'-':<9} | {t_peak_total:<9.3f}")
    print("-" * 70)
    print(f"{'TỔNG CỘNG MỖI FRAME':<20} | {s_total['mean']:<10.3f} | {s_total['median']:<11.3f} | {s_total['p95']:<9.3f} | {s_total['max']:<9.3f}")
    print("=" * 70)


if __name__ == "__main__":
    profile_pipeline()