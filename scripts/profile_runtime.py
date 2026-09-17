"""
===============================================================================
Script: PC Reference Runtime Profiler (scripts/profile_runtime.py)
===============================================================================

Profiles frame-by-frame computational latency of the reference NMF pipeline:
    T_per_frame = T_STFT + T_LOG + T_NNLS + T_HE_SUM
and measures peak detection cost using trained (P*, D*) parameters.

Purpose:
  - Benchmark software performance on host machine.
  - Identify computational bottlenecks in the reference implementation.
  - Compare against physical frame budget T_budget = Hop / fs ≈ 23.22 ms.
"""

import sys
import time
from pathlib import Path

# Đảm bảo PROJECT_ROOT nằm đầu sys.path trước khi import config
PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
from src.event.peak_detection import detect_peaks_envelope, load_trained_peak_params


def get_stats(arr):
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr))
    }


def profile_pipeline():
    test_files = sorted(list(config.TEST_AUDIO_DIR.glob("*.wav")))
    if not test_files:
        test_files = sorted(list(config.TRAIN_AUDIO_DIR.glob("*.wav")))
    if not test_files:
        print("[!] Không tìm thấy file WAV nào để profiling.")
        return

    wav_path = test_files[0]
    print(f"[*] Khởi chạy PC Reference Runtime Profiler trên: {wav_path.name}")
    audio, sr = load_audio(wav_path, target_sr=config.SAMPLE_RATE)
    w_matrix = load_dictionary(config.W_STANDARD_PATH)
    p_opt, d_opt = load_trained_peak_params()

    hop_length = config.HOP_LENGTH
    n_fft = config.N_FFT
    frame_budget_ms = (hop_length / float(sr)) * 1000.0

    total_samples = len(audio)
    num_frames = (total_samples - n_fft) // hop_length + 1

    latencies_stft = []
    latencies_log = []
    latencies_nnls = []
    latencies_hesum = []
    latencies_frame_total = []
    h_event_stream = []

    print(f"[*] Đang xử lý {num_frames} frames (Frame Processing Budget: {frame_budget_ms:.2f} ms)...")

    for i in range(num_frames):
        start_idx = i * hop_length
        frame_audio = audio[start_idx : start_idx + n_fft]

        t0 = time.perf_counter()

        # 1. STFT
        t_start = time.perf_counter()
        stft_frame = compute_frame_stft(frame_audio)
        t_stft = (time.perf_counter() - t_start) * 1000.0

        # 2. Log-magnitude compression
        t_start = time.perf_counter()
        v_frame = log_compression(stft_frame)
        t_log = (time.perf_counter() - t_start) * 1000.0

        # 3. Frame-wise NNLS (Scipy reference solver)
        t_start = time.perf_counter()
        h_frame = solve_nnls_frame(v_frame, w_matrix)
        t_nnls = (time.perf_counter() - t_start) * 1000.0

        # 4. H_event_sum accumulation
        t_start = time.perf_counter()
        he_val = compute_h_event_sum(h_frame)
        t_hesum = (time.perf_counter() - t_start) * 1000.0

        t_frame_total = (time.perf_counter() - t0) * 1000.0

        h_event_stream.append(he_val)
        latencies_stft.append(t_stft)
        latencies_log.append(t_log)
        latencies_nnls.append(t_nnls)
        latencies_hesum.append(t_hesum)
        latencies_frame_total.append(t_frame_total)

    # 5. Offline Reference Peak Detection trên toàn chuỗi với P*, D* chính thức
    h_event_array = np.array(h_event_stream, dtype=np.float32)
    t_start = time.perf_counter()
    peaks_detected, _ = detect_peaks_envelope(h_event_array, prominence=p_opt, distance=d_opt)
    t_peak_total = (time.perf_counter() - t_start) * 1000.0
    t_peak_amortized = t_peak_total / float(num_frames)

    s_stft = get_stats(latencies_stft)
    s_log = get_stats(latencies_log)
    s_nnls = get_stats(latencies_nnls)
    s_hesum = get_stats(latencies_hesum)
    s_total = get_stats(latencies_frame_total)

    # Workload ratio (%)
    pct_stft = (s_stft["mean"] / s_total["mean"]) * 100.0
    pct_log = (s_log["mean"] / s_total["mean"]) * 100.0
    pct_nnls = (s_nnls["mean"] / s_total["mean"]) * 100.0
    pct_hesum = (s_hesum["mean"] / s_total["mean"]) * 100.0

    ref_budget_usage = (s_total["mean"] / frame_budget_ms) * 100.0
    ref_margin = frame_budget_ms - s_total["mean"]

    print("\n" + "=" * 85)
    print("      PC REFERENCE RUNTIME BENCHMARK (NMF + PEAK DETECTION)")
    print("=" * 85)
    print(f"  Frame Processing Budget (Hop / fs) : {frame_budget_ms:.2f} ms")
    print(f"  PC Reference Execution Mean        : {s_total['mean']:.3f} ms/frame")
    print(f"  Reference Real-Time Workload Ratio : {ref_budget_usage:.2f}% of frame budget")
    print(f"  Reference Real-Time Margin         : {ref_margin:.2f} ms headroom (on PC)")
    print("-" * 85)
    print(f"{'Stage / Processing Block':<25} | {'Mean(ms)':<8} | {'Median':<8} | {'P95(ms)':<8} | {'P99(ms)':<8} | {'Max(ms)':<8} | {'Workload %':<10}")
    print("-" * 85)
    print(f"{'1. STFT Frame':<25} | {s_stft['mean']:<8.3f} | {s_stft['median']:<8.3f} | {s_stft['p95']:<8.3f} | {s_stft['p99']:<8.3f} | {s_stft['max']:<8.3f} | {pct_stft:<10.2f}%")
    print(f"{'2. Log-Compression':<25} | {s_log['mean']:<8.3f} | {s_log['median']:<8.3f} | {s_log['p95']:<8.3f} | {s_log['p99']:<8.3f} | {s_log['max']:<8.3f} | {pct_log:<10.2f}%")
    print(f"{'3. NNLS Solver':<25} | {s_nnls['mean']:<8.3f} | {s_nnls['median']:<8.3f} | {s_nnls['p95']:<8.3f} | {s_nnls['p99']:<8.3f} | {s_nnls['max']:<8.3f} | {pct_nnls:<10.2f}%")
    print(f"{'4. H_event_sum':<25} | {s_hesum['mean']:<8.3f} | {s_hesum['median']:<8.3f} | {s_hesum['p95']:<8.3f} | {s_hesum['p99']:<8.3f} | {s_hesum['max']:<8.3f} | {pct_hesum:<10.2f}%")
    print("-" * 85)
    print(f"{'FRAME PROCESSING TOTAL':<25} | {s_total['mean']:<8.3f} | {s_total['median']:<8.3f} | {s_total['p95']:<8.3f} | {s_total['p99']:<8.3f} | {s_total['max']:<8.3f} | 100.00%")
    print("=" * 85)
    print("  [Reference Findings & Notes]:")
    print(f"    - Peak Parameters Loaded             : P* = {p_opt}, D* = {d_opt} frames")
    print(f"    - Offline find_peaks Reference Cost  : {t_peak_amortized:.4f} ms/frame amortized ({t_peak_total:.2f} ms total, {len(peaks_detected)} events)")
    print(f"    - Dominant computational stage       : NNLS (~{pct_nnls:.1f}% in PC reference implementation)")
    print("    - Important Caveat                   : Latency figures are measured on Host PC (x86_64).")
    print("                                           Actual MCU execution time depends on clock frequency and DSP/FPU units.\n")


if __name__ == "__main__":
    profile_pipeline()