"""
===============================================================================
Runtime Profiling Tool for Acoustic Virtual Sensor
===============================================================================

Profiles end-to-end execution latency, memory footprint, and NNLS 
reconstruction characteristics across audio recordings.

Supports two evaluation profiles:
1. Reference Offline Pipeline: Directly mirrors AcousticVirtualSensorCore (window=128, hop=64)
2. Streaming per-frame Schedule: Measures per-frame latency for MCU feasibility analysis
"""

import sys
import time
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import scipy.signal
import torch

# Setup project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.audio.loader import load_audio
from src.preprocessing.stft import compute_stft, compute_frame_stft
from src.preprocessing.log_compression import log_compression
from src.nmf.dictionary import load_dictionary
from src.nmf.nnls import solve_nnls_frame, solve_nnls_batch
from src.nmf.activation import compute_h_event_sum
from src.crnn.model import HybridCRNN
from src.crnn.inference import CRNNPredictor
from src.event.binary_to_event import probability_to_binary, group_binary_events
from src.event.counting import count_parts_from_events
import librosa


def profile_offline_pipeline(audio_path: Path):
    """
    Profiles the exact software reference pipeline implemented in sensor.py:
    Audio -> STFT -> Log-Compression -> NNLS Batch -> H_event_sum
          -> Log-Mel -> CRNNPredictor (Window=128, Hop=64) -> Threshold -> Group -> Count
    """
    print("\n" + "=" * 80)
    print("      SCENARIO A: REFERENCE OFFLINE PIPELINE PROFILING (sensor.py match)")
    print("=" * 80)

    audio, sr = load_audio(audio_path, target_sr=config.SAMPLE_RATE)
    duration_sec = len(audio) / sr
    w_matrix = load_dictionary(config.W_STANDARD_PATH)
    predictor = CRNNPredictor(config.CRNN_MODEL_PATH, device=torch.device("cpu"))

    # STFT & Log Compression
    t0 = time.perf_counter()
    stft_matrix = compute_stft(audio)
    t_stft = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    v_matrix = log_compression(stft_matrix)
    t_pre = (time.perf_counter() - t0) * 1000.0

    # NNLS Batch
    t0 = time.perf_counter()
    h_matrix = solve_nnls_batch(v_matrix, w_matrix)
    t_nnls = (time.perf_counter() - t0) * 1000.0

    # H_event_sum
    t0 = time.perf_counter()
    h_event_sum = compute_h_event_sum(h_matrix)
    t_h = (time.perf_counter() - t0) * 1000.0

    # Log-Mel
    t0 = time.perf_counter()
    magnitude = np.abs(stft_matrix)
    mel_basis = librosa.filters.mel(
        sr=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        n_mels=config.N_MELS,
        fmin=config.F_MIN,
        fmax=config.F_MAX
    )
    log_mel = np.log1p(np.dot(mel_basis, magnitude)).astype(np.float32)
    t_mel = (time.perf_counter() - t0) * 1000.0

    # CRNN Inference
    t0 = time.perf_counter()
    probs = predictor.predict_probabilities(
        log_mel, 
        h_event_sum, 
        window_len=config.SEQUENCE_LENGTH, 
        hop_size=config.SEQUENCE_HOP
    )
    t_crnn = (time.perf_counter() - t0) * 1000.0

    # Event Processing & Count
    t0 = time.perf_counter()
    binary_seq = probability_to_binary(probs, threshold=config.CRNN_THRESHOLD)
    events = group_binary_events(
        binary_seq, 
        probabilities=probs, 
        min_distance_frames=config.MIN_EVENT_DISTANCE_FRAMES, 
        representative_method=config.EVENT_REPRESENTATIVE_METHOD
    )
    total_count = count_parts_from_events(events)
    t_event = (time.perf_counter() - t0) * 1000.0

    total_time_ms = t_stft + t_pre + t_nnls + t_h + t_mel + t_crnn + t_event
    total_frames = stft_matrix.shape[1]
    time_per_frame_ms = total_time_ms / total_frames
    rtf = (total_time_ms / 1000.0) / duration_sec

    print(f"• Audio Target         : {audio_path.name} ({duration_sec:.2f} s, {total_frames} frames)")
    print(f"• Total Execution Time : {total_time_ms:.2f} ms")
    print(f"• Mean Time per Frame  : {time_per_frame_ms:.3f} ms/frame")
    print(f"• Real-Time Factor     : {rtf:.4f} (Processing is {1.0/rtf:.2f}x faster than real-time on CPU)")
    print(f"• Detected Count       : {total_count} parts")
    print("-" * 80)
    print(f"| {'Stage':<20} | {'Total Time (ms)':<15} | {'Mean Time/Frame (ms)':<20} | {'Ratio (%)':<10} |")
    print("-" * 80)
    stages = [
        ("STFT", t_stft),
        ("Log-Compression", t_pre),
        ("NNLS Batch", t_nnls),
        ("H_event_sum", t_h),
        ("Log-Mel Basis", t_mel),
        ("CRNN (Predictor)", t_crnn),
        ("Event Grouping", t_event),
    ]
    for name, t_stage in stages:
        print(f"| {name:<20} | {t_stage:15.2f} | {t_stage/total_frames:20.4f} | {(t_stage/total_time_ms)*100:9.1f}% |")
    print("=" * 80)


def profile_streaming_feasibility(audio_path: Path):
    """
    Profiles frame-by-frame latency distribution, NNLS convergence error, 
    and peak memory usage to evaluate MCU real-time scheduling feasibility.
    Real-time Frame Budget = Hop / Fs = 1024 / 44100 = 23.22 ms.
    """
    print("\n" + "=" * 80)
    print("      SCENARIO B: MCU DEPLOYMENT FEASIBILITY PROFILING (Per-frame Streaming)")
    print("=" * 80)

    audio, sr = load_audio(audio_path, target_sr=config.SAMPLE_RATE)
    w_matrix = load_dictionary(config.W_STANDARD_PATH)
    device = torch.device("cpu")
    
    model = HybridCRNN().to(device)
    model.load_state_dict(torch.load(config.CRNN_MODEL_PATH, map_location=device))
    model.eval()

    n_fft = config.N_FFT
    hop_length = config.HOP_LENGTH
    t_budget_ms = (hop_length / sr) * 1000.0  # 23.22 ms
    num_frames = (len(audio) - n_fft) // hop_length + 1
    hann_window = scipy.signal.windows.hann(n_fft, sym=False).astype(np.float32)

    mel_basis = librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=config.N_MELS,
        fmin=config.F_MIN,
        fmax=config.F_MAX
    ).astype(np.float32)

    # Circular buffers
    seq_len = config.SEQUENCE_LENGTH
    mel_buffer = np.zeros((config.N_MELS, seq_len), dtype=np.float32)
    he_buffer = np.zeros((1, seq_len), dtype=np.float32)

    t_stft_list = np.zeros(num_frames)
    t_pre_list = np.zeros(num_frames)
    t_nnls_list = np.zeros(num_frames)
    t_h_list = np.zeros(num_frames)
    t_mel_list = np.zeros(num_frames)
    t_crnn_list = np.zeros(num_frames)
    t_event_list = np.zeros(num_frames)
    recon_error_list = np.zeros(num_frames)

    prob_history = []

    with torch.no_grad():
        for t in range(num_frames):
            frame_audio = audio[t * hop_length : t * hop_length + n_fft]

            # 1. T_STFT
            t0 = time.perf_counter()
            spectrum = compute_frame_stft(frame_audio, n_fft=n_fft, window=hann_window)
            t_stft_list[t] = (time.perf_counter() - t0) * 1000.0

            # 2. T_PRE
            t0 = time.perf_counter()
            v_frame = log_compression(spectrum)
            t_pre_list[t] = (time.perf_counter() - t0) * 1000.0

            # 3. T_NNLS
            t0 = time.perf_counter()
            h_t = solve_nnls_frame(v_frame, w_matrix)
            t_nnls_list[t] = (time.perf_counter() - t0) * 1000.0

            # Calculate Reconstruction Error outside latency metrics
            v_rec = np.dot(w_matrix, h_t)[:, np.newaxis]
            recon_error_list[t] = np.linalg.norm(v_frame - v_rec)

            # 4. T_H
            t0 = time.perf_counter()
            h_event_val = compute_h_event_sum(h_t[:config.EVENT_COMPONENTS])
            t_h_list[t] = (time.perf_counter() - t0) * 1000.0

            # 5. T_MEL
            t0 = time.perf_counter()
            mag_linear = np.abs(spectrum).squeeze()
            mel_frame = np.log1p(np.dot(mel_basis, mag_linear))
            
            mel_buffer[:, :-1] = mel_buffer[:, 1:]
            mel_buffer[:, -1] = mel_frame
            he_buffer[:, :-1] = he_buffer[:, 1:]
            he_buffer[:, -1] = h_event_val
            t_mel_list[t] = (time.perf_counter() - t0) * 1000.0

            # 6. T_CRNN (Single-step forward over window context)
            t0 = time.perf_counter()
            x_mel_t = torch.from_numpy(mel_buffer).unsqueeze(0).to(device)
            x_he_t = torch.from_numpy(he_buffer).unsqueeze(0).to(device)
            p_seq = model.predict_probability(x_mel_t, x_he_t)
            curr_p = p_seq[0, -1].item()
            t_crnn_list[t] = (time.perf_counter() - t0) * 1000.0
            prob_history.append(curr_p)

            # 7. T_EVENT (State update)
            t0 = time.perf_counter()
            _ = 1 if curr_p >= config.CRNN_THRESHOLD else 0
            t_event_list[t] = (time.perf_counter() - t0) * 1000.0

    t_total_list = (
        t_stft_list + t_pre_list + t_nnls_list + 
        t_h_list + t_mel_list + t_crnn_list + t_event_list
    )

    def print_stat_row(name: str, arr: np.ndarray):
        print(f"| {name:<12} | {np.mean(arr):7.3f} | {np.std(arr):7.3f} | {np.median(arr):7.3f} | {np.min(arr):7.3f} | {np.max(arr):7.3f} | {np.percentile(arr, 95):7.3f} | {np.percentile(arr, 99):7.3f} |")

    print(f"| {'Module':<12} | {'Mean':<7} | {'Std':<7} | {'Median':<7} | {'Min':<7} | {'Max':<7} | {'P95':<7} | {'P99':<7} |")
    print("-" * 80)
    print_stat_row("STFT", t_stft_list)
    print_stat_row("Log-Compress", t_pre_list)
    print_stat_row("NNLS", t_nnls_list)
    print_stat_row("H_event_sum", t_h_list)
    print_stat_row("Log-Mel", t_mel_list)
    print_stat_row("CRNN Step", t_crnn_list)
    print_stat_row("Threshold", t_event_list)
    print("-" * 80)
    print_stat_row("TOTAL", t_total_list)
    print("=" * 80)

    # Quantitative Bottleneck Analysis
    mean_total = np.mean(t_total_list)
    mean_nnls = np.mean(t_nnls_list)
    mean_crnn = np.mean(t_crnn_list)

    print("\n" + "=" * 50)
    print("       HARDWARE RESOURCE ESTIMATION & BOTTLENECK    ")
    print("=" * 50)
    print(f"• Real-time Budget : {t_budget_ms:.2f} ms")
    print(f"• PC Mean Latency  : {mean_total:.3f} ms/frame ({(mean_total / t_budget_ms) * 100:.1f}% budget)")
    print(f"• Worst-Case Frame : {np.max(t_total_list):.3f} ms")
    print(f"• NNLS Ratio       : {(mean_nnls / mean_total) * 100:.1f}% of total compute")
    print(f"• CRNN Ratio       : {(mean_crnn / mean_total) * 100:.1f}% of total compute")
    print(f"• Mean Recon Error : {np.mean(recon_error_list):.4f} (Std: {np.std(recon_error_list):.4f})")

    # Static Weights Footprint
    w_bytes = w_matrix.nbytes
    crnn_params = sum(p.numel() for p in model.parameters())
    crnn_bytes = crnn_params * 4
    print("\n[*] Static Storage Requirement (Parameters only):")
    print(f"  - Dictionary W (1025x48 FP32) : {w_bytes / 1024:.2f} KB")
    print(f"  - CRNN Model ({crnn_params} params, FP32) : {crnn_bytes / 1024:.2f} KB")
    print(f"  - Total Weights Flash Footprint : {(w_bytes + crnn_bytes) / 1024:.2f} KB")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile Acoustic Virtual Sensor Runtime")
    parser.add_argument("--file", type=str, default=None, help="Path to WAV file")
    args = parser.parse_args()

    target_path = Path(args.file) if args.file else config.TEST_AUDIO_DIR / config.DEFAULT_TEST_WAV
    if not target_path.is_file():
        candidates = list(config.TEST_AUDIO_DIR.glob("*.wav"))
        if candidates:
            target_path = candidates[0]
        else:
            raise FileNotFoundError(f"No WAV file found at {config.TEST_AUDIO_DIR}")

    profile_offline_pipeline(target_path)
    profile_streaming_feasibility(target_path)