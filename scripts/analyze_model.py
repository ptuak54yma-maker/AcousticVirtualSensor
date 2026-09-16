"""
===============================================================================
Script: Static & Dynamic Model Resource Profiler (scripts/analyze_model.py)
===============================================================================

Extracts exact model specifications, memory footprints, computational complexity,
and dataset characteristics for NMF + Peak Detection pipeline.
Does NOT run runtime benchmarks; provides input constraints for MCU selection.
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import config
from src.audio.loader import load_audio
from src.annotation.annotation_io import load_annotation


def parse_label_txt(label_file_path: Path) -> dict:
    ground_truth = {}
    if not label_file_path.is_file():
        return ground_truth
    with open(label_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            parts = line.split(":")
            fname = parts[0].strip()
            if not fname.lower().endswith(".wav"):
                fname = f"{fname}.wav"
            try:
                ground_truth[fname] = int(parts[1].strip())
            except ValueError:
                continue
    return ground_truth


def analyze_audio_dataset(dir_path: Path, label_path: Path):
    wav_files = sorted(list(dir_path.glob("*.wav"))) if dir_path.is_dir() else []
    labels_dict = parse_label_txt(label_path)

    total_duration = 0.0
    durations = []
    total_samples = 0
    sample_rates = set()
    channels = set()

    for p in wav_files:
        audio, sr = load_audio(p, target_sr=config.SAMPLE_RATE)
        d = len(audio) / float(sr)
        durations.append(d)
        total_duration += d
        total_samples += len(audio)
        sample_rates.add(sr)
        channels.add(audio.ndim)

    return {
        "files": wav_files,
        "count": len(wav_files),
        "total_duration": total_duration,
        "avg_duration": np.mean(durations) if durations else 0.0,
        "durations": durations,
        "sample_rates": list(sample_rates),
        "channels": list(channels),
        "total_samples": total_samples,
        "labels": labels_dict,
        "total_workpieces": sum(labels_dict.values()) if labels_dict else 0
    }


def analyze_annotations(dir_path: Path):
    npz_files = sorted(list(dir_path.glob("*_labels.npz"))) if dir_path.is_dir() else []
    total_events = 0
    intervals_sec = []
    events_per_file = []

    hop_sec = config.HOP_LENGTH / float(config.SAMPLE_RATE)

    for p in npz_files:
        data = load_annotation(p)
        binary_labels = data.get("binary_labels", np.array([]))
        event_frames = np.where(binary_labels == 1)[0]
        count = len(event_frames)
        total_events += count
        events_per_file.append(count)

        if count > 1:
            diffs = np.diff(event_frames) * hop_sec
            intervals_sec.extend(diffs.tolist())

    return {
        "files_count": len(npz_files),
        "total_events": total_events,
        "avg_events": np.mean(events_per_file) if events_per_file else 0.0,
        "min_interval": np.min(intervals_sec) if intervals_sec else 0.0,
        "max_interval": np.max(intervals_sec) if intervals_sec else 0.0,
        "mean_interval": np.mean(intervals_sec) if intervals_sec else 0.0,
        "intervals": intervals_sec
    }


def run_analysis():
    print("=" * 65)
    print("      ACOUSTIC VIRTUAL SENSOR - MODEL & HARDWARE SPECIFICATION")
    print("=" * 65)

    # 1. AUDIO & STFT
    sr = config.SAMPLE_RATE
    n_fft = config.N_FFT
    hop = config.HOP_LENGTH
    n_freq = n_fft // 2 + 1
    t_frame_ms = (hop / float(sr)) * 1000.0

    print("\n[1. AUDIO & SPECTRAL CONFIGURATION]")
    print(f"  Sampling rate (fs)         : {sr} Hz")
    print(f"  FFT length (N_FFT)         : {n_fft} samples ({(n_fft / sr) * 1000.0:.2f} ms)")
    print(f"  Hop length (H)             : {hop} samples")
    print(f"  Window function            : {config.WINDOW_FUNCTION} (center={config.CENTER_STFT})")
    print(f"  Frequency bins (F)         : {n_freq} bins (0 - {sr / 2.0:.1f} Hz)")
    print(f"  Frame budget (T_frame)     : {t_frame_ms:.2f} ms (Physical Arrival Interval)")

    # 2. NMF MODEL
    w_path = config.W_STANDARD_PATH
    if not w_path.is_file():
        print(f"\n[!] Không tìm thấy file từ điển W tại: {w_path}")
        return

    W = np.load(str(w_path))
    w_shape = W.shape
    w_dtype = W.dtype
    w_bytes = W.nbytes
    w_kib = w_bytes / 1024.0

    print("\n[2. NMF DICTIONARY SPECIFICATION]")
    print(f"  Dictionary file            : {w_path.name}")
    print(f"  Matrix shape (F x C)       : {w_shape} (Freq Bins x Components)")
    print(f"  Data type (dtype)          : {w_dtype}")
    print(f"  Total parameters           : {W.size:,} floats")
    print(f"  Memory footprint (FP32)    : {w_bytes:,} bytes ({w_kib:.2f} KiB)")
    print(f"  Numerical statistics       : Min={W.min():.5f} | Max={W.max():.5f} | Mean={W.mean():.5f} | Std={W.std():.5f}")
    print(f"  Integrity check            : NaN={np.isnan(W).sum()} | Inf={np.isinf(W).sum()}")
    print(f"  Decomposition components   : Total={config.TOTAL_COMPONENTS} (Event={config.EVENT_COMPONENTS}, Bowl={config.BOWL_COMPONENTS}, Env={config.ENV_COMPONENTS})")

    # 3. PEAK DETECTION PARAMETERS
    peak_json = config.PEAK_PARAMS_PATH
    prominence = config.DEFAULT_PEAK_PROMINENCE
    distance = config.DEFAULT_PEAK_DISTANCE_FRAMES
    mae_train = None

    if peak_json.is_file():
        with open(peak_json, "r", encoding="utf-8") as f:
            p_data = json.load(f)
            prominence = p_data.get("prominence", prominence)
            distance = p_data.get("distance", distance)
            mae_train = p_data.get("mae", None)

    min_event_time_ms = distance * t_frame_ms

    print("\n[3. PEAK DETECTOR CONFIGURATION]")
    print(f"  Parameter source           : {peak_json.name if peak_json.is_file() else 'Default values'}")
    print(f"  Prominence threshold (P*)  : {prominence}")
    print(f"  Min distance (D*)          : {distance} frames ({min_event_time_ms:.2f} ms)")
    if mae_train is not None:
        print(f"  Training Count MAE         : {mae_train:.2f} parts")

    # 4. DATASET ANALYSIS
    train_data = analyze_audio_dataset(config.TRAIN_AUDIO_DIR, config.TRAIN_AUDIO_DIR / "label.txt")
    test_data = analyze_audio_dataset(config.TEST_AUDIO_DIR, config.TEST_AUDIO_DIR / "label.txt")
    annot_data = analyze_annotations(config.TRAIN_ANNOTATION_DIR)

    print("\n[4. DATASET & EVENT CHARACTERISTICS]")
    print(f"  Training set files         : {train_data['count']} files | Total duration: {train_data['total_duration']:.2f} s")
    print(f"  Testing set files          : {test_data['count']} files | Total duration: {test_data['total_duration']:.2f} s")
    print(f"  Total verified workpieces  : Train={train_data['total_workpieces']} | Test={test_data['total_workpieces']}")

    if annot_data["files_count"] > 0 and annot_data["intervals"]:
        min_dt = annot_data["min_interval"]
        max_rate = 1.0 / min_dt if min_dt > 0 else 0.0
        print(f"  Frame annotations available: {annot_data['files_count']} files ({annot_data['total_events']} events)")
        print(f"  Workpiece time gap (Δt)    : Min={min_dt * 1000.0:.1f} ms | Max={annot_data['max_interval']:.2f} s | Mean={annot_data['mean_interval'] * 1000.0:.1f} ms")
        print(f"  Physical Maximum Feed Rate : {max_rate:.2f} events/second")
    else:
        print("  Frame annotations available: No *_labels.npz files detected.")

    # 5. COMPUTATIONAL COMPLEXITY PER FRAME
    stft_macs = n_fft * int(np.log2(n_fft))  # Cooley-Tukey RFFT approx
    log_ops = n_freq
    w_t_v_macs = n_freq * config.TOTAL_COMPONENTS  # W^T * V_t
    hesum_adds = config.EVENT_COMPONENTS

    print("\n[5. COMPUTATIONAL WORKLOAD PER FRAME (Hop = 1024)]")
    print(f"  STFT (Real FFT {n_fft})     : ~{stft_macs:,} FLOPs (O(N log N))")
    print(f"  Log-Magnitude Compression  : {log_ops:,} log1p operations (O(F))")
    print(f"  W^T * V projection         : {w_t_v_macs:,} MACs ({w_t_v_macs * 2:,} FLOPs)")
    print(f"  NNLS Active-Set Solver     : Iterative (Matrix Gram W^T*W = {config.TOTAL_COMPONENTS}x{config.TOTAL_COMPONENTS})")
    print(f"  H_event_sum Accumulation   : {hesum_adds} additions (O(K_event))")
    print(f"  Peak Detector              : O(1) state transitions per frame")

    # 6. ESTIMATED MEMORY FOOTPRINT ON MCU
    # Flash: W constant + Window table + Firmware / Math code
    flash_w_fp32 = w_bytes
    flash_w_fp16 = w_bytes // 2
    flash_window = n_fft * 4  # Hann float32
    flash_firmware_approx = 40 * 1024  # C-runtime CMSIS-DSP, NNLS, state machine

    # Peak RAM (Live simultaneous buffers for 1 frame)
    ram_audio_ring = n_fft * 4  # 2048 float32
    ram_rfft_out = n_freq * 8   # 1025 complex64
    ram_log_v = n_freq * 4      # 1025 float32
    ram_h_vec = config.TOTAL_COMPONENTS * 4  # 48 float32
    ram_nnls_workspace = (config.TOTAL_COMPONENTS ** 2 + config.TOTAL_COMPONENTS * 4) * 4  # Gram + temp
    ram_peak_state = 64         # FSM struct
    ram_stack_rtos = 8 * 1024   # Stack / ISR margin

    total_live_ram = ram_audio_ring + ram_rfft_out + ram_log_v + ram_h_vec + ram_nnls_workspace + ram_peak_state + ram_stack_rtos
    safety_ram = total_live_ram * 1.20

    print("\n[6. ESTIMATED HARDWARE RESOURCE REQUIREMENTS]")
    print(f"  STATIC FLASH / ROM (Constant Storage):")
    print(f"    - W Matrix (FP32)        : {flash_w_fp32 / 1024.0:.2f} KiB")
    print(f"    - W Matrix (FP16 compact): {flash_w_fp16 / 1024.0:.2f} KiB")
    print(f"    - Hann Window Table      : {flash_window / 1024.0:.2f} KiB")
    print(f"    - Firmware Code Estimate : ~{flash_firmware_approx / 1024.0:.2f} KiB")
    print(f"    => Total Flash (FP32)    : ~{(flash_w_fp32 + flash_window + flash_firmware_approx) / 1024.0:.2f} KiB")

    print(f"\n  PEAK DYNAMIC RAM (One-Frame-At-A-Time Streaming):")
    print(f"    - Audio Ring Buffer      : {ram_audio_ring / 1024.0:.2f} KiB")
    print(f"    - RFFT Complex Output    : {ram_rfft_out / 1024.0:.2f} KiB")
    print(f"    - Log-Magnitude Vector Vt: {ram_log_v / 1024.0:.2f} KiB")
    print(f"    - Activation Vector ht   : {ram_h_vec} bytes")
    print(f"    - NNLS Workspace (Gram)  : {ram_nnls_workspace / 1024.0:.2f} KiB")
    print(f"    - Stack & System Margin  : {ram_stack_rtos / 1024.0:.2f} KiB")
    print(f"    => Peak Simultaneous RAM : {total_live_ram / 1024.0:.2f} KiB")
    print(f"    => RAM with +20% Margin  : {safety_ram / 1024.0:.2f} KiB")

    # 7. FEASIBILITY BOUNDS
    print("\n[7. HARDWARE SELECTION CRITERIA]")
    print(f"  1. Timing Budget           : Total frame execution latency T_exec < {t_frame_ms:.2f} ms")
    print(f"  2. Flash Requirement       : Flash_available >= {(flash_w_fp32 + flash_window + flash_firmware_approx) / 1024.0:.1f} KiB (FP32)")
    print(f"  3. SRAM Requirement        : SRAM_available  >= {safety_ram / 1024.0:.1f} KiB")
    print(f"  4. Arithmetic Support      : Single-Precision FPU & DSP extension mandatory")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_analysis()