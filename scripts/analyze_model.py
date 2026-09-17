"""
===============================================================================
Script: Static Model & Hardware Specification Profiler (scripts/analyze_model.py)
===============================================================================

Extracts exact model specifications, storage requirements, computational workload,
and dataset temporal characteristics for NMF + Peak Detection pipeline.
Does NOT run runtime latency benchmarks; provides structural constraints for MCU selection.
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
        "avg_duration": float(np.mean(durations)) if durations else 0.0,
        "min_duration": float(np.min(durations)) if durations else 0.0,
        "max_duration": float(np.max(durations)) if durations else 0.0,
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
        "avg_events": float(np.mean(events_per_file)) if events_per_file else 0.0,
        "min_interval": float(np.min(intervals_sec)) if intervals_sec else 0.0,
        "max_interval": float(np.max(intervals_sec)) if intervals_sec else 0.0,
        "mean_interval": float(np.mean(intervals_sec)) if intervals_sec else 0.0,
        "median_interval": float(np.median(intervals_sec)) if intervals_sec else 0.0,
        "std_interval": float(np.std(intervals_sec)) if intervals_sec else 0.0,
        "intervals": intervals_sec
    }


def run_analysis():
    print("=" * 80)
    print("      ACOUSTIC VIRTUAL SENSOR - MODEL & HARDWARE SPECIFICATION")
    print("=" * 80)

    # 1. AUDIO & SPECTRAL CONFIGURATION
    sr = config.SAMPLE_RATE
    n_fft = config.N_FFT
    hop = config.HOP_LENGTH
    n_freq = n_fft // 2 + 1
    t_fft_window_ms = (n_fft / float(sr)) * 1000.0
    t_frame_interval_ms = (hop / float(sr)) * 1000.0

    print("\n[1. AUDIO & SPECTRAL CONFIGURATION (CALCULATED / SYSTEM SPEC)]")
    print(f"  Sampling rate (fs)                : {sr} Hz")
    print(f"  FFT length (N_FFT)                : {n_fft} samples")
    print(f"  FFT window duration               : {t_fft_window_ms:.2f} ms")
    print(f"  Hop length (H)                    : {hop} samples")
    print(f"  Frame processing interval (Budget): {t_frame_interval_ms:.2f} ms (Periodic arrival interval)")
    print(f"  Window function                   : {getattr(config, 'WINDOW_FUNCTION', 'hann')} (center={getattr(config, 'CENTER_STFT', False)})")
    print(f"  Frequency bins (F)                : {n_freq} bins")
    print(f"  Frequency range                   : 0.0 - {sr / 2.0:.1f} Hz")

    # 2. NMF MODEL SPECIFICATION
    w_path = config.W_STANDARD_PATH
    if not w_path.is_file():
        print(f"\n[!] Không tìm thấy file từ điển W tại: {w_path}")
        return

    W = np.load(str(w_path))
    w_shape = W.shape
    w_dtype = W.dtype
    w_bytes = W.nbytes
    w_kib = w_bytes / 1024.0

    min_val, max_val = float(W.min()), float(W.max())
    mean_val, std_val = float(W.mean()), float(W.std())
    nan_count = int(np.isnan(W).sum())
    inf_count = int(np.isinf(W).sum())

    # Xử lý Dynamic Range chuẩn xác trên các phần tử dương (W > 0)
    positive_mask = W > 0
    zero_count = int((W == 0).sum())
    zero_ratio = (zero_count / float(W.size)) * 100.0

    if np.any(positive_mask):
        min_positive = float(W[positive_mask].min())
        dynamic_range_db = 20.0 * np.log10(max_val / min_positive)
    else:
        min_positive = 0.0
        dynamic_range_db = 0.0

    print("\n[2. NMF DICTIONARY SPECIFICATION (MEASURED & CALCULATED)]")
    print(f"  Dictionary file                   : {w_path.name}")
    print(f"  Matrix shape (F x C)              : {w_shape} (Freq Bins x Components)")
    print(f"  Total parameters                  : {W.size:,} floats")
    print(f"  Dtype (FP32 storage)              : {w_dtype} ({w_bytes:,} bytes / {w_kib:.2f} KiB) [MEASURED]")
    print(f"  FP16 storage potential            : {w_bytes // 2:,} bytes ({w_kib / 2.0:.2f} KiB) [TO BE VALIDATED]")
    print(f"  INT16 storage potential           : {w_bytes // 2:,} bytes ({w_kib / 2.0:.2f} KiB) [TO BE VALIDATED]")
    print(f"  INT8 storage potential            : {w_bytes // 4:,} bytes ({w_kib / 4.0:.2f} KiB) [TO BE VALIDATED]")
    print(f"  Absolute Min / Max                : [{min_val:.5e}, {max_val:.5e}]")
    print(f"  Positive Minimum (W_min+)         : {min_positive:.5e}")
    print(f"  Dynamic Range of Non-zeros (DR)   : ~{dynamic_range_db:.2f} dB (20*log10(W_max / W_min+))")
    print(f"  Zero coefficient ratio (Sparsity) : {zero_ratio:.2f}% ({zero_count:,}/{W.size:,} elements)")
    print(f"  Distribution stats                : Mean={mean_val:.5e} | Std={std_val:.5e}")
    print(f"  Integrity check                   : NaN={nan_count} | Inf={inf_count}")
    print(f"  Component breakdown               : Total={config.TOTAL_COMPONENTS} (Event={config.EVENT_COMPONENTS}, Bowl={config.BOWL_COMPONENTS}, Env={config.ENV_COMPONENTS})")

    # 3. PEAK DETECTOR SPECIFICATION
    peak_json = config.PEAK_PARAMS_PATH
    prominence = getattr(config, "DEFAULT_PEAK_PROMINENCE", 0.01)
    distance = getattr(config, "DEFAULT_PEAK_DISTANCE_FRAMES", 4)
    mae_train = None
    num_train_files_used = None

    if peak_json.is_file():
        try:
            with open(peak_json, "r", encoding="utf-8") as f:
                p_data = json.load(f)
                prominence = p_data.get("prominence", prominence)
                distance = p_data.get("distance", distance)
                mae_train = p_data.get("mae", None)
                # Đọc chuẩn key total_train_files từ peak_params.json
                num_train_files_used = p_data.get("total_train_files", None)
        except Exception:
            pass

    min_event_distance_ms = distance * t_frame_interval_ms

    print("\n[3. PEAK DETECTOR SPECIFICATION (CONFIGURATION & TRAIN PERFORMANCE)]")
    print(f"  Configuration source              : {peak_json.name if peak_json.is_file() else 'Config Defaults'}")
    print(f"  Prominence threshold (P*)         : {prominence} [FROZEN CONFIG]")
    print(f"  Minimum distance (D*)             : {distance} frames ({min_event_distance_ms:.2f} ms) [FROZEN CONFIG]")
    if mae_train is not None:
        print(f"  Training Count MAE                : {mae_train:.2f} parts [MEASURED]")

    # 4. DATASET & TEMPORAL CHARACTERISTICS
    train_data = analyze_audio_dataset(config.TRAIN_AUDIO_DIR, config.TRAIN_AUDIO_DIR / "label.txt")
    test_data = analyze_audio_dataset(config.TEST_AUDIO_DIR, config.TEST_AUDIO_DIR / "label.txt")
    annot_data = analyze_annotations(config.TRAIN_ANNOTATION_DIR)

    print("\n[4. DATASET & EVENT CHARACTERISTICS (MEASURED & CALCULATED)]")
    print(f"  Training Audio files (Train_*.wav): {train_data['count']} files")
    print(f"  Testing Audio files (Test_*.wav)  : {test_data['count']} files")
    print(f"  Total Duration                    : Train={train_data['total_duration']:.2f} s | Test={test_data['total_duration']:.2f} s")
    print(f"  Average File Duration             : Train={train_data['avg_duration']:.2f} s | Test={test_data['avg_duration']:.2f} s")
    print(f"  Min / Max File Duration           : [{train_data['min_duration']:.2f}s, {train_data['max_duration']:.2f}s]")
    print(f"  Total Workpieces (label.txt)      : Train={train_data['total_workpieces']} parts | Test={test_data['total_workpieces']} parts")

    print(f"\n  Dataset Provenance & Verification Status:")
    print(f"    - Audio training files count    : {train_data['count']}")
    print(f"    - Frame-annotated files count   : {annot_data['files_count']} ({annot_data['total_events']} confirmed events)")
    print(f"    - Peak-parameter source files   : {num_train_files_used if num_train_files_used is not None else 'Unrecorded'}")

    if num_train_files_used is not None and annot_data['files_count'] > 0 and num_train_files_used != annot_data['files_count']:
        print(f"    [!] WARNING: The number of files used for peak-parameter optimization ({num_train_files_used})")
        print(f"                 does not match the reported frame-annotated files ({annot_data['files_count']}).")
        print(f"                 Please verify your dataset provenance split prior to thesis defense.")

    if annot_data["files_count"] > 0 and annot_data["intervals"]:
        min_dt = annot_data["min_interval"]
        max_rate = 1.0 / min_dt if min_dt > 0 else 0.0
        print(f"\n  Temporal Inter-event Ground-Truth (Observed Physics):")
        print(f"    - Minimum observed gap (Δt_min)  : {min_dt * 1000.0:.2f} ms")
        print(f"    - Mean event gap                 : {annot_data['mean_interval'] * 1000.0:.2f} ms")
        print(f"    - Median event gap               : {annot_data['median_interval'] * 1000.0:.2f} ms")
        print(f"    - Std event gap                  : {annot_data['std_interval'] * 1000.0:.2f} ms")
        print(f"    - Max event gap                  : {annot_data['max_interval']:.2f} s")
        print(f"    - Physical Maximum Feed Rate     : {max_rate:.2f} events/s (R_max = 1 / Δt_min)")
        print(f"    - Detector Distance Check        : D* * (H/fs) = {min_event_distance_ms:.2f} ms ≈ Δt_min ({min_dt * 1000.0:.2f} ms)")

    # 5. COMPUTATIONAL WORKLOAD PER FRAME
    w_t_v_macs = n_freq * config.TOTAL_COMPONENTS

    print("\n[5. COMPUTATIONAL WORKLOAD PER FRAME (CALCULATED & SPECIFICATION)]")
    print(f"  1. STFT (Real FFT 2048 pts)       : O(N log N)")
    print(f"  2. Log Compression (log1p)        : O(F) ({n_freq:,} scalar operations)")
    print(f"  3. Matrix Projection (W^T * V_t)  : O(F x C) ({w_t_v_macs:,} MACs / {w_t_v_macs * 2:,} FLOPs) [CALCULATED]")
    print(f"  4. NNLS Solver (W: 1025 x 48)     : Iterative Active-Set (Complexity runtime & iteration dependent)")
    print(f"  5. Event Sum Accumulation (H_e)   : O(K_event) (24 additions)")
    print(f"  6. Peak Detection (State Machine) : O(1) state transitions per frame")

    # 6. ESTIMATED MEMORY FOOTPRINT ON MCU
    flash_w_measured = w_bytes
    flash_hann_table_est = n_fft * 4  # 8 KiB
    flash_firmware_est = 40 * 1024    # ~40 KiB
    total_flash_est = flash_w_measured + flash_hann_table_est + flash_firmware_est

    # Peak RAM workspace model
    ram_audio_ring = n_fft * 4        # 2048 float32 = 8 KiB
    ram_rfft_out = n_freq * 8         # 1025 complex64 = 8.2 KiB
    ram_v_frame = n_freq * 4          # 1025 float32 = 4.1 KiB
    ram_h_frame = config.TOTAL_COMPONENTS * 4  # 48 float32 = 192 bytes
    ram_nnls_workspace = (config.TOTAL_COMPONENTS ** 2 + config.TOTAL_COMPONENTS * 4) * 4  # ~10 KiB
    ram_peak_state = 64               # Struct FSM
    ram_stack_rtos = 8 * 1024         # ~8 KiB

    total_live_ram_est = (ram_audio_ring + ram_rfft_out + ram_v_frame + 
                          ram_h_frame + ram_nnls_workspace + ram_peak_state + ram_stack_rtos)
    ram_with_margin_est = total_live_ram_est * 1.20

    print("\n[6. MEMORY FOOTPRINT MODEL (MEASURED vs ESTIMATED)]")
    print(f"  STATIC FLASH / ROM STORAGE:")
    print(f"    - W Matrix (FP32) [MEASURED]    : {flash_w_measured:,} bytes ({flash_w_measured / 1024.0:.2f} KiB)")
    print(f"    - Hann Window Table [ESTIMATED] : ~{flash_hann_table_est / 1024.0:.2f} KiB")
    print(f"    - Firmware Code [ESTIMATED]     : ~{flash_firmware_est / 1024.0:.2f} KiB")
    print(f"    => Preliminary Flash Estimate   : ~{total_flash_est / 1024.0:.2f} KiB")

    print(f"\n  DYNAMIC SRAM WORKSPACE MODEL (One-Frame-At-A-Time Streaming):")
    print(f"    - Audio Ring Buffer (DMA/I2S)   : {ram_audio_ring / 1024.0:.2f} KiB")
    print(f"    - RFFT Complex Output (X_t)     : {ram_rfft_out / 1024.0:.2f} KiB")
    print(f"    - Log-Magnitude Spectrum (V_t)  : {ram_v_frame / 1024.0:.2f} KiB")
    print(f"    - Activation Vector (h_t)       : {ram_h_frame} bytes")
    print(f"    - NNLS Workspace (Gram + temp)  : ~{ram_nnls_workspace / 1024.0:.2f} KiB")
    print(f"    - Peak Detector FSM State       : <0.1 KiB")
    print(f"    - Stack & System Margin         : ~{ram_stack_rtos / 1024.0:.2f} KiB")
    print(f"    => Estimated Peak Runtime SRAM  : ~{total_live_ram_est / 1024.0:.2f} KiB")
    print(f"    => Engineering SRAM Target (+20%: ~{ram_with_margin_est / 1024.0:.2f} KiB (*Note: Engineering margin, not algorithm requirement)")

    # 7. HARDWARE REQUIREMENT SUMMARY
    print("\n" + "=" * 80)
    print("                      HARDWARE REQUIREMENT SUMMARY")
    print("=" * 80)
    print(f"  Frame Interval Budget             : {t_frame_interval_ms:.2f} ms")
    print(f"  Estimated Flash (FP32)            : ~{total_flash_est / 1024.0:.1f} KiB (Preliminary Estimate)")
    print(f"  Estimated SRAM Required           : ~{total_live_ram_est / 1024.0:.1f} KiB (~{ram_with_margin_est / 1024.0:.1f} KiB target)")
    print(f"  Dictionary Precision              : FP32 ({w_kib:.1f} KiB) | FP16/INT16/INT8 (Numerical Validation: Pending)")
    print(f"  Dominant Computation in Reference : NNLS (Iterative Solver)")
    print(f"  Projection Workload               : {w_t_v_macs:,} MAC/frame")
    print(f"  Required Arithmetic Support       : Floating-Point (FPU recommended / strongly preferred)")
    print(f"  DSP / SIMD Instructions           : Recommended for Real-Time Headroom")
    print(f"  MCU Hardware Selection Status     : Candidate comparison pending benchmark")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_analysis()