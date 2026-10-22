#!/usr/bin/env python3
"""Compare native C++ pipeline outputs to the repository's Python reference.

This script does not retrain NMF. It compiles the frozen W_standard.npy into a
C header, runs the host harness, and reports Stage-1 numerical differences.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.audio.loader import load_audio
from src.nmf.activation import compute_h_event_sum
from src.nmf.dictionary import load_dictionary
from src.nmf.nnls import solve_nnls_frame
from src.preprocessing.log_compression import log_compression
from src.preprocessing.stft import compute_frame_stft


def build_host(binary_dir: Path) -> Path:
    subprocess.run([
        sys.executable, str(ROOT / "embedded/tools/export_dictionary.py"),
        "--input", str(config.W_STANDARD_PATH),
        "--output", str(ROOT / "embedded/src/w_standard_data.h")
    ], check=True)
    subprocess.run(["cmake", "-S", str(ROOT / "embedded"), "-B", str(binary_dir),
                    "-DCMAKE_BUILD_TYPE=Release"], check=True)
    subprocess.run(["cmake", "--build", str(binary_dir), "--config", "Release"], check=True)
    candidates = [
        binary_dir / "Release" / "avs_infer.exe",
        binary_dir / "avs_infer.exe",
        binary_dir / "avs_infer",
        binary_dir / "Release" / "avs_infer",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Built avs_infer executable not found. Check the CMake build output.")


def python_reference(audio: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_fft, hop = config.N_FFT, config.HOP_LENGTH
    num_frames = (len(audio) - n_fft) // hop + 1
    h_all = np.zeros((num_frames, w.shape[1]), dtype=np.float32)
    he = np.zeros(num_frames, dtype=np.float32)
    for t in range(num_frames):
        frame = np.array(audio[t * hop:t * hop + n_fft], dtype=np.float32, copy=True)
        frame -= np.mean(frame, dtype=np.float32)
        spectrum = compute_frame_stft(frame)
        v = log_compression(spectrum).reshape(-1)
        h = solve_nnls_frame(v, w, max_iter=config.NNLS_MAX_ITER)
        h_all[t] = h
        he[t] = compute_h_event_sum(h)
    with open(config.PEAK_PARAMS_PATH, "r", encoding="utf-8") as f:
        import json
        p = json.load(f)
    peaks, _ = find_peaks(he, prominence=float(p["prominence"]), distance=int(p["distance"]))
    return np.column_stack((np.arange(num_frames), he, h_all)), peaks.astype(np.int64)


def read_peak_csv(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return np.asarray([int(row["peak_frame"]) for row in reader], dtype=np.int64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", type=Path, help="WAV at exactly 44.1 kHz")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "embedded/build")
    parser.add_argument("--max-h-abs", type=float, default=0.05,
                        help="Fail if max absolute activation error exceeds this value")
    parser.add_argument("--require-exact-peaks", action="store_true",
                        help="Fail if C++ peak indices differ from SciPy")
    args = parser.parse_args()

    audio, _ = load_audio(args.wav, target_sr=config.SAMPLE_RATE, remove_dc=False)
    w = load_dictionary(config.W_STANDARD_PATH)
    if len(audio) < config.N_FFT:
        raise SystemExit("WAV is shorter than one analysis frame.")

    binary = build_host(args.build_dir)
    scratch = args.build_dir / "verify"
    scratch.mkdir(parents=True, exist_ok=True)
    raw_path = scratch / "audio_float32.bin"
    out_path = scratch / "cpp_output.csv"
    np.asarray(audio, dtype=np.float32).tofile(raw_path)

    completed = subprocess.run([str(binary), str(raw_path), str(out_path)],
                               text=True, capture_output=True)
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        return completed.returncode
    print(completed.stdout)

    cpp = np.loadtxt(out_path, delimiter=",", skiprows=1, dtype=np.float64)
    cpp_peaks = read_peak_csv(Path(str(out_path) + ".peaks.csv"))
    ref, ref_peaks = python_reference(audio, w)
    if cpp.ndim == 1:
        cpp = cpp.reshape(1, -1)
    if cpp.shape != ref.shape:
        print(f"FAIL: shape mismatch C++={cpp.shape}, Python={ref.shape}")
        return 20

    abs_error = np.abs(cpp[:, 2:] - ref[:, 2:])
    he_error = np.abs(cpp[:, 1] - ref[:, 1])
    max_h = float(abs_error.max()) if abs_error.size else 0.0
    rmse_h = float(np.sqrt(np.mean(np.square(cpp[:, 2:] - ref[:, 2:]))))
    max_he = float(he_error.max()) if he_error.size else 0.0
    rmse_he = float(np.sqrt(np.mean(np.square(cpp[:, 1] - ref[:, 1]))))

    print("\n=== Stage 1 numeric comparison ===")
    print(f"Frames                  : {len(ref)}")
    print(f"Max |H_cpp - H_py|      : {max_h:.8g}")
    print(f"RMSE(H)                 : {rmse_h:.8g}")
    print(f"Max |h_event_cpp-py|    : {max_he:.8g}")
    print(f"RMSE(h_event)           : {rmse_he:.8g}")
    print(f"SciPy peak count        : {len(ref_peaks)}")
    print(f"C++ peak count          : {len(cpp_peaks)}")
    print(f"Peak indices identical  : {np.array_equal(cpp_peaks, ref_peaks)}")
    if not np.array_equal(cpp_peaks, ref_peaks):
        print(f"SciPy only (first 20)   : {np.setdiff1d(ref_peaks, cpp_peaks)[:20].tolist()}")
        print(f"C++ only (first 20)     : {np.setdiff1d(cpp_peaks, ref_peaks)[:20].tolist()}")

    failed = max_h > args.max_h_abs
    if args.require_exact_peaks and not np.array_equal(cpp_peaks, ref_peaks):
        failed = True
    if failed:
        print("\nRESULT: FAIL — numerical tolerance not met.")
        return 21
    print("\nRESULT: numerical activation threshold passed.")
    if not np.array_equal(cpp_peaks, ref_peaks):
        print("NOTE: peak picking differs from SciPy on this file; inspect local maxima, prominence bases, and distance ties.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
