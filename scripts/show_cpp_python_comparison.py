"""Compare native embedded C++ and the Python reference, then plot the results.

This is a new helper. It calls the existing verifier without changing it.
Run from the repository root:

    python scripts/show_cpp_python_comparison.py

The default input is data/train/Train_0.wav. The script saves a PNG report and
opens the figure. Use --wav, --binary, --build-dir, --output, or --no-show to
override those defaults.
"""

from __future__ import annotations

import argparse
import os
import runpy
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WAV = ROOT / "data" / "train" / "Train_0.wav"
DEFAULT_BUILD_DIR = ROOT / "embedded" / "build" / "train0-comparison"
DEFAULT_BINARY = ROOT / "embedded" / "build" / "ninja" / "avs_infer.exe"
DEFAULT_OUTPUT = ROOT / "outputs" / "figures" / "train0_cpp_python_comparison.png"
VERIFIER = ROOT / "embedded" / "tools" / "verify_against_python.py"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show a frame-by-frame comparison of C++ and Python on Train_0.wav"
    )
    parser.add_argument("--wav", type=Path, default=DEFAULT_WAV, help="44.1 kHz WAV input")
    parser.add_argument("--binary", type=Path, default=None,
                        help="Existing avs_infer executable; otherwise use the standard build if present")
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR,
                        help="Directory for verifier build and temporary CSV files")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="PNG path for the comparison figure")
    parser.add_argument("--no-show", action="store_true", help="Save the PNG without opening a window")
    args = parser.parse_args()

    wav_path = args.wav.expanduser().resolve()
    build_dir = args.build_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not wav_path.is_file():
        parser.error(f"WAV not found: {wav_path}")
    if not VERIFIER.is_file():
        parser.error(f"Existing comparison helper not found: {VERIFIER}")

    binary = args.binary.expanduser().resolve() if args.binary else None
    if binary is None and DEFAULT_BINARY.is_file():
        binary = DEFAULT_BINARY

    # Keep paths passed to the native C++ executable ASCII and relative. On
    # Windows, the host firmware test uses narrow C/C++ file APIs; an absolute
    # project path containing Vietnamese characters can be converted to "?"
    # before the executable opens its input file. An ASCII junction is already
    # used by the embedded build scripts when available.
    run_cwd = ROOT
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        ascii_root = Path(local_app_data) / "AVSRepoLink"
        try:
            if ascii_root.exists() and ascii_root.resolve() == ROOT.resolve():
                run_cwd = ascii_root
        except OSError:
            pass

    def cli_path(path: Path) -> str:
        try:
            return path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            return str(path.resolve())

    command = [sys.executable, str(VERIFIER), cli_path(wav_path),
               "--build-dir", cli_path(build_dir), "--report-only"]
    if binary is not None:
        command.extend(["--binary", str(binary)])

    print("Running the existing C++/Python verifier (source file unchanged)…", flush=True)
    result = subprocess.run(command, cwd=run_cwd, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if result.returncode != 0:
        return result.returncode

    # Reuse the exact Python reference implementation from the existing verifier.
    verifier_namespace = runpy.run_path(str(VERIFIER))
    import config
    from src.audio.loader import load_audio
    from src.nmf.dictionary import load_dictionary

    audio, _ = load_audio(wav_path, target_sr=config.SAMPLE_RATE, remove_dc=False)
    dictionary = load_dictionary(config.W_STANDARD_PATH)
    reference, python_peaks, reference_spectrum = verifier_namespace["python_reference"](audio, dictionary)
    read_peak_csv = verifier_namespace["read_peak_csv"]

    output_csv = build_dir / "verify" / "cpp_output.csv"
    cpp = np.loadtxt(output_csv, delimiter=",", skiprows=1, dtype=np.float64)
    if cpp.ndim == 1:
        cpp = cpp.reshape(1, -1)
    cpp_peaks = read_peak_csv(Path(str(output_csv) + ".peaks.csv"))
    cpp_spectrum = np.loadtxt(Path(str(output_csv) + ".spectrum.csv"),
                              delimiter=",", skiprows=1, dtype=np.float64)
    if cpp_spectrum.ndim == 1:
        cpp_spectrum = cpp_spectrum.reshape(1, -1)
    spectrum_error = np.abs(cpp_spectrum[:, 1:] - reference_spectrum)

    if cpp.shape != reference.shape:
        raise ValueError(f"Frame/output mismatch: C++ {cpp.shape}, Python {reference.shape}")
    h_error = np.abs(cpp[:, 2:] - reference[:, 2:])
    event_error = np.abs(cpp[:, 1] - reference[:, 1])
    frame_times = np.arange(len(reference)) * config.HOP_LENGTH / config.SAMPLE_RATE
    event_rmse = float(np.sqrt(np.mean(np.square(event_error))))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_event, ax_components, ax_event_error) = plt.subplots(
        3, 1, figsize=(13, 9), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1, 1]},
    )
    fig.suptitle(f"Train_0.wav — C++ host và Python reference ({len(reference)} frames)", fontsize=15)

    ax_event.plot(frame_times, reference[:, 1], color="#1769aa", linewidth=1.2,
                  label="Python H_event_sum")
    ax_event.plot(frame_times, cpp[:, 1], color="#e07a19", linewidth=1,
                  linestyle="--", label="C++ H_event_sum")
    ax_event.scatter(frame_times[python_peaks], reference[python_peaks, 1],
                     marker="o", s=18, facecolors="none", edgecolors="#1769aa",
                     label=f"Python peaks ({len(python_peaks)})", zorder=3)
    ax_event.scatter(frame_times[cpp_peaks], cpp[cpp_peaks, 1],
                     marker="x", s=22, color="#e07a19",
                     label=f"C++ peaks ({len(cpp_peaks)})", zorder=3)
    ax_event.set_ylabel("H_event_sum")
    ax_event.set_title("Event envelope và vị trí đỉnh")
    ax_event.legend(loc="upper right", ncols=2)
    ax_event.grid(alpha=0.25)

    ax_components.plot(frame_times, np.max(h_error, axis=1), color="#7d4ca5", linewidth=1)
    ax_components.set_ylabel("Max |ΔH|")
    ax_components.set_title("Sai số activation lớn nhất trong 48 thành phần tại mỗi frame")
    ax_components.grid(alpha=0.25)

    ax_event_error.plot(frame_times, event_error, color="#27834a", linewidth=1)
    ax_event_error.set_ylabel("|ΔH_event_sum|")
    ax_event_error.set_xlabel("Thời gian (s)")
    ax_event_error.set_title("Sai số tuyệt đối của H_event_sum")
    ax_event_error.grid(alpha=0.25)

    peaks_identical = np.array_equal(cpp_peaks, python_peaks)
    summary = (
        f"H_event_sum: max error {event_error.max():.3g}, RMSE {event_rmse:.3g}\n"
        f"All activations: max error {h_error.max():.3g}\n"
        f"Log spectrum: max error {spectrum_error.max():.3g}\n"
        f"Peaks identical: {peaks_identical} ({len(cpp_peaks)}/{len(python_peaks)} matched)"
    )
    ax_event.text(0.01, 0.97, summary, transform=ax_event.transAxes,
                  va="top", ha="left", fontsize=9,
                  bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.85})

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    # Windows terminals using a legacy code page may not be able to print the
    # Vietnamese characters in this repository's absolute path.
    printable_output_path = output_path.as_posix().encode("ascii", "backslashreplace").decode("ascii")
    print(f"Comparison plot saved to: {printable_output_path}")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
