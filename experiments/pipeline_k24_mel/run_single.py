"""Run only the K=24 Mel model and peak detector on one or all Bowl WAVs."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.pipeline_k24_mel import experiment_config as config
from experiments.pipeline_k24_mel import pipeline


def get_results_dir() -> Path:
    results_dir = config.EXPERIMENT_DIR / "results"
    params_path = results_dir / "peak_params_k24_mel.json"
    if not params_path.is_file():
        raise FileNotFoundError(
            "No K=24 peak parameters found. Run run_comparison.py first."
        )
    return results_dir


def reset_single_inference_dir(results_dir: Path) -> Path:
    target = results_dir / "single_inference"
    expected = (config.EXPERIMENT_DIR / "results" / "single_inference").resolve()
    if target.resolve() != expected or target.is_symlink():
        raise RuntimeError(f"Refusing to clear unexpected inference output path: {target}")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def lookup_true_count(wav_path: Path) -> int | None:
    for label_path in (config.PROJECT_ROOT / "data" / "test" / "label.txt",
                       config.PROJECT_ROOT / "data" / "train" / "label.txt"):
        if not label_path.is_file():
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            name, value = line.split(":", 1)
            name = name.strip()
            if not name.lower().endswith(".wav"):
                name += ".wav"
            if name.lower() == wav_path.name.lower():
                try:
                    return int(value.strip())
                except ValueError:
                    return None
    return None


def run_one(
    wav_path: Path,
    results_dir: Path,
    model: dict[str, object],
    peak_params: dict[str, object],
) -> dict[str, object]:
    wav_path = wav_path.resolve()
    if not wav_path.is_file():
        raise FileNotFoundError(wav_path)
    audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
    if sample_rate != config.SAMPLE_RATE:
        raise ValueError(f"Expected {config.SAMPLE_RATE} Hz audio; got {sample_rate} Hz.")
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono or stereo WAV, got shape {audio.shape}.")

    features = pipeline.extract_features(audio, model)
    envelope = features["H_event_sum"]
    peaks, properties = find_peaks(
        envelope,
        prominence=float(peak_params["prominence"]),
        distance=int(peak_params["distance"]),
    )
    peak_prominences = properties.get("prominences", np.zeros(peaks.size, dtype=np.float32))
    frame_times = np.arange(envelope.size, dtype=np.float32) * (
        config.HOP_LENGTH / config.SAMPLE_RATE
    )
    duration = len(audio) / config.SAMPLE_RATE
    true_count = lookup_true_count(wav_path)
    predicted_count = int(peaks.size)

    output_dir = results_dir / "single_inference" / wav_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "activations.npz",
        H=features["H"],
        H_event_sum=envelope,
        H_bowl_sum=features["H_bowl_sum"],
        H_env_sum=features["H_env_sum"],
        frame_times=frame_times,
    )
    pd.DataFrame({
        "frame": np.arange(envelope.size),
        "time_s": frame_times,
        "H_event_sum": envelope,
    }).to_csv(output_dir / "event_envelope.csv", index=False)
    pd.DataFrame({
        "peak_number": np.arange(1, peaks.size + 1),
        "frame": peaks,
        "time_s": frame_times[peaks],
        "H_event_sum": envelope[peaks],
        "prominence": peak_prominences,
    }).to_csv(output_dir / "peak_results.csv", index=False)

    summary = {
        "input_wav": str(wav_path),
        "output_dir": "experiments/pipeline_k24_mel/results",
        "pipeline": "experimental K=24 Mel-NMF",
        "components_event_bowl_env": [
            config.EVENT_COMPONENTS, config.BOWL_COMPONENTS, config.ENV_COMPONENTS
        ],
        "n_mels": model["n_mels"],
        "mel_input": model["mel_input"],
        "compressor": model["compressor"],
        "compressor_reference": model["reference"],
        "peak_prominence": peak_params["prominence"],
        "peak_distance_frames": peak_params["distance"],
        "duration_s": duration,
        "predicted_count": predicted_count,
        "true_count": true_count,
        "absolute_count_error": abs(predicted_count - true_count) if true_count is not None else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    ax.plot(frame_times, envelope, linewidth=0.8, label="H_event_sum")
    ax.scatter(frame_times[peaks], envelope[peaks], color="crimson", s=22,
               label=f"Detected peaks: {predicted_count}", zorder=3)
    gt_text = (
        f"Ground-truth count (label.txt): {true_count}"
        if true_count is not None else "Ground-truth count: unavailable"
    )
    error_text = (
        f"Absolute count error: {abs(predicted_count - true_count)}"
        if true_count is not None else "Absolute count error: unavailable"
    )
    ax.text(
        0.015, 0.97,
        f"{gt_text}\nPredicted count: {predicted_count}\n{error_text}\n"
        "Ground truth gives count only; no event timestamps.",
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85},
    )
    ax.set(title=f"{wav_path.name} — K=24 Mel peak detection",
           xlabel="Time (s)", ylabel="H_event_sum")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    fig.savefig(output_dir / "peak_detection.png", dpi=150)
    plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run only the experimental K=24 Mel-NMF and peak detector."
    )
    parser.add_argument("wav", type=Path, nargs="?", help="One input WAV at 44.1 kHz.")
    parser.add_argument("--all-bowl", action="store_true",
                        help="Run every Bowl_*.wav under data/test independently (default without WAV).")
    args = parser.parse_args()

    if args.all_bowl and args.wav:
        parser.error("provide either a WAV path or --all-bowl, not both")
    # Running the script without arguments means the common batch workflow:
    # process every test Bowl recording independently.
    if not args.wav and not args.all_bowl:
        args.all_bowl = True

    results_dir = get_results_dir()
    model = pipeline.load_model(config.MODEL_PATH)
    with (results_dir / "peak_params_k24_mel.json").open("r", encoding="utf-8") as f:
        peak_params = json.load(f)
    reset_single_inference_dir(results_dir)
    if args.all_bowl:
        input_wavs = sorted((config.PROJECT_ROOT / "data" / "test").glob("Bowl_*.wav"))
        if not input_wavs:
            raise FileNotFoundError("No Bowl_*.wav files found under data/test.")
    else:
        input_wavs = [args.wav]

    rows = []
    for index, wav_path in enumerate(input_wavs, start=1):
        summary = run_one(wav_path, results_dir, model, peak_params)
        rows.append(summary)
        print(f"[{index}/{len(input_wavs)}] {wav_path.name}: "
              f"predicted={summary['predicted_count']}, true={summary['true_count']}, "
              f"abs_error={summary['absolute_count_error']}")
        print(f"  Plot: experiments/pipeline_k24_mel/results/"
              f"single_inference/{wav_path.stem}/peak_detection.png")

    if args.all_bowl:
        batch_dir = results_dir / "single_inference"
        batch_dir.mkdir(parents=True, exist_ok=True)
        batch_path = batch_dir / "batch_summary.csv"
        pd.DataFrame(rows).to_csv(batch_path, index=False)
        print(f"Processed {len(rows)} Bowl file(s) independently.")
        print("Batch summary: experiments/pipeline_k24_mel/results/"
              "single_inference/batch_summary.csv")


if __name__ == "__main__":
    main()
