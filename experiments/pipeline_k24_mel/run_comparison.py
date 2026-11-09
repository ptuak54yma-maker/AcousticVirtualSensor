"""Train/tune/evaluate K=24 Mel NMF beside the unchanged K=48 pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as old_config
from src.audio.loader import load_audio
from src.evaluation.count_metrics import compute_count_metrics
from src.evaluation.event_metrics import compute_event_metrics
from src.preprocessing.log_compression import log_compression
from src.preprocessing.stft import compute_stft
from src.nmf.activation import compute_h_event_sum
from src.nmf.dictionary import load_dictionary
from src.nmf.nnls import solve_nnls_batch

from experiments.pipeline_k24_mel import experiment_config as exp_config
from experiments.pipeline_k24_mel import mel_frontend
from experiments.pipeline_k24_mel import pipeline as new_pipeline
from experiments.pipeline_k24_mel import train_dictionary


RESULTS_DIR = exp_config.EXPERIMENT_DIR / "results"
CACHE_DIR = RESULTS_DIR / "activations"
PLOT_DIR = RESULTS_DIR / "plots"


def reset_results_directory() -> None:
    """Replace only the fixed experiment output folder; model files stay intact."""
    experiment_root = exp_config.EXPERIMENT_DIR.resolve()
    model_path = exp_config.MODEL_PATH.resolve()
    for folder_name in ("results", "runs"):
        folder = exp_config.EXPERIMENT_DIR / folder_name
        target = folder.resolve()
        if target.parent != experiment_root or target.name.lower() != folder_name:
            raise RuntimeError(f"Refusing to clear unexpected output path: {target}")
        if target in model_path.parents or target == model_path:
            raise RuntimeError(f"Refusing to clear a folder containing the model: {target}")
        if folder.is_symlink():
            raise RuntimeError(f"Refusing to clear a linked output folder: {folder}")
        if folder.exists():
            shutil.rmtree(folder)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_labels(path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    if not path.is_file():
        return labels
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        name = name.strip()
        if not name.lower().endswith(".wav"):
            name += ".wav"
        try:
            labels[name] = int(value.strip())
        except ValueError:
            continue
    return labels


def cached_path(split: str, model_name: str, stem: str) -> Path:
    return CACHE_DIR / split / model_name / f"{stem}.npz"


def save_activation(
    split: str,
    model_name: str,
    stem: str,
    h: np.ndarray,
    envelope: np.ndarray,
    peaks: np.ndarray | None,
) -> Path:
    path = cached_path(split, model_name, stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        H=np.asarray(h, dtype=np.float32),
        event_envelope=np.asarray(envelope, dtype=np.float32),
        frame_times=np.arange(envelope.size, dtype=np.float32)
        * (old_config.HOP_LENGTH / old_config.SAMPLE_RATE),
        peak_frames=np.asarray([] if peaks is None else peaks, dtype=np.int64),
    )
    return path


def process_new(audio: np.ndarray, model: dict[str, object]) -> dict[str, object]:
    start = time.perf_counter()
    mel = mel_frontend.mel_spectrogram(
        audio, mel_bank=model["mel_bank"], mel_input=model["mel_input"]
    )
    mel_ms = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    v = mel_frontend.compress_mel(
        mel, reference=model["reference"], compressor=model["compressor"],
        gamma=model["gamma"],
    )
    compress_ms = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    h = new_pipeline.solve_nnls_batch(v, model["W"])
    nnls_ms = (time.perf_counter() - start) * 1000.0
    event_count = exp_config.EVENT_COMPONENTS
    bowl_end = event_count + exp_config.BOWL_COMPONENTS
    return {
        "V": v,
        "H": h,
        "envelope": np.sum(h[:event_count], axis=0, dtype=np.float32),
        "bowl_envelope": np.sum(h[event_count:bowl_end], axis=0, dtype=np.float32),
        "env_envelope": np.sum(h[bowl_end:], axis=0, dtype=np.float32),
        "timings_ms": {"mel_frontend": mel_ms, "compressor": compress_ms, "nnls": nnls_ms},
    }


def process_old(audio: np.ndarray, w_old: np.ndarray) -> dict[str, object]:
    """Call the existing STFT -> log-magnitude -> K=48 NNLS code unchanged."""
    start = time.perf_counter()
    stft = compute_stft(audio)
    stft_ms = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    v = log_compression(stft)
    compress_ms = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    h = solve_nnls_batch(v, w_old)
    nnls_ms = (time.perf_counter() - start) * 1000.0
    return {
        "V": v,
        "H": h,
        "envelope": compute_h_event_sum(h),
        "timings_ms": {"stft": stft_ms, "compressor": compress_ms, "nnls": nnls_ms},
    }


def tune_peak_params(
    samples: list[dict[str, object]],
    distances: range = range(1, 16),
    extra_prominences: tuple[float, ...] = (),
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Tune prominence/distance and return the full P-by-D calibration-MAE matrix."""
    prominence_values: list[float] = [0.0]
    for sample in samples:
        envelope = sample["envelope"]
        _peaks, properties = find_peaks(envelope, prominence=0.0, distance=1)
        prominences = properties.get("prominences", np.empty(0, dtype=np.float32))
        prominence_values.extend(float(p) for p in prominences if p > 0 and np.isfinite(p))
    if len(prominence_values) == 1:
        candidates = np.asarray([0.0], dtype=np.float32)
    else:
        candidates = np.unique(
            np.quantile(np.asarray(prominence_values), np.linspace(0.0, 1.0, 101))
        ).astype(np.float32)
    if extra_prominences:
        candidates = np.unique(np.concatenate((
            candidates,
            np.asarray(extra_prominences, dtype=np.float32),
        )))

    distance_values = list(distances)
    if not distance_values:
        raise ValueError("At least one peak distance candidate is required.")
    mae_values = np.empty((len(candidates), len(distance_values)), dtype=np.float64)
    best: tuple[float, int, float] | None = None
    for p_index, prominence in enumerate(candidates):
        for d_index, distance in enumerate(distance_values):
            abs_errors = []
            for sample in samples:
                peaks, _ = find_peaks(
                    sample["envelope"],
                    prominence=float(prominence),
                    distance=distance,
                )
                abs_errors.append(abs(int(peaks.size) - int(sample["true_count"])))
            mae = float(np.mean(abs_errors))
            mae_values[p_index, d_index] = mae
            candidate = (mae, int(distance), float(prominence))
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise RuntimeError("Could not tune peak parameters: no labeled training files.")
    params = {
        "prominence": best[2],
        "distance": best[1],
        "calibration_count_mae": best[0],
        "calibration_files": len(samples),
    }
    matrix = pd.DataFrame(
        mae_values,
        index=pd.Index(candidates.astype(np.float64), name="P_prominence"),
        columns=[f"D_{distance}_frames" for distance in distance_values],
    )
    return params, matrix


def save_pd_search_outputs(
    model_name: str,
    matrix: pd.DataFrame,
    selected: dict[str, float | int],
    reference: dict[str, float | int] | None = None,
) -> None:
    """Save the P-by-D MAE table and mark selected/reference parameter pairs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    matrix_path = RESULTS_DIR / f"p_d_mae_matrix_{model_name}.csv"
    matrix.to_csv(matrix_path, float_format="%.8g")

    values = matrix.to_numpy(dtype=np.float64)
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    image = ax.imshow(values, origin="lower", aspect="auto", cmap="viridis_r")
    fig.colorbar(image, ax=ax, label="Calibration mean absolute count error (MAE)")
    d_values = [int(column.split("_")[1]) for column in matrix.columns]
    ax.set_xticks(np.arange(len(d_values)), labels=d_values)
    ax.set_xlabel("D: minimum peak distance (frames)")

    row_count = len(matrix.index)
    y_ticks = np.unique(np.linspace(0, row_count - 1, min(11, row_count), dtype=int))
    ax.set_yticks(y_ticks, labels=[f"{matrix.index[i]:.4g}" for i in y_ticks])
    ax.set_ylabel("P: peak prominence")
    selected_row = int(np.argmin(np.abs(matrix.index.to_numpy(dtype=float)
                                       - float(selected["prominence"]))))
    selected_col = d_values.index(int(selected["distance"]))
    ax.scatter([selected_col], [selected_row], marker="*", s=210,
               facecolors="red", edgecolors="white", linewidths=0.8,
               label=(f"selected P={float(selected['prominence']):.5g}, "
                      f"D={int(selected['distance'])}"))
    if reference is not None:
        reference_row = int(np.argmin(np.abs(
            matrix.index.to_numpy(dtype=float) - float(reference["prominence"])
        )))
        reference_col = d_values.index(int(reference["distance"]))
        reference_mae = float(matrix.iloc[reference_row, reference_col])
        ax.scatter(
            [reference_col], [reference_row], marker="X", s=150,
            facecolors="deepskyblue", edgecolors="black", linewidths=0.8,
            label=(f"original P={float(reference['prominence']):.5g}, "
                   f"D={int(reference['distance'])}, MAE={reference_mae:.3g}"),
        )
    ax.set_title(f"P/D search — {model_name}")
    ax.legend(loc="upper right")
    heatmap_path = RESULTS_DIR / f"p_d_mae_heatmap_{model_name}.png"
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)
    print(f"  P/D matrix: {matrix_path.name} ({matrix.shape[0]} x {matrix.shape[1]})")
    print(f"  P/D heatmap: {heatmap_path.name}")


def read_old_peak_params() -> dict[str, float | int]:
    path = old_config.PEAK_PARAMS_PATH
    if path.is_file():
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            return {
                "prominence": float(payload["prominence"]),
                "distance": int(payload["distance"]),
            }
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
    return {
        "prominence": float(old_config.DEFAULT_PEAK_PROMINENCE),
        "distance": int(old_config.DEFAULT_PEAK_DISTANCE_FRAMES),
    }


def evaluate_file(
    split: str,
    wav_path: Path,
    true_count: int | None,
    old_w: np.ndarray,
    old_params: dict[str, float | int],
    old_calibrated_params: dict[str, float | int],
    new_model: dict[str, object],
    new_params: dict[str, float | int],
    include_old: bool = True,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    audio, _ = load_audio(wav_path, target_sr=old_config.SAMPLE_RATE)
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []

    if include_old:
        old_result = process_old(audio, old_w)
        old_h = old_result["H"]
        old_envelope = old_result["envelope"]
        start = time.perf_counter()
        old_peaks, old_properties = find_peaks(
            old_envelope,
            prominence=float(old_params["prominence"]),
            distance=int(old_params["distance"]),
        )
        old_result["timings_ms"]["peak_detection"] = (time.perf_counter() - start) * 1000.0
        save_activation(split, "k48_original", wav_path.stem, old_h, old_envelope, old_peaks)
        rows.append(make_row(split, wav_path, "k48_original", true_count, old_peaks,
                             old_envelope.size, old_params))
        diagnostics.append(diagnostic_row(
            split, wav_path.name, "k48_original", old_result, old_peaks,
            old_properties, old_params, old_w, (0, 24, 36, 48),
        ))
        calibrated_start = time.perf_counter()
        calibrated_peaks, calibrated_properties = find_peaks(
            old_envelope,
            prominence=float(old_calibrated_params["prominence"]),
            distance=int(old_calibrated_params["distance"]),
        )
        calibrated_ms = (time.perf_counter() - calibrated_start) * 1000.0
        calibrated_result = {**old_result, "timings_ms": {
            **old_result["timings_ms"], "peak_detection": calibrated_ms,
        }}
        save_activation(split, "k48_calibrated", wav_path.stem, old_h,
                        old_envelope, calibrated_peaks)
        rows.append(make_row(split, wav_path, "k48_calibrated", true_count,
                             calibrated_peaks, old_envelope.size, old_calibrated_params))
        diagnostics.append(diagnostic_row(
            split, wav_path.name, "k48_calibrated", calibrated_result,
            calibrated_peaks, calibrated_properties, old_calibrated_params,
            old_w, (0, 24, 36, 48),
        ))
    else:
        old_envelope = None
        old_peaks = None

    new_result = process_new(audio, new_model)
    new_h = new_result["H"]
    new_envelope = new_result["envelope"]
    start = time.perf_counter()
    new_peaks, new_properties = find_peaks(
        new_envelope,
        prominence=float(new_params["prominence"]),
        distance=int(new_params["distance"]),
    )
    new_result["timings_ms"]["peak_detection"] = (time.perf_counter() - start) * 1000.0
    save_activation(split, "k24_mel", wav_path.stem, new_h, new_envelope, new_peaks)
    new_row = make_row(split, wav_path, "k24_mel", true_count, new_peaks,
                       new_envelope.size, new_params)
    rows.append(new_row)
    diagnostics.append(diagnostic_row(
        split, wav_path.name, "k24_mel", new_result, new_peaks,
        new_properties, new_params, new_model["W"],
        (0, 12, 18, 24),
    ))

    if split == "test" and old_envelope is not None:
        save_comparison_plot(
            wav_path.stem, old_envelope, old_peaks, new_envelope, new_peaks,
            old_params, new_params, true_count,
        )
    return rows, diagnostics


def diagnostic_row(
    split: str,
    filename: str,
    model_name: str,
    result: dict[str, object],
    peaks: np.ndarray,
    peak_properties: dict[str, np.ndarray],
    params: dict[str, float | int],
    w: np.ndarray,
    group_bounds: tuple[int, int, int, int],
) -> dict[str, object]:
    v = np.asarray(result["V"], dtype=np.float32)
    h = np.asarray(result["H"], dtype=np.float32)
    envelope = np.asarray(result["envelope"], dtype=np.float32)
    reconstruction = w @ h
    residual = v - reconstruction
    v_norm = float(np.linalg.norm(v))
    residual_norm = float(np.linalg.norm(residual))
    flat_h = h.reshape(-1)
    peak_values = envelope[peaks] if peaks.size else np.empty(0, dtype=np.float32)
    peak_prominences = peak_properties.get("prominences", np.empty(0, dtype=np.float32))
    intervals = np.diff(peaks).astype(np.float32)
    row: dict[str, object] = {
        "split": split,
        "filename": filename,
        "model": model_name,
        "feature_bins": int(v.shape[0]),
        "frames": int(v.shape[1]),
        "feature_mean": float(np.mean(v)),
        "feature_std": float(np.std(v)),
        "feature_p95": float(np.percentile(v, 95)),
        "feature_max": float(np.max(v)) if v.size else 0.0,
        "relative_reconstruction_error": residual_norm / max(v_norm, 1e-12),
        "reconstruction_rmse": float(np.sqrt(np.mean(np.square(residual)))) if residual.size else 0.0,
        "H_mean": float(np.mean(flat_h)),
        "H_p95": float(np.percentile(flat_h, 95)),
        "H_nonzero_fraction": float(np.mean(flat_h > 1e-8)),
        "W_bytes": int(w.nbytes),
        "V_bytes": int(v.nbytes),
        "H_bytes": int(h.nbytes),
        "event_envelope_mean": float(np.mean(envelope)),
        "event_envelope_std": float(np.std(envelope)),
        "event_envelope_p95": float(np.percentile(envelope, 95)),
        "event_envelope_max": float(np.max(envelope)) if envelope.size else 0.0,
        "event_envelope_sum": float(np.sum(envelope, dtype=np.float64)),
        "event_envelope_sum_per_frame": float(np.mean(envelope)) if envelope.size else 0.0,
        "event_envelope_cv": (
            float(np.std(envelope) / np.mean(envelope))
            if envelope.size and np.mean(envelope) > 0 else 0.0
        ),
        "peak_count": int(peaks.size),
        "peak_prominence": float(params["prominence"]),
        "peak_distance_frames": int(params["distance"]),
        "peak_height_mean": float(np.mean(peak_values)) if peak_values.size else 0.0,
        "peak_prominence_mean": float(np.mean(peak_prominences)) if peak_prominences.size else 0.0,
        "peak_prominence_p95": float(np.percentile(peak_prominences, 95)) if peak_prominences.size else 0.0,
        "peak_prominence_over_event_p95": (
            float(params["prominence"]) / max(float(np.percentile(envelope, 95)), 1e-12)
            if envelope.size else 0.0
        ),
        "peak_density_per_minute": (
            float(peaks.size * old_config.SAMPLE_RATE * 60
                  / (envelope.size * old_config.HOP_LENGTH))
            if envelope.size else 0.0
        ),
        "peak_interval_median_frames": float(np.median(intervals)) if intervals.size else None,
        "peak_interval_cv": (
            float(np.std(intervals) / np.mean(intervals)) if intervals.size and np.mean(intervals) > 0 else None
        ),
    }
    e0, e1, b1, end = group_bounds
    total_group_sum = np.sum(h, axis=0)
    for name, lo, hi in (("event", e0, e1), ("bowl", e1, b1), ("env", b1, end)):
        group = h[lo:hi]
        group_sum_by_frame = np.sum(group, axis=0)
        row[f"{name}_components"] = int(group.shape[0])
        row[f"{name}_H_sum_mean"] = float(np.mean(group_sum_by_frame))
        row[f"{name}_activation_share_mean"] = float(np.mean(
            group_sum_by_frame / np.maximum(total_group_sum, 1e-12)
        ))
        row[f"{name}_dominant_frame_fraction"] = float(np.mean(
            group_sum_by_frame >= np.maximum.reduce([
                np.sum(h[e0:e1], axis=0), np.sum(h[e1:b1], axis=0), np.sum(h[b1:end], axis=0)
            ])
        ))
        row[f"{name}_H_mean_per_component"] = float(np.mean(group))
        row[f"{name}_H_p95"] = float(np.percentile(group, 95))
        row[f"{name}_H_nonzero_fraction"] = float(np.mean(group > 1e-8))
    for name, value in result["timings_ms"].items():
        row[f"host_{name}_ms"] = float(value)
    row["host_total_dsp_ms"] = float(sum(result["timings_ms"].values()))
    row["host_total_dsp_ms_per_frame"] = (
        float(sum(result["timings_ms"].values()) / h.shape[1]) if h.shape[1] else 0.0
    )
    return row


def make_row(
    split: str,
    wav_path: Path,
    model_name: str,
    true_count: int | None,
    peaks: np.ndarray,
    frame_count: int,
    params: dict[str, float | int],
) -> dict[str, object]:
    pred_count = int(peaks.size)
    row: dict[str, object] = {
        "split": split,
        "model": model_name,
        "filename": wav_path.name,
        "true_count": true_count,
        "pred_count": pred_count,
        "abs_error": abs(pred_count - true_count) if true_count is not None else None,
        "accuracy_percent": (
            max(0.0, 1.0 - abs(pred_count - true_count) / max(1, true_count)) * 100.0
            if true_count is not None else None
        ),
        "frames": frame_count,
        "peak_prominence": float(params["prominence"]),
        "peak_distance_frames": int(params["distance"]),
    }
    annotation = old_config.TEST_ANNOTATION_DIR / f"{wav_path.stem}_labels.npz"
    if split == "test" and annotation.is_file():
        with np.load(annotation, allow_pickle=True) as data:
            labels = np.asarray(data["binary_labels"])
        if labels.size == frame_count:
            ev = compute_event_metrics(np.flatnonzero(labels == 1), peaks)
            row.update({
                "event_precision": ev["event_precision"],
                "event_recall": ev["event_recall"],
                "event_f1": ev["event_f1"],
            })
    return row


def save_comparison_plot(
    stem: str,
    old_env: np.ndarray,
    old_peaks: np.ndarray,
    new_env: np.ndarray,
    new_peaks: np.ndarray,
    old_params: dict[str, float | int],
    new_params: dict[str, float | int],
    true_count: int | None,
    output_dir: Path | None = None,
) -> None:
    output_dir = output_dir or PLOT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    hop_time = old_config.HOP_LENGTH / old_config.SAMPLE_RATE
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    for ax, env, peaks, title, color, params in (
        (axes[0], old_env, old_peaks, "Original pipeline (K=48)", "tab:blue", old_params),
        (axes[1], new_env, new_peaks, "Mel experiment (K=24)", "tab:orange", new_params),
    ):
        t = np.arange(env.size) * hop_time
        ax.plot(t, env, color=color, linewidth=0.8)
        ax.scatter(
            t[peaks], env[peaks], color="crimson", s=22, zorder=3,
            label=f"Detected peaks: {peaks.size}",
        )
        ax.set_ylabel("H_event sum")
        count_label = f"Predicted count: {peaks.size}"
        if true_count is not None:
            count_label += f" | absolute error: {abs(int(peaks.size) - int(true_count))}"
        ax.set_title(
            f"{title} — {count_label} — P={float(params['prominence']):.4g}, "
            f"D={params['distance']} frames"
        )
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    count_text = (
        f"Ground-truth count from label.txt: {true_count}"
        if true_count is not None else "Ground-truth count unavailable"
    )
    fig.suptitle(f"{stem} | {count_text} (label.txt has counts, not event timestamps)")
    fig.savefig(output_dir / f"{stem}_comparison.png", dpi=140)
    plt.close(fig)


def save_peak_parameter_plot(
    stem: str,
    split: str,
    envelope: np.ndarray,
    original_peaks: np.ndarray,
    calibrated_peaks: np.ndarray,
    original_params: dict[str, float | int],
    calibrated_params: dict[str, float | int],
    true_count: int,
) -> None:
    """Compare two peak thresholds on the exact same K=48 event envelope."""
    output_dir = PLOT_DIR / "train_peak_params"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_times = np.arange(envelope.size) * (
        old_config.HOP_LENGTH / old_config.SAMPLE_RATE
    )
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    ax.plot(frame_times, envelope, color="0.35", linewidth=0.75, label="K=48 H_event_sum")
    ax.scatter(
        frame_times[original_peaks], envelope[original_peaks],
        marker="x", color="tab:blue", s=42, zorder=3,
        label=f"Original P/D peaks: {original_peaks.size}",
    )
    ax.scatter(
        frame_times[calibrated_peaks], envelope[calibrated_peaks],
        marker="o", facecolors="none", edgecolors="crimson", s=55, zorder=4,
        label=f"Calibrated P/D peaks: {calibrated_peaks.size}",
    )
    original_error = abs(int(original_peaks.size) - true_count)
    calibrated_error = abs(int(calibrated_peaks.size) - true_count)
    ax.text(
        0.012, 0.975,
        f"Split: {split}\nGround-truth count: {true_count}\n"
        f"Original: {original_peaks.size} peaks, error={original_error}, "
        f"P={float(original_params['prominence']):.5g}, D={original_params['distance']}\n"
        f"Calibrated: {calibrated_peaks.size} peaks, error={calibrated_error}, "
        f"P={float(calibrated_params['prominence']):.5g}, D={calibrated_params['distance']}",
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9},
    )
    ax.set(
        title=f"{stem} — K=48, same H_event_sum with original vs calibrated P/D",
        xlabel="Time (s)", ylabel="H_event_sum",
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    fig.savefig(output_dir / f"{stem}_p_d_comparison.png", dpi=140)
    plt.close(fig)


def compare_train_peak_params(
    calibration_samples: list[dict[str, object]],
    validation_samples: list[dict[str, object]],
    original_params: dict[str, float | int],
    calibrated_params: dict[str, float | int],
) -> pd.DataFrame:
    """Write per-file and aggregate K=48 original/calibrated P/D train results."""
    samples: list[tuple[str, str, int, np.ndarray]] = []
    for sample in calibration_samples:
        samples.append(("train_calibration", str(sample["name"]),
                        int(sample["true_count"]), sample["envelope"]))
    for sample in validation_samples:
        samples.append(("validation", sample["wav_path"].name,
                        int(sample["true_count"]), sample["old_result"]["envelope"]))

    rows: list[dict[str, object]] = []
    for split, filename, true_count, envelope in samples:
        predictions: dict[str, np.ndarray] = {}
        for model_name, params in (
            ("k48_original", original_params),
            ("k48_calibrated", calibrated_params),
        ):
            peaks, _ = find_peaks(
                envelope, prominence=float(params["prominence"]),
                distance=int(params["distance"]),
            )
            predictions[model_name] = peaks
            rows.append({
                "split": split,
                "filename": filename,
                "model": model_name,
                "true_count": true_count,
                "pred_count": int(peaks.size),
                "abs_error": abs(int(peaks.size) - true_count),
                "accuracy_percent": max(
                    0.0, 1.0 - abs(int(peaks.size) - true_count) / max(1, true_count)
                ) * 100.0,
                "peak_prominence": float(params["prominence"]),
                "peak_distance_frames": int(params["distance"]),
            })
        save_peak_parameter_plot(
            Path(filename).stem, split, envelope,
            predictions["k48_original"], predictions["k48_calibrated"],
            original_params, calibrated_params, true_count,
        )

    details = pd.DataFrame(rows)
    details.to_csv(RESULTS_DIR / "train_peak_parameter_comparison.csv", index=False)
    summaries: list[dict[str, object]] = []
    for (split, model_name), group in details.groupby(["split", "model"], sort=False):
        metrics = compute_count_metrics(
            group["true_count"].to_numpy(), group["pred_count"].to_numpy()
        )
        summaries.append({
            "split": split, "model": model_name, **metrics,
            "evaluated_files": len(group),
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv(RESULTS_DIR / "train_peak_parameter_summary.csv", index=False)
    return details


def save_summary(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "per_file_metrics.csv", index=False)
    test_df = df[(df["split"] == "test") & df["true_count"].notna()]
    summaries: list[dict[str, object]] = []
    for model_name, group in test_df.groupby("model", sort=False):
        metrics = compute_count_metrics(
            group["true_count"].to_numpy(), group["pred_count"].to_numpy()
        )
        summaries.append({"split": "test", "model": model_name, **metrics,
                          "evaluated_files": len(group)})
    summary = pd.DataFrame(summaries)
    summary.to_csv(RESULTS_DIR / "summary_metrics.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/tune the new K=24 Mel pipeline and compare it with K=48."
    )
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain the K=24 dictionary even if a bundle already exists.")
    parser.add_argument("--compressor", choices=("log1p", "power", "linear"), default=None,
                        help="Override the Mel compressor and retrain this experiment.")
    parser.add_argument("--mel-input", choices=("power", "magnitude"), default=None,
                        help="Override the quantity passed into the Mel filterbank.")
    parser.add_argument("--n-mels", type=int, default=None,
                        help="Override the number of Mel bands and retrain this experiment.")
    args = parser.parse_args()

    config_overridden = any(
        value is not None for value in (args.compressor, args.mel_input, args.n_mels)
    )
    if args.compressor:
        exp_config.COMPRESSOR = args.compressor
    if args.mel_input:
        exp_config.MEL_INPUT = args.mel_input
    if args.n_mels is not None:
        if args.n_mels <= 0 or args.n_mels > exp_config.N_FFT // 2 + 1:
            raise ValueError("--n-mels must be between 1 and N_FFT/2+1.")
        exp_config.N_MELS = args.n_mels

    start = time.perf_counter()
    train_labels = parse_labels(old_config.TRAIN_AUDIO_DIR / "label.txt")
    test_labels = parse_labels(old_config.TEST_AUDIO_DIR / "label.txt")
    train_files = sorted([
        (old_config.TRAIN_AUDIO_DIR / name, count)
        for name, count in train_labels.items()
        if name.startswith("Train_") and (old_config.TRAIN_AUDIO_DIR / name).is_file()
    ], key=lambda item: item[0].name)
    test_files = sorted(old_config.TEST_AUDIO_DIR.glob("*.wav"))
    if not train_files:
        raise FileNotFoundError("No labeled Train_*.wav files found for peak tuning.")
    if not test_files:
        raise FileNotFoundError(f"No test WAVs found in {old_config.TEST_AUDIO_DIR}.")
    if not exp_config.MODEL_PATH.is_file() or args.retrain or config_overridden:
        print("Training the separate K=24 Mel dictionary...")
        train_dictionary.main()

    reset_results_directory()
    model = new_pipeline.load_model()
    with np.load(exp_config.MODEL_PATH, allow_pickle=False) as bundle:
        nmf_iterations = [int(value) for value in bundle["nmf_iterations"]]
    w_old = load_dictionary(old_config.W_STANDARD_PATH)
    old_params = read_old_peak_params()

    if len(train_files) < 5:
        raise ValueError("Need at least 5 labeled Train_*.wav files for calibration/validation split.")
    val_size = max(3, int(np.ceil(0.2 * len(train_files))))
    val_indices = set(np.random.default_rng(42).choice(
        len(train_files), size=val_size, replace=False
    ).tolist())
    calibration_files = [item for i, item in enumerate(train_files) if i not in val_indices]
    validation_files = [item for i, item in enumerate(train_files) if i in val_indices]

    print(f"Preparing {len(train_files)} labeled training files: "
          f"{len(calibration_files)} calibration, {len(validation_files)} validation...")
    new_calibration_samples: list[dict[str, object]] = []
    old_calibration_samples: list[dict[str, object]] = []
    validation_cache: list[dict[str, object]] = []
    calibration_names = {path.name for path, _count in calibration_files}
    for index, (wav_path, true_count) in enumerate(train_files, start=1):
        audio, _ = load_audio(wav_path, target_sr=old_config.SAMPLE_RATE)
        new_result = process_new(audio, model)
        old_result = process_old(audio, w_old)
        split = "train_calibration" if wav_path.name in calibration_names else "validation"
        if split == "train_calibration":
            for model_name, result, target in (
                ("k24_mel", new_result, new_calibration_samples),
                ("k48_original", old_result, old_calibration_samples),
            ):
                save_activation(split, model_name, wav_path.stem, result["H"], result["envelope"], None)
                target.append({"name": wav_path.name, "true_count": true_count,
                               "envelope": result["envelope"]})
        else:
            validation_cache.append({
                "wav_path": wav_path,
                "true_count": true_count,
                "new_result": new_result,
                "old_result": old_result,
            })
        print(f"  train {index}/{len(train_files)}: {wav_path.name} [{split}]")

    new_params, new_pd_matrix = tune_peak_params(new_calibration_samples)
    old_calibrated_params, old_pd_matrix = tune_peak_params(
        old_calibration_samples,
        extra_prominences=(float(old_params["prominence"]),),
    )
    save_pd_search_outputs("k24_mel", new_pd_matrix, new_params)
    save_pd_search_outputs(
        "k48_original_vs_calibrated", old_pd_matrix, old_calibrated_params,
        reference=old_params,
    )
    (RESULTS_DIR / "peak_params_k24_mel.json").write_text(
        json.dumps(new_params, indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / "peak_params_k48_calibrated.json").write_text(
        json.dumps(old_calibrated_params, indent=2), encoding="utf-8"
    )
    compare_train_peak_params(
        old_calibration_samples, validation_cache, old_params, old_calibrated_params
    )

    # Plot every labeled training recording after freezing P/D. Calibration
    # and validation files remain in separate folders for easy inspection.
    old_calibration_by_name = {
        sample["name"]: sample for sample in old_calibration_samples
    }
    for sample in new_calibration_samples:
        old_sample = old_calibration_by_name[sample["name"]]
        old_peaks, _ = find_peaks(
            old_sample["envelope"], prominence=float(old_params["prominence"]),
            distance=int(old_params["distance"]),
        )
        new_peaks, _ = find_peaks(
            sample["envelope"], prominence=float(new_params["prominence"]),
            distance=int(new_params["distance"]),
        )
        save_comparison_plot(
            Path(sample["name"]).stem, old_sample["envelope"], old_peaks,
            sample["envelope"], new_peaks, old_params, new_params,
            int(sample["true_count"]), output_dir=PLOT_DIR / "train_calibration",
        )
    for sample in validation_cache:
        old_result = sample["old_result"]
        new_result = sample["new_result"]
        old_peaks, _ = find_peaks(
            old_result["envelope"], prominence=float(old_params["prominence"]),
            distance=int(old_params["distance"]),
        )
        new_peaks, _ = find_peaks(
            new_result["envelope"], prominence=float(new_params["prominence"]),
            distance=int(new_params["distance"]),
        )
        save_comparison_plot(
            sample["wav_path"].stem, old_result["envelope"], old_peaks,
            new_result["envelope"], new_peaks, old_params, new_params,
            int(sample["true_count"]), output_dir=PLOT_DIR / "validation",
        )

    validation_rows: list[dict[str, object]] = []
    validation_diagnostics: list[dict[str, object]] = []
    for sample in validation_cache:
        wav_path = sample["wav_path"]
        for model_name, result, params, w, bounds in (
            ("k48_original", sample["old_result"], old_params,
             w_old, (0, 24, 36, 48)),
            ("k48_calibrated", sample["old_result"], old_calibrated_params,
             w_old, (0, 24, 36, 48)),
            ("k24_mel", sample["new_result"], new_params,
             model["W"], (0, 12, 18, 24)),
        ):
            peak_start = time.perf_counter()
            peaks, props = find_peaks(
                result["envelope"], prominence=float(params["prominence"]),
                distance=int(params["distance"]),
            )
            save_activation("validation", model_name, wav_path.stem,
                            result["H"], result["envelope"], peaks)
            validation_rows.append(make_row(
                "validation", wav_path, model_name, sample["true_count"],
                peaks, result["envelope"].size, params,
            ))
            result["timings_ms"]["peak_detection"] = (time.perf_counter() - peak_start) * 1000.0
            validation_diagnostics.append(diagnostic_row(
                "validation", wav_path.name, model_name, result, peaks,
                props, params, w, bounds,
            ))
    validation_df = pd.DataFrame(validation_rows)
    validation_df.to_csv(RESULTS_DIR / "validation_metrics.csv", index=False)
    validation_diag_df = pd.DataFrame(validation_diagnostics)
    validation_diag_df.to_csv(RESULTS_DIR / "validation_pipeline_diagnostics.csv", index=False)
    val_acc = {
        name: float(group["accuracy_percent"].mean())
        for name, group in validation_df.groupby("model")
    }
    accuracy_non_degraded = all(
        val_acc.get("k24_mel", -1.0) >= val_acc.get(baseline, float("inf"))
        for baseline in ("k48_original", "k48_calibrated")
    )
    all_nmf_converged = all(value < exp_config.NMF_MAX_ITER for value in nmf_iterations)
    promotion_pass = accuracy_non_degraded and all_nmf_converged

    metadata = {
        "output_policy": "fixed results/ folder; each comparison overwrites prior outputs",
        "comparison": "old K=48 log-magnitude vs experimental K=24 Mel-NMF",
        "old_dictionary": str(old_config.W_STANDARD_PATH),
        "new_dictionary": str(exp_config.MODEL_PATH),
        "new_frontend": {
            "n_mels": exp_config.N_MELS,
            "mel_input": exp_config.MEL_INPUT,
            "compressor": exp_config.COMPRESSOR,
            "compressor_reference": model["reference"],
            "nmf_init": exp_config.NMF_INIT,
            "nmf_max_iter": exp_config.NMF_MAX_ITER,
            "nmf_iterations_event_bowl_env": nmf_iterations,
            "nmf_converged_event_bowl_env": [
                value < exp_config.NMF_MAX_ITER for value in nmf_iterations
            ],
            "components_event_bowl_env": [
                exp_config.EVENT_COMPONENTS,
                exp_config.BOWL_COMPONENTS,
                exp_config.ENV_COMPONENTS,
            ],
        },
        "peak_params": {
            "k48_original_deployed": old_params,
            "k48_original_calibrated": old_calibrated_params,
            "k24_mel": new_params,
        },
        "threshold_tuning_data": [path.name for path, _ in calibration_files],
        "validation_data": [path.name for path, _ in validation_files],
        "validation_accuracy_percent": val_acc,
        "accuracy_non_degraded_on_validation": accuracy_non_degraded,
        "all_nmf_sources_converged": all_nmf_converged,
        "promotion_status": "PASS" if promotion_pass else "HOLD",
        "test_data": [path.name for path in test_files],
        "test_event_annotations_available": any(
            (old_config.TEST_ANNOTATION_DIR / f"{path.stem}_labels.npz").is_file()
            for path in test_files
        ),
    }
    (RESULTS_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    rows: list[dict[str, object]] = []
    pipeline_diagnostics: list[dict[str, object]] = []
    for index, wav_path in enumerate(test_files, start=1):
        true_count = test_labels.get(wav_path.name)
        file_rows, file_diagnostics = evaluate_file(
            "test", wav_path, true_count, w_old, old_params,
            old_calibrated_params, model, new_params
        )
        rows.extend(file_rows)
        pipeline_diagnostics.extend(file_diagnostics)
        print(f"  test {index}/{len(test_files)}: {wav_path.name}")

    pd.DataFrame(pipeline_diagnostics).to_csv(
        RESULTS_DIR / "pipeline_diagnostics.csv", index=False
    )
    diagnostic_frame = pd.DataFrame(pipeline_diagnostics)
    numeric_diagnostics = diagnostic_frame.select_dtypes(include=[np.number]).columns
    diagnostic_frame.groupby("model", sort=False)[list(numeric_diagnostics)].mean().reset_index().to_csv(
        RESULTS_DIR / "diagnostic_summary.csv", index=False
    )
    # Save count-level calibration results for transparency; do not mix them
    # into held-out test metrics.
    calibration_rows = []
    for sample in new_calibration_samples:
        peaks, _ = find_peaks(
            sample["envelope"], prominence=float(new_params["prominence"]),
            distance=int(new_params["distance"]),
        )
        calibration_rows.append({
            "split": "train_calibration",
            "model": "k24_mel",
            "filename": sample["name"],
            "true_count": sample["true_count"],
            "pred_count": int(peaks.size),
            "abs_error": abs(int(peaks.size) - int(sample["true_count"])),
            "peak_prominence": new_params["prominence"],
            "peak_distance_frames": new_params["distance"],
        })
    for sample in old_calibration_samples:
        peaks, _ = find_peaks(
            sample["envelope"], prominence=float(old_calibrated_params["prominence"]),
            distance=int(old_calibrated_params["distance"]),
        )
        calibration_rows.append({
            "split": "train_calibration", "model": "k48_calibrated",
            "filename": sample["name"], "true_count": sample["true_count"],
            "pred_count": int(peaks.size),
            "abs_error": abs(int(peaks.size) - int(sample["true_count"])),
            "peak_prominence": old_calibrated_params["prominence"],
            "peak_distance_frames": old_calibrated_params["distance"],
        })
    pd.DataFrame(calibration_rows).to_csv(
        RESULTS_DIR / "training_calibration_metrics.csv", index=False
    )

    summary = save_summary(rows)
    test_acc = {
        str(row["model"]): float(row["mean_accuracy"])
        for row in summary.to_dict("records")
    }
    test_non_degraded = (
        test_acc.get("k24_mel", -1.0) >= test_acc.get("k48_original", float("inf"))
    )
    promotion_pass = accuracy_non_degraded and test_non_degraded and all_nmf_converged
    metadata.update({
        "test_accuracy_percent": test_acc,
        "accuracy_non_degraded_on_test_vs_deployed_k48": test_non_degraded,
        "promotion_status": "PASS" if promotion_pass else "HOLD",
    })
    (RESULTS_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / "promotion_gate.json").write_text(json.dumps({
        "status": "PASS" if promotion_pass else "HOLD",
        "rule": (
            "K=24 mean count accuracy must be >= both K=48 deployed and recalibrated "
            "on validation, >= K=48 deployed on held-out test, and all source NMF fits converged"
        ),
        "validation_accuracy_percent": val_acc,
        "test_accuracy_percent": test_acc,
        "accuracy_non_degraded_on_validation": accuracy_non_degraded,
        "accuracy_non_degraded_on_test_vs_deployed_k48": test_non_degraded,
        "all_nmf_sources_converged": all_nmf_converged,
    }, indent=2), encoding="utf-8")
    print("\nHeld-out test comparison (count metrics):")
    print(summary.to_string(index=False) if not summary.empty else "No labeled test counts found.")
    print(f"Validation promotion gate: {'PASS' if promotion_pass else 'HOLD'}")
    print("\nComparison results overwritten under experiments/pipeline_k24_mel/results.")
    print(f"Elapsed: {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
