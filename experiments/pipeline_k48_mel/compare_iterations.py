"""Compare SciPy NNLS with MCU-style FISTA iteration counts on identical Mel features."""

from __future__ import annotations

import json
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.audio.loader import load_audio
from experiments.pipeline_k48_mel import experiment_config as cfg
from experiments.pipeline_k48_mel import mel_frontend
from experiments.pipeline_k48_mel import pipeline

RESULTS = cfg.EXPERIMENT_DIR / "results"
ITERATIONS = (8, 16, 32, 64, 128, 256, 512, 1_024, 2_048, 4_096)
CALIBRATION_NAMES = {
    "Train_0.wav", "Train_2.wav", "Train_5.wav", "Train_6.wav", "Train_7.wav",
    "Train_8.wav", "Train_9.wav", "Train_10.wav", "Train_11.wav", "Train_12.wav",
    "Train_13.wav", "Train_14.wav", "Train_16.wav", "Train_17.wav",
}


def labels(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            name, value = line.split(":", 1)
            out[name.strip() + ("" if name.strip().lower().endswith(".wav") else ".wav")] = int(value)
    return out


def fista(w: np.ndarray, v: np.ndarray, steps: int) -> np.ndarray:
    """Float32 FISTA with the same Gram matrix and safe row-sum step as firmware."""
    w = np.ascontiguousarray(w, dtype=np.float32)
    v = np.ascontiguousarray(v, dtype=np.float32)
    gram = np.asarray(w.T @ w, dtype=np.float32)
    rhs = np.asarray(w.T @ v, dtype=np.float32)
    lipschitz = float(np.max(np.sum(gram, axis=1, dtype=np.float32)))
    x = np.zeros_like(rhs)
    y = np.zeros_like(rhs)
    scale = np.float32(1.0)
    step = np.float32(1.0 / lipschitz)
    for _ in range(steps):
        next_scale = np.float32(0.5) * (
            np.float32(1.0) + np.sqrt(np.float32(1.0) + np.float32(4.0) * scale * scale)
        )
        beta = (scale - np.float32(1.0)) / next_scale
        x_next = np.maximum(np.float32(0.0), y + step * (rhs - gram @ y))
        y = x_next + beta * (x_next - x)
        x, scale = x_next, next_scale
    return np.ascontiguousarray(x, dtype=np.float32)


def exact_nnls(w: np.ndarray, v: np.ndarray) -> np.ndarray:
    h = np.empty((w.shape[1], v.shape[1]), dtype=np.float32)
    for i in range(v.shape[1]):
        h[:, i], _ = nnls(w, v[:, i], maxiter=cfg.NNLS_MAX_ITER)
    return h


def tune(samples: list[dict], event_count: int) -> dict:
    candidates = [0.0]
    for sample in samples:
        peaks, props = find_peaks(sample["env"], prominence=0, distance=1)
        candidates.extend(float(p) for p in props["prominences"] if p > 0 and np.isfinite(p))
    thresholds = np.unique(np.quantile(candidates, np.linspace(0, 1, 101)))
    best = None
    for prominence in thresholds:
        for distance in range(1, 16):
            errors = [
                abs(find_peaks(s["env"], prominence=float(prominence), distance=distance)[0].size
                    - s["truth"])
                for s in samples
            ]
            candidate = (float(np.mean(errors)), distance, float(prominence))
            if best is None or candidate < best:
                best = candidate
    return {"calibration_mae": best[0], "distance": best[1], "prominence": best[2]}


def metrics(samples: list[dict], params: dict, model_name: str, split: str) -> list[dict]:
    rows = []
    for sample in samples:
        peaks = find_peaks(
            sample["env"], prominence=params["prominence"], distance=params["distance"]
        )[0]
        error = abs(int(peaks.size) - sample["truth"])
        accuracy = max(0.0, 1.0 - error / max(1, sample["truth"])) * 100.0
        rows.append({"split": split, "model": model_name, "file": sample["file"],
                     "truth": sample["truth"], "prediction": int(peaks.size),
                     "absolute_error": error, "accuracy_percent": accuracy,
                     "peak_prominence": params["prominence"], "peak_distance": params["distance"]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, nargs="+", default=list(ITERATIONS))
    args = parser.parse_args()
    iterations = tuple(args.iterations)
    if not iterations or any(value <= 0 for value in iterations):
        raise ValueError("FISTA iteration counts must be positive integers.")
    model = pipeline.load_model()
    w = model["W"]
    test_labels = labels(ROOT / "data" / "test" / "label.txt")
    train_labels = labels(ROOT / "data" / "train" / "label.txt")
    paths = [("train", p, train_labels[p.name]) for p in sorted((ROOT / "data" / "train").glob("Train_*.wav"))]
    paths += [("test", p, test_labels[p.name]) for p in sorted((ROOT / "data" / "test").glob("*.wav")) if p.name in test_labels]
    bank = model["mel_bank"]
    prepared: dict[str, dict] = {}
    for original_split, path, truth in paths:
        audio, _ = load_audio(path)
        mel = mel_frontend.mel_spectrogram(audio, bank, model["mel_input"])
        v = mel_frontend.compress_mel(mel, model["reference"], model["compressor"], model["gamma"])
        split = ("calibration" if path.name in CALIBRATION_NAMES else "validation") if original_split == "train" else "test"
        prepared[path.name] = {"file": path.name, "truth": truth, "V": v, "split": split}
        print(f"features {path.name}: {v.shape[1]} frames", flush=True)

    python_cache: dict[str, np.ndarray] = {}
    for name, sample in prepared.items():
        start = time.perf_counter()
        python_cache[name] = exact_nnls(w, sample["V"])
        print(f"Python NNLS {name}: {time.perf_counter() - start:.1f}s", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    np.save(cfg.STANDARD_W_PATH, w)
    all_rows: list[dict] = []
    activation_rows: list[dict] = []
    params_json: dict[str, dict] = {}

    def from_h(name: str, h: np.ndarray) -> dict:
        sample = prepared[name]
        return {"file": name, "truth": sample["truth"],
                "env": np.sum(h[:cfg.EVENT_COMPONENTS], axis=0, dtype=np.float32)}

    python_samples = {name: from_h(name, h) for name, h in python_cache.items()}
    calibration = [s for n, s in python_samples.items() if prepared[n]["split"] == "calibration"]
    python_params = tune(calibration, cfg.EVENT_COMPONENTS)
    params_json["python_scipy_nnls"] = python_params
    for split in ("calibration", "validation", "test"):
        samples = [s for n, s in python_samples.items() if prepared[n]["split"] == split]
        all_rows.extend(metrics(samples, python_params, "python_scipy_nnls", split))
        for sample in samples:
            name = sample["file"]
            v = prepared[name]["V"]
            h = python_cache[name]
            activation_rows.append({
                "split": split, "model": "python_scipy_nnls", "file": name,
                "relative_H_error": 0.0,
                "event_envelope_correlation": 1.0,
                "relative_reconstruction_error": float(np.linalg.norm(v - w @ h) / max(np.linalg.norm(v), 1e-12)),
            })

    for steps in iterations:
        cache: dict[str, np.ndarray] = {}
        for name, sample in prepared.items():
            start = time.perf_counter()
            cache[name] = fista(w, sample["V"], steps)
            print(f"FISTA {steps:>3} {name}: {time.perf_counter() - start:.1f}s", flush=True)
            v_file = RESULTS / "host_cpp_inputs" / f"{Path(name).stem}.f32"
            v_file.parent.mkdir(parents=True, exist_ok=True)
            with v_file.open("wb") as f:
                np.asarray(sample["V"].shape[1], dtype=np.uint32).tofile(f)
                np.asarray(sample["V"].T, dtype=np.float32, order="C").tofile(f)
        fista_samples = {name: from_h(name, h) for name, h in cache.items()}
        for name, h in cache.items():
            v = prepared[name]["V"]
            h_ref = python_cache[name]
            env = np.sum(h[:cfg.EVENT_COMPONENTS], axis=0, dtype=np.float32)
            env_ref = np.sum(h_ref[:cfg.EVENT_COMPONENTS], axis=0, dtype=np.float32)
            activation_rows.append({
                "split": prepared[name]["split"], "model": f"fista_{steps}", "file": name,
                "relative_H_error": float(np.linalg.norm(h - h_ref) / max(np.linalg.norm(h_ref), 1e-12)),
                "event_envelope_correlation": float(np.corrcoef(env, env_ref)[0, 1]),
                "relative_reconstruction_error": float(np.linalg.norm(v - w @ h) / max(np.linalg.norm(v), 1e-12)),
            })
        cal = [s for n, s in fista_samples.items() if prepared[n]["split"] == "calibration"]
        params = tune(cal, cfg.EVENT_COMPONENTS)
        params_json[f"fista_{steps}"] = params
        for split in ("calibration", "validation", "test"):
            samples = [s for n, s in fista_samples.items() if prepared[n]["split"] == split]
            all_rows.extend(metrics(samples, params, f"fista_{steps}", split))
        np.savez_compressed(RESULTS / f"activations_fista_{steps}.npz",
                            **{name: h for name, h in cache.items()})

    frame = pd.DataFrame(all_rows)
    frame.to_csv(RESULTS / "iteration_accuracy_per_file.csv", index=False)
    pd.DataFrame(activation_rows).to_csv(RESULTS / "fista_vs_python_activation_metrics.csv", index=False)
    summary = frame.groupby(["split", "model"], sort=False).agg(
        files=("file", "count"), mean_accuracy_percent=("accuracy_percent", "mean"),
        count_mae=("absolute_error", "mean"), total_true=("truth", "sum"),
        total_pred=("prediction", "sum"),
    ).reset_index()
    summary.to_csv(RESULTS / "iteration_accuracy_summary.csv", index=False)
    (RESULTS / "fista_peak_params.json").write_text(json.dumps(params_json, indent=2), encoding="utf-8")
    reference_accuracy = {
        row["split"]: float(row["mean_accuracy_percent"])
        for row in summary[summary["model"] == "python_scipy_nnls"].to_dict("records")
    }
    eligible = []
    for candidate in sorted(iterations):
        candidate_name = f"fista_{candidate}"
        candidate_scores = {
            row["split"]: float(row["mean_accuracy_percent"])
            for row in summary[summary["model"] == candidate_name].to_dict("records")
        }
        if (candidate_scores.get("calibration", -1.0) >= reference_accuracy["calibration"]
                and candidate_scores.get("validation", -1.0) >= reference_accuracy["validation"]):
            eligible.append(candidate)
    recommended = min(eligible) if eligible else None
    legacy_path = ROOT / "experiments" / "pipeline_k24_mel" / "results" / "summary_metrics.csv"
    legacy_baseline = None
    if legacy_path.is_file():
        legacy_frame = pd.read_csv(legacy_path)
        legacy_row = legacy_frame[
            (legacy_frame["split"] == "test") & (legacy_frame["model"] == "k48_original")
        ]
        if not legacy_row.empty:
            legacy_baseline = {
                "model": "original 1,025-bin K=48 Python pipeline",
                "test_accuracy_percent": float(legacy_row.iloc[0]["mean_accuracy"]),
                "source": "experiments/pipeline_k24_mel/results/summary_metrics.csv",
            }
    (RESULTS / "iteration_comparison.json").write_text(json.dumps({
        "model": str(cfg.MODEL_PATH), "W_shape": list(w.shape), "source_W_saved_to": str(cfg.STANDARD_W_PATH),
        "compressor": model["compressor"], "compressor_reference": model["reference"],
        "n_mels": cfg.N_MELS, "components_event_bowl_env": [cfg.EVENT_COMPONENTS, cfg.BOWL_COMPONENTS, cfg.ENV_COMPONENTS],
        "fista_iterations": list(iterations), "python_reference": "SciPy scipy.optimize.nnls, fit on same W and Mel features",
        "mcu_solver": "same projected FISTA algorithm and float32 Gram-matrix math as espidf_experiments/avs_pipeline.cpp",
        "split_policy": {"calibration": "14 fixed Train_*.wav recordings", "validation": "4 held-out Train_*.wav recordings", "test": "all labeled data/test WAVs"},
        "python_params": python_params, "fista_params": params_json,
        "firmware_selection": {
            "criterion": "lowest FISTA iteration count with calibration and validation accuracy >= SciPy NNLS",
            "recommended_iterations": recommended,
            "test_labels_used_for_selection": False,
        },
        "legacy_python_baseline": legacy_baseline,
    }, indent=2), encoding="utf-8")
    print("\n", summary.to_string(index=False))
    best = summary[summary["split"] == "test"].sort_values("mean_accuracy_percent", ascending=False)
    print("\nTest accuracy by model:\n", best.to_string(index=False))


if __name__ == "__main__":
    main()
