#!/usr/bin/env python3
"""Replay one WAV through Arduino experiment profiles and save CSV/PNG results."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import serial
import soundfile as sf
from scipy.signal import find_peaks, resample_poly

ROOT = Path(__file__).resolve().parents[3]
PROFILES = ["legacy_nnls", "m64_nnls", "m128_nnls"] + [f"m64_fista{n}" for n in (8,16,32,64,128,256,512,1024,2048,4096)] + ["m128_fista512"]
BLOCK = 512


def line(port: serial.Serial, timeout: float = 30) -> str:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        raw = port.readline()
        if raw:
            value = raw.decode("ascii", "replace").strip()
            if value:
                return value
    raise TimeoutError("ESP32 stopped responding; check the selected port and sketch.")


def expect(port: serial.Serial, prefix: str, timeout: float = 30) -> str:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = line(port, max(1, end - time.monotonic()))
        if value.startswith("ERROR"):
            raise RuntimeError(value)
        if value.startswith(prefix):
            return value
    raise TimeoutError(f"Waiting for {prefix}")


def read_audio(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = np.mean(audio, axis=1, dtype=np.float32)
    if rate != 44100:
        from math import gcd
        d = gcd(int(rate), 44100)
        audio = resample_poly(audio, 44100 // d, int(rate) // d).astype(np.float32)
    if audio.size < 2048:
        raise ValueError("WAV must contain at least 2048 samples.")
    return np.ascontiguousarray(audio, dtype=np.float32)


def stream_profile(port: serial.Serial, audio: np.ndarray, profile: str) -> list[dict]:
    port.write(f"PROFILE {profile}\n".encode()); port.flush()
    expect(port, "PROFILE_OK")
    port.reset_input_buffer()
    port.write(f"START {audio.size} {float(np.mean(audio, dtype=np.float64)):.10g}\n".encode()); port.flush()
    expect(port, "ACK_START")
    for start in range(0, audio.size, BLOCK):
        chunk = np.asarray(audio[start:start + BLOCK], dtype="<f4")
        port.write(f"DATA {chunk.size}\n".encode()); port.write(chunk.tobytes()); port.flush()
        expect(port, "ACK", timeout=90)
    port.write(b"END\n"); port.flush()
    rows: list[dict] = []
    while True:
        value = line(port, timeout=120)
        if value.startswith("ROW,"):
            _, name, index, seconds, score, micros = value.split(",")
            rows.append({"profile": profile, "frame": int(index), "time_s": float(seconds),
                         "event_score": float(score), "processing_us": int(micros)})
        elif value.startswith("DONE,"):
            return rows
        elif value.startswith("ERROR"):
            raise RuntimeError(value)


def metrics(rows: list[dict], wav: Path, profile: str) -> dict | None:
    label_files = [ROOT / "data" / split / "label.txt" for split in ("test", "train")]
    truth = None
    for file in label_files:
        if file.exists():
            for item in file.read_text(encoding="utf-8").splitlines():
                if ":" in item:
                    name, value = item.split(":", 1)
                    if name.strip().lower().removesuffix(".wav") == wav.stem.lower():
                        truth = int(value.strip()); break
        if truth is not None: break
    params_path = ROOT / "experiments" / "pipeline_k48_mel" / "results" / "fista_peak_params.json"
    if truth is None or not params_path.exists() or not rows:
        return None
    all_params = json.loads(params_path.read_text(encoding="utf-8"))
    if profile == "legacy_nnls":
        params = json.loads((ROOT / "models" / "peak_detection" / "peak_params.json").read_text(encoding="utf-8"))
    else:
        key = "python_scipy_nnls" if profile.endswith("nnls") else "fista_" + profile.removeprefix("m64_fista")
        if profile == "m128_fista512": key = "fista_512"
        params = all_params.get(key, all_params.get("fista_512"))
    env = np.asarray([r["event_score"] for r in rows])
    peaks = find_peaks(env, prominence=float(params["prominence"]), distance=int(params["distance"]))[0]
    return {"profile": profile, "truth": truth, "prediction": int(peaks.size),
            "absolute_error": abs(int(peaks.size) - truth),
            "accuracy_percent": max(0., 1. - abs(int(peaks.size)-truth) / max(1, truth)) * 100}


def python_reference(wav: Path) -> tuple[np.ndarray, dict] | None:
    """Exact SciPy NNLS Python baseline for the selected Mel64 model."""
    params_file = ROOT / "experiments" / "pipeline_k48_mel" / "results" / "fista_peak_params.json"
    model_file = ROOT / "experiments" / "pipeline_k48_mel" / "models" / "W_standard_mel.npz"
    if not params_file.exists() or not model_file.exists():
        return None
    sys.path.insert(0, str(ROOT))
    from scipy.optimize import nnls
    from experiments.pipeline_k48_mel import mel_frontend
    with np.load(model_file, allow_pickle=False) as m:
        w, bank = m["W"], m["mel_bank"]
        ref = float(m["compressor_reference"])
        compressor, mode = str(m["compressor"].item()), str(m["mel_input"].item())
    values = read_audio(wav)
    spec = mel_frontend.mel_spectrogram(values, bank, mode)
    v = mel_frontend.compress_mel(spec, ref, compressor)
    h = np.empty((48, v.shape[1]), dtype=np.float32)
    for i in range(v.shape[1]): h[:, i], _ = nnls(w, v[:, i], maxiter=2000)
    env = np.sum(h[:24], axis=0, dtype=np.float32)
    params = json.loads(params_file.read_text(encoding="utf-8"))["python_scipy_nnls"]
    return env, params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="ESP32 COM port, for example COM5")
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--baud", type=int, default=2_000_000)
    parser.add_argument("--profiles", nargs="+", default=PROFILES, choices=PROFILES)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "arduino_experiment_viewer")
    args = parser.parse_args()
    audio = read_audio(args.wav); args.out.mkdir(parents=True, exist_ok=True)
    py_reference = python_reference(args.wav)
    with serial.Serial(args.port, args.baud, timeout=1, write_timeout=30) as port:
        time.sleep(2); port.reset_input_buffer()
        ready = False; deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if port.in_waiting and line(port).startswith("AVS_EXPERIMENT_READY"):
                ready = True; break
            port.write(b"PING\n"); port.flush()
            try:
                if line(port, 1) == "PONG": ready = True; break
            except TimeoutError: pass
        if not ready: raise TimeoutError("Experiment sketch did not become ready.")
        combined = []; summaries = []
        for profile in args.profiles:
            print(f"Running {profile} on {args.wav.name} ...", flush=True)
            rows = stream_profile(port, audio, profile); combined.extend(rows)
            summary = metrics(rows, args.wav, profile)
            if summary: summaries.append(summary)
            print(f"  {len(rows)} frames, mean MCU time {np.mean([r['processing_us'] for r in rows]):.0f} us/frame")
    stem = args.wav.stem
    with (args.out / f"{stem}_frames.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["profile", "frame", "time_s", "event_score", "processing_us"])
        writer.writeheader(); writer.writerows(combined)
    if py_reference is not None:
        py_env, py_params = py_reference
        py_peaks = find_peaks(py_env, prominence=float(py_params["prominence"]), distance=int(py_params["distance"]))[0]
        truth = next((s["truth"] for s in summaries), None)
        if truth is not None:
            err = abs(int(py_peaks.size) - truth)
            summaries.insert(0, {"profile": "python_scipy_nnls_mel64", "truth": truth,
                                 "prediction": int(py_peaks.size), "absolute_error": err,
                                 "accuracy_percent": max(0., 1. - err / max(1, truth)) * 100})
    else:
        py_env = None
    if summaries:
        with (args.out / f"{stem}_accuracy.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
        for s in summaries: print(f"{s['profile']}: truth={s['truth']} prediction={s['prediction']} accuracy={s['accuracy_percent']:.2f}%")
    import matplotlib.pyplot as plt
    fig, (ax, latency) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for profile in args.profiles:
        rows = [r for r in combined if r["profile"] == profile]
        ax.plot([r["time_s"] for r in rows], [r["event_score"] for r in rows], label=profile, linewidth=1)
        latency.plot([r["time_s"] for r in rows], [r["processing_us"] for r in rows], linewidth=.8, label=profile)
    if py_env is not None:
        py_times = np.arange(py_env.size) * (1024 / 44100)
        ax.plot(py_times, py_env, "k--", linewidth=1, alpha=.8, label="Python SciPy NNLS Mel64")
    ax.set_ylabel("Sum of first 24 activations"); ax.set_title(f"ESP32 experiment comparison: {args.wav.name}")
    ax.grid(alpha=.25); ax.legend(ncol=3, fontsize=8)
    latency.set_xlabel("Audio time (s)"); latency.set_ylabel("MCU time (µs/frame)"); latency.grid(alpha=.25)
    fig.tight_layout(); png = args.out / f"{stem}_comparison.png"; fig.savefig(png, dpi=160)
    print(f"Saved: {args.out / (stem + '_frames.csv')}\nSaved: {png}")


if __name__ == "__main__":
    main()
