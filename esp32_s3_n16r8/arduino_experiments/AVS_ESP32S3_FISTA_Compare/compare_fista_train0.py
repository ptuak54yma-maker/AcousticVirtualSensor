#!/usr/bin/env python3
"""Stream Train_0.wav through the ESP32 FISTA sketch and compare with Python."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import serial
import soundfile as sf
from scipy.optimize import nnls
from scipy.signal import find_peaks, get_window

ROOT = Path(__file__).resolve().parents[3]
SAMPLE_RATE = 44_100
N_FFT = 2048
HOP = 1024
BLOCK = 512


def read_line(port: serial.Serial, deadline: float) -> str:
    while time.monotonic() < deadline:
        raw = port.readline()
        if raw:
            line = raw.decode("ascii", errors="replace").strip()
            if line:
                print(f"ESP32: {line}")
                return line
    raise TimeoutError("Timed out waiting for a line from the ESP32.")


def read_until(port: serial.Serial, prefixes: tuple[str, ...], timeout: float) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = read_line(port, deadline)
        if line.startswith("ERROR") or line.startswith("BENCH_ERROR"):
            raise RuntimeError(line)
        if line.startswith(prefixes):
            return line
    raise TimeoutError(f"Timed out waiting for {prefixes}.")


def load_audio(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz audio, got {sr} Hz.")
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1, dtype=np.float32)
    else:
        audio = audio[:, 0]
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if not np.isfinite(audio).all() or len(audio) < N_FFT:
        raise ValueError("Audio is too short or contains non-finite samples.")
    return audio


def python_aligned_reference(audio: np.ndarray, dictionary: np.ndarray,
                             prominence: float, distance: int
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """SciPy active-set NNLS with the same per-frame mean removal as the MCU."""
    frames = np.lib.stride_tricks.sliding_window_view(audio, N_FFT)[::HOP].copy()
    frames -= frames.mean(axis=1, keepdims=True, dtype=np.float32)
    window = get_window("hann", N_FFT, fftbins=True).astype(np.float32)
    spectrum = np.fft.rfft(frames * window, axis=1).T.astype(np.complex64)
    v = np.log1p(np.abs(spectrum)).astype(np.float32)

    start = time.perf_counter()
    activations = np.empty((dictionary.shape[1], v.shape[1]), dtype=np.float32)
    for frame in range(v.shape[1]):
        activations[:, frame] = nnls(dictionary, v[:, frame], maxiter=20000)[0]
    elapsed = time.perf_counter() - start
    event_score = np.sum(activations[:24], axis=0, dtype=np.float32)
    peaks, _ = find_peaks(event_score, prominence=prominence, distance=distance)
    return v, activations, event_score, peaks.astype(np.int64), elapsed


def read_exact(port: serial.Serial, size: int, timeout: float = 90.0) -> bytes:
    deadline = time.monotonic() + timeout
    chunks = bytearray()
    while len(chunks) < size and time.monotonic() < deadline:
        block = port.read(min(65536, size - len(chunks)))
        if block:
            chunks.extend(block)
    if len(chunks) != size:
        raise TimeoutError(f"Timed out receiving binary arrays: {len(chunks)}/{size} bytes.")
    return bytes(chunks)


def wait_for_ready(port: serial.Serial, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        port.write(b"PING\n")
        port.flush()
        try:
            line = read_until(port, ("AVS_READY", "PONG"), 2.0)
        except TimeoutError:
            continue
        if line.startswith("AVS_READY"):
            return
    raise TimeoutError("No AVS_READY from the new FISTA sketch. Check upload and COM port.")


def stream_audio(port: serial.Serial, audio: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[str]]:
    port.write(f"BENCH_WAV {len(audio)}\n".encode("ascii"))
    port.flush()
    read_until(port, ("ACK_BENCH_START",), 10.0)

    sent = 0
    for offset in range(0, len(audio), BLOCK):
        block = audio[offset:offset + BLOCK]
        port.write(f"DATA {len(block)}\n".encode("ascii"))
        port.write(block.astype("<f4", copy=False).tobytes())
        port.flush()
        ack = read_until(port, ("ACK ",), 60.0)
        sent = min(offset + len(block), len(audio))
        if ack != f"ACK {sent}":
            raise RuntimeError(f"Unexpected board acknowledgment: {ack}; expected {sent}.")
        if sent % (SAMPLE_RATE * 5) < BLOCK or sent == len(audio):
            print(f"Sent {sent}/{len(audio)} samples ({sent / SAMPLE_RATE:.1f}s)")

    port.write(b"END\n")
    port.flush()
    scores: list[float] = []
    peaks: list[int] = []
    records: list[str] = []
    spectrum = np.empty((0, 0), dtype=np.float32)
    activations = np.empty((0, 0), dtype=np.float32)
    deadline = time.monotonic() + 180.0
    while time.monotonic() < deadline:
        line = read_line(port, deadline)
        if line.startswith("H_EVENT "):
            fields = dict(token.split("=", 1) for token in line.split()[1:] if "=" in token)
            frame = int(fields["frame"])
            while len(scores) <= frame:
                scores.append(float("nan"))
            scores[frame] = float(fields["score"])
        elif line.startswith("BENCH_") or line.startswith("RESULT "):
            records.append(line)
            if line.startswith("RESULT "):
                fields = dict(token.split("=", 1) for token in line.split()[1:] if "=" in token)
                peaks = [int(x) for x in fields.get("peak_frames", "").split(",") if x]
        elif line.startswith("FISTA_DATA_BEGIN "):
            fields = dict(token.split("=", 1) for token in line.split()[1:] if "=" in token)
            frames = int(fields["frames"])
            bins = int(fields["bins"])
            components = int(fields["components"])
            spectrum_bytes = int(fields["spectrum_bytes"])
            activation_bytes = int(fields["activation_bytes"])
            payload = read_exact(port, spectrum_bytes + activation_bytes)
            spectrum = np.frombuffer(payload[:spectrum_bytes], dtype="<f4").reshape(
                frames, bins).T.copy()
            activations = np.frombuffer(payload[spectrum_bytes:], dtype="<f4").reshape(
                frames, components).T.copy()
            records.append(line)
        if line == "FISTA_COMPARE_DONE":
            return (np.asarray(scores, dtype=np.float32), spectrum, activations,
                    peaks, records)
    raise TimeoutError("Timed out waiting for FISTA_COMPARE_DONE.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="ESP32 COM port, e.g. COM5")
    parser.add_argument("--baud", type=int, default=2_000_000)
    parser.add_argument("--wav", type=Path, default=ROOT / "data" / "train" / "Train_0.wav")
    args = parser.parse_args()

    audio = load_audio(args.wav)
    dictionary = np.ascontiguousarray(np.load(ROOT / "models" / "nmf" / "W_standard.npy"),
                                      dtype=np.float32)
    with (ROOT / "models" / "peak_detection" / "peak_params.json").open(encoding="utf-8") as f:
        peak_params = json.load(f)
    prominence = float(peak_params["prominence"])
    distance = int(peak_params["distance"])
    ref_spectrum, ref_activations, ref_score, ref_peaks, python_seconds = python_aligned_reference(
        audio, dictionary, prominence, distance)
    frame_count = 0 if len(audio) < N_FFT else 1 + (len(audio) - N_FFT) // HOP
    print(f"Python reference: {frame_count} frames, {len(ref_peaks)} peaks, "
          f"active-set solve {python_seconds:.3f}s on this computer.")

    try:
        with serial.Serial(args.port, args.baud, timeout=0.5, write_timeout=10) as port:
            wait_for_ready(port)
            esp_score, esp_spectrum, esp_activations, esp_peaks, records = stream_audio(port, audio)
    except serial.SerialTimeoutException as error:
        raise SystemExit(
            f"Serial write timed out on {args.port}. Close Serial Monitor, check the COM port, "
            "and reconnect the board."
        ) from error

    if len(esp_score) != frame_count or not np.isfinite(esp_score).all():
        raise RuntimeError(f"ESP32 returned {len(esp_score)} scores for {frame_count} expected frames.")
    if esp_spectrum.shape != ref_spectrum.shape or esp_activations.shape != ref_activations.shape:
        raise RuntimeError(
            f"Output shape mismatch: spectrum {esp_spectrum.shape}/{ref_spectrum.shape}, "
            f"activations {esp_activations.shape}/{ref_activations.shape}."
        )
    delta_spectrum = np.abs(esp_spectrum - ref_spectrum)
    delta_activations = np.abs(esp_activations - ref_activations)
    esp_event_from_h = np.sum(esp_activations[:24], axis=0, dtype=np.float32)
    delta_event = np.abs(esp_score - ref_score)
    direct_score_consistency = np.abs(esp_event_from_h - esp_score)
    common = len(set(esp_peaks) & set(ref_peaks.tolist()))
    print("\n=== FISTA ESP32 vs Python active-set, same per-frame preprocessing ===")
    print(f"frames={frame_count}")
    print(f"FFT/log-spectrum: MAE={float(delta_spectrum.mean()):.9g}; "
          f"RMSE={float(np.sqrt(np.mean((esp_spectrum-ref_spectrum)**2))):.9g}; "
          f"max_abs={float(delta_spectrum.max()):.9g}")
    print(f"NNLS H (all 48 components): MAE={float(delta_activations.mean()):.9g}; "
          f"RMSE={float(np.sqrt(np.mean((esp_activations-ref_activations)**2))):.9g}; "
          f"max_abs={float(delta_activations.max()):.9g}")
    print(f"H_event_sum (24 event components): MAE={float(delta_event.mean()):.9g}; "
          f"max_abs={float(delta_event.max()):.9g}; "
          f"H_sum_vs_direct_score_max_abs={float(direct_score_consistency.max()):.9g}")
    print(f"peak counts: ESP32={len(esp_peaks)}, Python={len(ref_peaks)}; "
          f"matching frame indices={common}/{len(ref_peaks)}")
    print("\n=== ESP32 timing records ===")
    for line in records:
        print(line)
    print("\nThe Python time is a host-side reference, not an MCU timing comparison.")


if __name__ == "__main__":
    main()
