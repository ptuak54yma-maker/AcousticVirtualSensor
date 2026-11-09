#!/usr/bin/env python3
"""Profile the ESP32-S3 pipeline using a real WAV segment or synthetic frames."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import serial
import soundfile as sf

SAMPLE_RATE = 44_100
MAX_BLOCK_SAMPLES = 512
MAX_PROFILE_FRAMES = 256
FFT_LENGTH = 2048
HOP_LENGTH = 1024


def read_response(port: serial.Serial, timeout_seconds: float, expected_prefix: str,
                  benchmark_lines: list[str] | None = None) -> str:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        raw = port.readline()
        if not raw:
            continue
        line = raw.decode("ascii", errors="replace").strip()
        if not line:
            continue
        print(f"ESP32: {line}")
        if benchmark_lines is not None and line.startswith("BENCH_"):
            benchmark_lines.append(line)
        if line.startswith("ERROR") or line.startswith("BENCH_ERROR"):
            raise RuntimeError(line)
        if line.startswith(expected_prefix):
            return line
    raise TimeoutError(f"Timed out waiting for ESP32 response: {expected_prefix}")


def wait_until_ready(port: serial.Serial, timeout_seconds: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    next_ping = 0.0
    while time.monotonic() < deadline:
        if time.monotonic() >= next_ping:
            port.write(b"PING\n")
            port.flush()
            next_ping = time.monotonic() + 0.5
        raw = port.readline()
        if not raw:
            continue
        line = raw.decode("ascii", errors="replace").strip()
        if line:
            print(f"ESP32: {line}")
        if line.startswith("AVS_READY") or line == "PONG":
            return
    raise TimeoutError("ESP32 did not become ready. Check the COM port and firmware.")


def clear_previous_stream(port: serial.Serial) -> None:
    # END clears a stale/incomplete stream if the preceding sender was stopped early.
    port.write(b"END\n")
    port.flush()
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        raw = port.readline()
        if not raw:
            continue
        line = raw.decode("ascii", errors="replace").strip()
        if line.startswith("ERROR END_COUNT"):
            print("ESP32: cleared stale or incomplete WAV stream state")
            return
        if line == "DONE":
            print("ESP32: previous WAV stream was already complete")
            return
        if line:
            print(f"ESP32: {line}")
    raise TimeoutError("ESP32 did not respond while clearing the previous stream state.")


def load_wav_segment(wav_path: Path, start_sec: float, duration_sec: float) -> np.ndarray:
    audio, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz WAV, got {sample_rate} Hz")
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    start_sample = round(start_sec * SAMPLE_RATE)
    stop_sample = round((start_sec + duration_sec) * SAMPLE_RATE)
    if start_sample < 0 or stop_sample > len(audio) or stop_sample <= start_sample:
        raise ValueError(
            f"Requested segment [{start_sec:.3f}, {start_sec + duration_sec:.3f}] s "
            f"is outside the WAV duration {len(audio) / sample_rate:.3f} s"
        )
    segment = np.ascontiguousarray(audio[start_sample:stop_sample], dtype="<f4")
    if not np.isfinite(segment).all():
        raise ValueError("WAV segment contains non-finite samples")
    frames = 0 if len(segment) < FFT_LENGTH else 1 + (len(segment) - FFT_LENGTH) // HOP_LENGTH
    if not 1 <= frames <= MAX_PROFILE_FRAMES:
        raise ValueError(f"Segment produces {frames} frames; profile limit is {MAX_PROFILE_FRAMES}")
    if frames < 129:
        print("Note: segment is shorter than a full 3-second peak window; peak timing may be absent.")
    return segment


def parse_fields(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in line.split()[1:]:
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    return fields


def print_benchmark_summary(lines: list[str]) -> None:
    records = [(line.split()[0], parse_fields(line)) for line in lines]
    by_name = {name: fields for name, fields in records}
    chip = by_name.get("BENCH_CHIP", {})
    config = by_name.get("BENCH_CONFIG", {})
    frame = by_name.get("BENCH_FRAME", {})
    total = by_name.get("BENCH_TOTAL_FRAME", {})
    realtime = by_name.get("BENCH_REALTIME", {})
    nnls = by_name.get("BENCH_NNLS_DIAGNOSTICS", {})
    event_score = by_name.get("BENCH_EVENT_SCORE", {})
    stages = [fields for name, fields in records if name == "BENCH_STAGE"]

    print("\n=== MCU benchmark summary ===")
    print(
        f"Chip: {chip.get('model', 'unknown')} | CPU: {chip.get('cpu_mhz', '?')} MHz | "
        f"core: {chip.get('measured_core', '?')} | cores: {chip.get('cores', '?')}"
    )
    print(
        f"Compiler optimization: {config.get('optimization', '?')} | "
        f"NNLS: {config.get('nnls_algorithm', '?')} {config.get('nnls_precision', '?')}, "
        f"{config.get('nnls_iterations', '?')} iterations | "
        f"deadline: {frame.get('deadline_us', '?')} us | "
        f"input: {frame.get('input_kind', '?')}"
    )
    print(
        f"Total/frame: avg {total.get('average_us', '?')} us, "
        f"P95 {total.get('p95_us', '?')} us | "
        f"realtime: {realtime.get('meets_realtime', '?')} "
        f"(capacity {realtime.get('capacity_x', '?')}x)"
    )
    print("\nStage                         avg us      P95 us      max us    avg cycles")
    for stage in stages:
        print(
            f"{stage.get('name', '?'):<28} "
            f"{stage.get('average_us', '?'):>9} "
            f"{stage.get('p95_us', '?'):>11} "
            f"{stage.get('max_us', '?'):>11} "
            f"{stage.get('average_cycles', '?'):>13}"
        )
    if nnls:
        print(
            "\nNNLS relative projected-gradient: "
            f"avg {nnls.get('avg_relative_projected_gradient', '?')}, "
            f"max {nnls.get('max_relative_projected_gradient', '?')}"
        )
    if event_score:
        print(
            "Event score: "
            f"min {event_score.get('min', '?')}, "
            f"avg {event_score.get('average', '?')}, "
            f"max {event_score.get('max', '?')}"
        )
    print(
        "\nInput copy excludes UART transfer/wait. Peak detection is measured "
        "separately from per-frame DSP."
    )


def run_synthetic(port: serial.Serial, frames: int) -> None:
    benchmark_lines: list[str] = []
    port.write(f"BENCH {frames}\n".encode("ascii"))
    port.flush()
    read_response(port, 120.0, "BENCH_DONE", benchmark_lines)
    print_benchmark_summary(benchmark_lines)


def run_wav_segment(port: serial.Serial, segment: np.ndarray,
                    wav_path: Path, start_sec: float) -> None:
    benchmark_lines: list[str] = []
    sample_count = len(segment)
    duration_sec = sample_count / SAMPLE_RATE
    print(f"WAV: {wav_path}")
    print(f"Selected real audio: {start_sec:.3f}-{start_sec + duration_sec:.3f}s, "
          f"{sample_count} samples, {duration_sec:.3f}s, float32 mono")
    port.write(f"BENCH_WAV {sample_count}\n".encode("ascii"))
    port.flush()
    read_response(port, 10.0, "ACK_BENCH_START")

    transfer_started = time.monotonic()
    samples_sent = 0
    for offset in range(0, sample_count, MAX_BLOCK_SAMPLES):
        block = segment[offset:offset + MAX_BLOCK_SAMPLES]
        port.write(f"DATA {len(block)}\n".encode("ascii"))
        port.write(block.tobytes())
        port.flush()
        ack = read_response(port, 60.0, "ACK ")
        samples_sent = min(offset + len(block), sample_count)
        if ack != f"ACK {samples_sent}":
            raise RuntimeError(f"Unexpected ACK {ack}; expected ACK {samples_sent}")
        if samples_sent == sample_count or samples_sent // SAMPLE_RATE != offset // SAMPLE_RATE:
            print(f"Laptop sent {samples_sent}/{sample_count} samples "
                  f"({samples_sent / SAMPLE_RATE:.2f}s from selected segment)")

    port.write(b"END\n")
    port.flush()
    read_response(port, 180.0, "BENCH_DONE", benchmark_lines)
    print_benchmark_summary(benchmark_lines)
    elapsed = time.monotonic() - transfer_started
    print(f"WAV_SERIAL_RUN audio_s={duration_sec:.3f} wall_s={elapsed:.2f} "
          f"wall_capacity_x={duration_sec / elapsed:.3f}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(errors="backslashreplace")
    project_root = Path(__file__).resolve().parents[2]
    default_wav = project_root / "data" / "train" / "Train_0.wav"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="ESP32 serial port, e.g. COM4")
    parser.add_argument("--baud", type=int, default=2_000_000)
    parser.add_argument("--wav", type=Path, default=default_wav,
                        help=f"real WAV to profile (default: {default_wav})")
    parser.add_argument("--start-sec", type=float, default=1.0,
                        help="start offset in WAV seconds (default: 1.0)")
    parser.add_argument("--duration-sec", type=float, default=3.05,
                        help="segment duration (default: 3.05; includes a full 3 s peak window)")
    parser.add_argument("--synthetic", action="store_true",
                        help="profile synthetic frames instead of a WAV segment")
    parser.add_argument("--frames", type=int, default=100,
                        help="measured synthetic frames, 1..256 (default: 100)")
    args = parser.parse_args()
    if args.synthetic and not 1 <= args.frames <= MAX_PROFILE_FRAMES:
        parser.error(f"--frames must be between 1 and {MAX_PROFILE_FRAMES}")
    if not args.synthetic and (args.start_sec < 0 or args.duration_sec <= 0):
        parser.error("--start-sec must be >= 0 and --duration-sec must be > 0")

    segment = None
    if not args.synthetic:
        if not args.wav.is_file():
            parser.error(f"WAV file not found: {args.wav}")
        try:
            segment = load_wav_segment(args.wav.resolve(), args.start_sec, args.duration_sec)
        except ValueError as error:
            parser.error(str(error))

    print("This profiles real MCU DSP stages; no microphone is required.")
    print("Close Arduino Serial Monitor before opening the COM port.")
    with serial.Serial(args.port, args.baud, timeout=0.5, write_timeout=10) as port:
        wait_until_ready(port)
        clear_previous_stream(port)
        if args.synthetic:
            run_synthetic(port, args.frames)
        else:
            run_wav_segment(port, segment, args.wav.resolve(), args.start_sec)


if __name__ == "__main__":
    main()
