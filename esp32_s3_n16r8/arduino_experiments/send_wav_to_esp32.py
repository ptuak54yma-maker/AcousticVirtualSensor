#!/usr/bin/env python3
"""Stream a mono 44.1 kHz WAV to the Arduino ESP32-S3 sketch over serial."""
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


def parse_fields(line: str) -> dict[str, str]:
    return dict(field.split("=", 1) for field in line.split()[1:] if "=" in field)


def collect_telemetry(line: str, telemetry: dict[str, list], plotter=None) -> None:
    if line.startswith("H_EVENT "):
        fields = parse_fields(line)
        try:
            telemetry["event_times"].append(float(fields["t"]))
            telemetry["event_sums"].append(float(fields["sum"]))
        except (KeyError, ValueError):
            print(f"ESP32: malformed event telemetry: {line}")
        return

    if line.startswith("STATUS "):
        fields = parse_fields(line)
        try:
            report = {
                "time": float(fields["t"]),
                "count": int(fields["count_3s"]),
                "feed_rate": float(fields["feed_rate_parts_s"]),
                "event_sum": float(fields["H_event_sum"]),
            }
            telemetry["reports"].append(report)
            peak_times = fields.get("peak_times_s", "-")
            telemetry["window_end"] = report["time"]
            telemetry["window_peak_times"] = (
                [] if peak_times == "-" else [float(value) for value in peak_times.split(",")]
            )
            print(
                f"ESP32: t={report['time']:.2f}s  "
                f"count_3s={report['count']}  "
                f"năng suất={report['feed_rate']:.3f} phôi/s  "
                f"H_event_sum={report['event_sum']:.6f}  "
                f"peak_times_s={peak_times}"
            )
        except (KeyError, ValueError):
            print(f"ESP32: malformed status telemetry: {line}")
        if plotter is not None:
            plotter.refresh(force=True)
        return

    # ACK is a transport confirmation for every 512-sample block, not an inference result.
    if not line.startswith("ACK "):
        print(f"ESP32: {line}")


class LiveResultsPlot:
    def __init__(self, wav_path: Path, telemetry: dict[str, list]):
        self.wav_path = wav_path
        self.telemetry = telemetry
        self.last_refresh = 0.0
        self.available = True
        self.live_display = False
        try:
            import matplotlib
            import matplotlib.pyplot as plt
        except ImportError:
            self.available = False
            print("Matplotlib is not installed. Install it with: python -m pip install matplotlib")
            return

        self.plt = plt
        backend = matplotlib.get_backend().lower()
        self.live_display = backend not in {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}
        self.plt.ion()
        self.fig, (self.event_ax, self.rate_ax) = self.plt.subplots(
            2, 1, figsize=(11, 8), sharex=True
        )
        self.event_line, = self.event_ax.plot(
            [], [], color="tab:blue", linewidth=1.2, label="H_event sum per frame"
        )
        self.peak_scatter = self.event_ax.scatter(
            [], [], color="tab:red", marker="^", s=42,
            label="Detected peaks in 3 s window", zorder=3
        )
        self.rate_line, = self.rate_ax.plot(
            [], [], color="tab:green", marker="o", markersize=3,
            linewidth=1.2, label="Estimated productivity"
        )
        self.event_ax.set_ylabel("H_event sum")
        self.event_ax.set_title(f"ESP32 acoustic analysis: {wav_path.name}")
        self.event_ax.grid(True, alpha=0.3)
        self.event_ax.legend()
        self.rate_ax.set_xlabel("Audio time (s)")
        self.rate_ax.set_ylabel("Parts / s")
        self.rate_ax.grid(True, alpha=0.3)
        self.rate_ax.legend()
        self.fig.tight_layout()
        if self.live_display:
            try:
                self.plt.show(block=False)
                self.plt.pause(0.1)
                self.live_display = self.plt.fignum_exists(self.fig.number)
            except Exception as error:
                self.live_display = False
                print(f"Live graph window unavailable: {error}")
        else:
            print(f"Matplotlib backend {backend} has no interactive graph window.")

    def refresh(self, force: bool = False) -> None:
        if not self.available:
            return
        now = time.monotonic()
        if not force and now - self.last_refresh < 0.2:
            return
        self.last_refresh = now

        window_end = self.telemetry["window_end"]
        if window_end is None:
            return
        window_start = max(0.0, window_end - 3.0)
        all_times = np.asarray(self.telemetry["event_times"], dtype=float)
        all_sums = np.asarray(self.telemetry["event_sums"], dtype=float)
        in_window = (all_times >= window_start) & (all_times <= window_end)
        event_times = all_times[in_window]
        event_sums = all_sums[in_window]
        self.event_line.set_data(event_times, event_sums)
        peak_times = np.asarray(self.telemetry["window_peak_times"], dtype=float)
        peak_times = peak_times[(peak_times >= window_start) & (peak_times <= window_end)]
        if peak_times.size and event_times.size:
            peak_sums = np.interp(peak_times, event_times, event_sums)
            self.peak_scatter.set_offsets(np.column_stack((peak_times, peak_sums)))
        else:
            self.peak_scatter.set_offsets(np.empty((0, 2)))

        reports = [
            report for report in self.telemetry["reports"]
            if window_start <= report["time"] <= window_end
        ]
        self.rate_line.set_data(
            [report["time"] for report in reports],
            [report["feed_rate"] for report in reports],
        )
        self.event_ax.relim()
        self.event_ax.autoscale_view()
        self.rate_ax.relim()
        self.rate_ax.autoscale_view()
        self.event_ax.set_xlim(window_start, window_end)
        try:
            self.fig.canvas.draw_idle()
            if self.live_display:
                self.plt.pause(0.001)
        except Exception as error:
            self.live_display = False
            print(f"Live graph window stopped updating: {error}")

    def finish(self) -> None:
        if not self.available:
            return
        if not self.telemetry["event_times"] or self.telemetry["window_end"] is None:
            print("No H_event telemetry received; no graph to display.")
            return
        self.refresh(force=True)
        if self.live_display and self.plt.fignum_exists(self.fig.number):
            self.plt.ioff()
            self.plt.show()
        else:
            print("The live graph window is unavailable; no image file was saved.")


def read_line(port: serial.Serial, timeout_seconds: float = 20.0) -> str:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        raw = port.readline()
        if raw:
            line = raw.decode("ascii", errors="replace").strip()
            if line:
                return line
    raise TimeoutError("No response from ESP32. Check COM port, sketch and baud rate.")


def wait_for_prefix(port: serial.Serial, prefix: str, telemetry: dict[str, list], plotter,
                    timeout_seconds: float = 20.0) -> str:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        raw = port.readline()
        if not raw:
            continue
        line = raw.decode("ascii", errors="replace").strip()
        if not line:
            continue
        collect_telemetry(line, telemetry, plotter)
        if line.startswith("ERROR"):
            raise RuntimeError(line)
        if line.startswith(prefix):
            return line
    raise TimeoutError(f"Timed out waiting for ESP32 response: {prefix}")


def wait_until_ready(port: serial.Serial, telemetry: dict[str, list], plotter,
                     timeout_seconds: float = 30.0) -> None:
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
            collect_telemetry(line, telemetry, plotter)
        if line.startswith("AVS_READY") or line == "PONG":
            return
    raise TimeoutError("ESP32 did not become ready. Check board, COM port and sketch.")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    project_root = Path(__file__).resolve().parents[2]
    default_wav = project_root / "data" / "train" / "Train_0.wav"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="ESP32 serial port, e.g. COM4")
    parser.add_argument("--wav", type=Path, default=default_wav,
                        help=f"input WAV (default: {default_wav})")
    parser.add_argument("--baud", type=int, default=2_000_000)
    parser.add_argument("--chunk-samples", type=int, default=MAX_BLOCK_SAMPLES,
                        help=f"samples per transfer block (1..{MAX_BLOCK_SAMPLES})")
    parser.add_argument("--fast", action="store_true",
                        help="send each acknowledged block immediately; default paces to 44.1 kHz")
    args = parser.parse_args()

    if not 1 <= args.chunk_samples <= MAX_BLOCK_SAMPLES:
        parser.error(f"--chunk-samples must be between 1 and {MAX_BLOCK_SAMPLES}")
    wav_path = args.wav.resolve()
    if not wav_path.is_file():
        parser.error(f"WAV file not found: {wav_path}")

    audio, rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE:
        parser.error(f"Expected {SAMPLE_RATE} Hz WAV, got {rate} Hz: {wav_path}")
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1, keepdims=True)
    audio = np.ascontiguousarray(audio[:, 0], dtype="<f4")
    if audio.size == 0 or not np.isfinite(audio).all():
        parser.error("WAV must contain finite audio samples")

    duration = audio.size / SAMPLE_RATE
    print(f"WAV: {wav_path}")
    print(f"Audio: mono, {rate} Hz, {audio.size} samples, {duration:.3f} s, float32")
    print("Opening the serial port now; Arduino Serial Monitor must be closed.")

    started = time.monotonic()
    samples_sent = 0
    telemetry: dict[str, list] = {
        "event_times": [],
        "event_sums": [],
        "window_peak_times": [],
        "window_end": None,
        "reports": [],
    }
    plotter = LiveResultsPlot(wav_path, telemetry)
    with serial.Serial(args.port, args.baud, timeout=0.25, write_timeout=10) as port:
        # Opening a USB-UART port may reset the ESP32. PING also handles the
        # case where its one-time ready banner was printed before this script opened.
        wait_until_ready(port, telemetry, plotter, timeout_seconds=30.0)
        port.write(f"START {audio.size}\n".encode("ascii"))
        port.flush()
        wait_for_prefix(port, "ACK_START", telemetry, plotter)
        stream_started = time.monotonic()

        for offset in range(0, audio.size, args.chunk_samples):
            block = audio[offset:offset + args.chunk_samples]
            port.write(f"DATA {block.size}\n".encode("ascii"))
            port.write(block.tobytes())
            port.flush()
            ack = wait_for_prefix(port, "ACK ", telemetry, plotter, timeout_seconds=60.0)
            samples_sent = min(offset + block.size, audio.size)
            expected_ack = f"ACK {samples_sent}"
            if ack != expected_ack:
                raise RuntimeError(f"Unexpected sample acknowledgement: {ack}; expected {expected_ack}")

            if not args.fast:
                deadline = stream_started + samples_sent / SAMPLE_RATE
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)

            if samples_sent == audio.size or samples_sent % SAMPLE_RATE < args.chunk_samples:
                print(f"Laptop sent {samples_sent}/{audio.size} samples ({samples_sent / SAMPLE_RATE:.1f}s audio)")

        port.write(b"END\n")
        port.flush()
        wait_for_prefix(port, "DONE", telemetry, plotter, timeout_seconds=60.0)
        elapsed = time.monotonic() - stream_started

    speed = duration / elapsed if elapsed > 0 else 0.0
    print(f"Transfer/processing time: {elapsed:.2f}s for {duration:.2f}s of audio ({speed:.2f}x audio rate)")
    if speed < 1.0 and not args.fast:
        print("The MCU did not keep up with real-time pacing; acknowledged blocks prevented sample loss.")
    plotter.finish()


if __name__ == "__main__":
    main()
