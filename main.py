"""
===============================================================================
Acoustic Virtual Sensor - Main Application Entry Point (main.py)
===============================================================================

Orchestrator for the entire Acoustic Virtual Sensor project.
Handles CLI dispatching, Offline Benchmarks, and GUI execution.
"""

import sys
import argparse
import logging
import queue
import threading
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.runtime.offline import OfflineAudioProcessor
from src.runtime.streaming import StreamingSensorEngine

# Configure Logging
log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format=log_format,
    handlers=[
        logging.FileHandler(config.LOG_DIR / config.LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Main")


def run_cli_mode(wav_path: Path, mode: str = "crnn"):
    """Run single file offline processing via CLI."""
    logger.info("Khởi chạy chế độ CLI Offline với mode: %s", mode)
    if not wav_path.is_file():
        logger.error("Không tìm thấy file WAV tại: %s", wav_path)
        return

    processor = OfflineAudioProcessor(mode=mode)
    result = processor.process_file(wav_path)

    print("\n" + "=" * 55)
    print(f"KẾT QUẢ ĐO CẢM BIẾN ẢO: {result['filename']}")
    print(f"  - Thời lượng ghi âm    : {result['duration_sec']:.2f} s")
    print(f"  - Tổng số phôi đếm được: {result['total_count']} parts")
    print(f"  - Năng suất trung bình : {result['feed_rate']:.2f} parts/s")
    print("=" * 55 + "\n")


def run_gui_mode(mode: str = "crnn"):
    """Launch GUI Dashboard with thread-safe worker."""
    import tkinter as tk
    from tkinter import ttk
    import soundfile as sf
    import numpy as np

    logger.info("Khởi chạy giao diện Tkinter Dashboard...")
    root = tk.Tk()
    root.title(f"{config.WINDOW_TITLE} (Mode: {mode.upper()})")
    root.geometry(f"{config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT}")
    root.resizable(False, False)

    # Threading and Communication
    data_queue = queue.Queue(maxsize=20)
    stop_event = threading.Event()
    engine = StreamingSensorEngine(mode="baseline_nmf" if mode == "baseline" else "crnn")

    # GUI State Variables
    var_count = tk.StringVar(value="0")
    var_rate = tk.StringVar(value="0.00 parts/s")
    var_he = tk.StringVar(value="0.000")
    var_status = tk.StringVar(value="● STOPPED")

    # Build Simple UI Layout
    tk.Label(root, text="Acoustic Virtual Sensor Dashboard", font=(config.FONT_NAME, config.FONT_SIZE_TITLE, "bold")).pack(pady=15)
    
    ttk.Separator(root, orient="horizontal").pack(fill="x", padx=20, pady=5)
    tk.Label(root, text="NĂNG SUẤT TỨC THỜI (FEED RATE)", font=(config.FONT_NAME, config.FONT_SIZE_LABEL)).pack()
    tk.Label(root, textvariable=var_rate, font=(config.FONT_NAME, config.FONT_SIZE_VALUE, "bold"), fg="#1f77b4").pack(pady=5)

    ttk.Separator(root, orient="horizontal").pack(fill="x", padx=20, pady=5)
    tk.Label(root, text="TỔNG SỐ PHÔI ĐẾM TÍCH LŨY (PARTS COUNT)", font=(config.FONT_NAME, config.FONT_SIZE_LABEL)).pack()
    tk.Label(root, textvariable=var_count, font=(config.FONT_NAME, 24, "bold"), fg="#2ca02c").pack(pady=5)

    ttk.Separator(root, orient="horizontal").pack(fill="x", padx=20, pady=5)
    tk.Label(root, text="TRẠNG THÁI HỆ THỐNG", font=(config.FONT_NAME, config.FONT_SIZE_LABEL)).pack()
    status_label = tk.Label(root, textvariable=var_status, font=(config.FONT_NAME, 14, "bold"), fg="red")
    status_label.pack(pady=5)

    # Worker Thread Logic
    def worker_loop():
        wav_file = config.TEST_AUDIO_DIR / config.DEFAULT_TEST_WAV
        if not wav_file.is_file():
            logger.error("Không tìm thấy file mặc định để stream: %s", wav_file)
            return

        audio, _ = sf.read(str(wav_file), dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        idx = 0
        chunk_sz = config.STREAM_CHUNK_SIZE
        while not stop_event.is_set() and idx < len(audio):
            chunk = audio[idx:idx + chunk_sz]
            idx += chunk_sz
            if len(chunk) < chunk_sz:
                chunk = np.pad(chunk, (0, chunk_sz - len(chunk)))

            state = engine.process_chunk(chunk)
            try:
                data_queue.put_nowait(state)
            except queue.Full:
                pass
            threading.Event().wait(chunk_sz / config.SAMPLE_RATE)  # Real-time sync

    worker_thread = None

    def on_start():
        nonlocal worker_thread
        if worker_thread is None or not worker_thread.is_alive():
            stop_event.clear()
            engine.reset()
            worker_thread = threading.Thread(target=worker_loop, daemon=True)
            worker_thread.start()
            var_status.set("● RUNNING")
            status_label.config(fg="green")

    def on_stop():
        stop_event.set()
        var_status.set("● STOPPED")
        status_label.config(fg="red")

    def on_reset():
        on_stop()
        engine.reset()
        var_count.set("0")
        var_rate.set("0.00 parts/s")
        var_he.set("0.000")

    # Control Buttons
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=20)
    tk.Button(btn_frame, text="Bắt đầu (Start)", width=12, bg="#e1e1e1", command=on_start).grid(row=0, column=0, padx=10)
    tk.Button(btn_frame, text="Tạm dừng (Stop)", width=12, bg="#e1e1e1", command=on_stop).grid(row=0, column=1, padx=10)
    tk.Button(btn_frame, text="Đặt lại (Reset)", width=12, bg="#e1e1e1", command=on_reset).grid(row=0, column=2, padx=10)

    # Queue Polling
    def poll_queue():
        while not data_queue.empty():
            try:
                state = data_queue.get_nowait()
                var_count.set(str(state["count"]))
                var_rate.set(f"{state['feed_rate']:.2f} parts/s")
                var_he.set(f"{state['latest_he']:.3f}")
            except queue.Empty:
                break
        root.after(config.REFRESH_RATE_MS, poll_queue)

    def on_close():
        stop_event.set()
        if worker_thread is not None and worker_thread.is_alive():
            worker_thread.join(timeout=0.5)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(100, poll_queue)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Chương trình Cảm biến Ảo Âm thanh (Acoustic Virtual Sensor)")
    parser.add_argument("--mode", type=str, choices=["crnn", "baseline"], default="crnn", help="Lựa chọn phương pháp xử lý")
    parser.add_argument("--file", type=str, default=None, help="Đường dẫn file WAV cần xử lý")
    parser.add_argument("--gui", action="store_true", help="Chạy ở chế độ giao diện đồ họa GUI")

    args = parser.parse_args()

    if args.gui or config.ENABLE_GUI and args.file is None:
        run_gui_mode(mode=args.mode)
    else:
        target_wav = Path(args.file) if args.file else config.TEST_AUDIO_DIR / config.DEFAULT_TEST_WAV
        run_cli_mode(target_wav, mode=args.mode)


if __name__ == "__main__":
    main()