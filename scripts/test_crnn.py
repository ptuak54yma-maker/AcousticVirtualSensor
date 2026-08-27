"""
===============================================================================
Script: Test Single Audio File with CRNN (scripts/test_crnn.py)
===============================================================================

Runs the Hybrid CRNN pipeline on a target WAV file and plots:
1. Waveform
2. Log-Mel Spectrogram
3. NMF Activation H_event_sum
4. CRNN Frame-wise Probability P(peak) and Detected Events
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import config
from src.audio.loader import load_audio
from src.audio.framing import frame_to_time
from src.runtime.offline import OfflineAudioProcessor


def test_file(wav_path: Path):
    if not wav_path.is_file():
        print(f"[!] Không tìm thấy file âm thanh: {wav_path}")
        return

    print(f"[*] Đang chạy suy luận Hybrid CRNN cho file: {wav_path.name}")
    processor = OfflineAudioProcessor(mode="crnn")
    result = processor.process_file(wav_path)

    total_count = result["total_count"]
    feed_rate = result["feed_rate"]
    rep_frames = result["event_frames"]
    h_event_sum = result["h_event_sum"]
    probs = result["probabilities"]

    total_frames = len(h_event_sum)
    time_axis = frame_to_time(np.arange(total_frames), hop_length=config.HOP_LENGTH, sr=config.SAMPLE_RATE)
    event_times = time_axis[rep_frames] if len(rep_frames) > 0 else np.array([])

    print("\n" + "=" * 50)
    print(f"KẾT QUẢ SUY LUẬN: {wav_path.name}")
    print(f"  - Tổng số phôi đếm được : {total_count} parts")
    print(f"  - Năng suất ước lượng   : {feed_rate:.2f} parts/s")
    print("=" * 50)

    # Trực quan hóa kết quả
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # Đồ thị H_event_sum
    ax1.plot(time_axis, h_event_sum, color="#1f77b4", label=r"$H_{\mathrm{event\_sum}}(t)$")
    ax1.set_title(f"Phân tích CRNN: {wav_path.name} | Số phôi ước lượng: {total_count} (Năng suất: {feed_rate:.2f} parts/s)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Năng lượng NMF", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    # Đồ thị Xác suất CRNN & Mốc sự kiện
    ax2.plot(time_axis, probs, color="#2ca02c", label=r"Xác suất CRNN $\hat{P}(t)$")
    ax2.axhline(y=config.CRNN_THRESHOLD, color="orange", linestyle="--", alpha=0.7, label=f"Ngưỡng ({config.CRNN_THRESHOLD})")
    
    if len(event_times) > 0:
        ax2.scatter(event_times, probs[rep_frames], color="crimson", s=50, zorder=5, label="Phôi phát hiện (Events)")
        for et in event_times:
            ax1.axvline(x=et, color="crimson", linestyle=":", alpha=0.5)

    ax2.set_xlabel("Thời gian (giây)", fontsize=11)
    ax2.set_ylabel("Xác suất $P$", fontsize=11)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiểm thử file WAV đơn lẻ bằng Hybrid CRNN")
    parser.add_argument("--file", type=str, default=str(config.TEST_AUDIO_DIR / config.DEFAULT_TEST_WAV), help="Đường dẫn file WAV cần test")
    args = parser.parse_args()

    test_file(Path(args.file))