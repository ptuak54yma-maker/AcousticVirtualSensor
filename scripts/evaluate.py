"""
===============================================================================
Script: Evaluation & Benchmark (scripts/evaluate.py)
===============================================================================

Runs NMF + Peak Detection pipeline on the independent test set,
evaluates Count-level metrics (MAE, MAPE, Accuracy) against data/test/label.txt,
and optionally evaluates Event-level F1 if frame-level labels exist.
Outputs results directly formatted for the thesis.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import config
from src.audio.loader import load_audio
from src.annotation.annotation_io import load_annotation
from src.runtime.sensor import AcousticVirtualSensorCore
from src.evaluation.count_metrics import compute_count_metrics
from src.evaluation.event_metrics import compute_event_metrics


def parse_label_txt(label_file_path: Path) -> dict:
    """Đọc file label.txt và chuyển thành dictionary {filename: count}."""
    ground_truth = {}
    if not label_file_path.is_file():
        return ground_truth

    with open(label_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            parts = line.split(":")
            fname = parts[0].strip()
            if not fname.lower().endswith(".wav"):
                fname = f"{fname}.wav"
            try:
                cnt = int(parts[1].strip())
                ground_truth[fname] = cnt
            except ValueError:
                continue
    return ground_truth


def run_benchmark():
    test_files = sorted(list(config.TEST_AUDIO_DIR.glob("*.wav")))
    if not test_files:
        print(f"[!] Không tìm thấy file WAV nào trong thư mục kiểm thử: {config.TEST_AUDIO_DIR}")
        return

    # Nạp Ground Truth Count từ data/test/label.txt
    test_label_file = config.TEST_AUDIO_DIR / "label.txt"
    test_gt_counts = parse_label_txt(test_label_file)

    print(f"[*] Khởi tạo Acoustic Virtual Sensor (NMF + Peak Detection)...")
    print(f"[*] Tìm thấy {len(test_files)} file test. Nhãn ground truth có sẵn: {len(test_gt_counts)} files.")
    sensor = AcousticVirtualSensorCore(mode="baseline_nmf")

    results = []

    for wav_path in test_files:
        audio, _ = load_audio(wav_path, target_sr=config.SAMPLE_RATE)
        
        # 1. Chạy suy luận qua pipeline NMF + Peak Detection
        res = sensor.process_offline_file(audio)
        pred_count = res["total_count"]
        pred_events = res["event_frames"]
        duration = res["duration_sec"]
        feed_rate = res["feed_rate"]

        # 2. Đọc ground truth count
        true_count = test_gt_counts.get(wav_path.name, None)

        record = {
            "Filename": wav_path.name,
            "Duration (s)": round(duration, 2),
            "Feed Rate (p/s)": round(feed_rate, 2),
            "True Count": true_count if true_count is not None else "N/A",
            "Pred Count": pred_count,
            "Abs Error": abs(pred_count - true_count) if true_count is not None else "N/A"
        }

        # 3. Tính Event-level F1 nếu tồn tại nhãn frame-level
        annot_path = config.TEST_ANNOTATION_DIR / f"{wav_path.stem}_labels.npz"
        if annot_path.is_file():
            annot_data = load_annotation(annot_path)
            true_event_frames = np.where(annot_data["binary_labels"] == 1)[0]
            ev_metrics = compute_event_metrics(true_event_frames, pred_events)
            record["Precision"] = round(ev_metrics["event_precision"], 4)
            record["Recall"] = round(ev_metrics["event_recall"], 4)
            record["Event F1"] = round(ev_metrics["event_f1"], 4)

        results.append(record)

    df = pd.DataFrame(results)
    print("\n" + "=" * 85)
    print("BẢNG KẾT QUẢ ĐÁNH GIÁ NMF + PEAK DETECTION TRÊN TẬP TEST")
    print("=" * 85)
    print(df.to_string(index=False))

    # 4. Tính toán các chỉ số tổng hợp
    valid_records = [r for r in results if r["True Count"] != "N/A"]
    if valid_records:
        y_true = [r["True Count"] for r in valid_records]
        y_pred = [r["Pred Count"] for r in valid_records]

        metrics = compute_count_metrics(y_true, y_pred)

        print("\n" + "-" * 85)
        print("TỔNG KẾT HIỆU NĂNG ĐẾM SẢN PHẨM:")
        print(f"  - Số lượng file đánh giá: {len(valid_records)}")
        print(f"  - Mean Accuracy         : {metrics['mean_accuracy']:.2f}%")
        print(f"  - MAE                   : {metrics['mae']:.2f} parts")
        print(f"  - MAPE                  : {metrics['mape']:.2f}%")
        print(f"  - Tổng phôi thực tế     : {metrics['total_true']}")
        print(f"  - Tổng phôi đếm được    : {metrics['total_pred']}")
        print("-" * 85)

    # 5. Xuất file kết quả CSV
    csv_file = config.OUTPUT_DIR / "results" / "comparison_results.csv"
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(csv_file), index=False)
    print(f"\n[✓] Đã xuất báo cáo chi tiết ra file: {csv_file}")


if __name__ == "__main__":
    run_benchmark()