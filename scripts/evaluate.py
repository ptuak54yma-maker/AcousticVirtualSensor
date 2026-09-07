"""
===============================================================================
Script: Full Evaluation & Benchmark (scripts/evaluate.py)
===============================================================================

Runs both Baseline (NMF + Peak Detection) and Proposed (Hybrid NMF + CRNN) 
on the independent test set, computes Frame/Event/Count metrics, and outputs 
comparison tables directly formatted for the thesis.
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
    """Đọc ground-truth count từ file label.txt nếu có."""
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
            try:
                ground_truth[fname] = int(parts[1].strip())
            except ValueError:
                continue
    return ground_truth


def run_benchmark():
    test_files = sorted(list(config.TEST_AUDIO_DIR.glob("*.wav")))
    if not test_files:
        print(f"[!] Không tìm thấy file WAV nào trong thư mục kiểm thử: {config.TEST_AUDIO_DIR}")
        return

    # 1. Đọc ground-truth count từ data/test/label.txt
    test_gt_counts = parse_label_txt(config.TEST_AUDIO_DIR / "label.txt")

    print(f"[*] Khởi tạo 2 cảm biến ảo đối chuẩn trên {len(test_files)} file test...")
    sensor_baseline = AcousticVirtualSensorCore(mode="baseline_nmf")
    
    # Kiểm tra an toàn trước khi khởi tạo CRNN
    has_crnn = config.CRNN_MODEL_PATH.is_file()
    sensor_crnn = AcousticVirtualSensorCore(mode="crnn") if has_crnn else None
    if not has_crnn:
        print(f"[!] Cảnh báo: Chưa tìm thấy {config.CRNN_MODEL_PATH}. Chỉ đánh giá Baseline.")

    results = []

    for wav_path in test_files:
        audio, _ = load_audio(wav_path, target_sr=config.SAMPLE_RATE)
        
        # Ưu tiên lấy ground truth từ label.txt, sau đó mới đến file .npz
        true_count = test_gt_counts.get(wav_path.name, None)
        true_event_frames = []
        
        annot_path = config.TEST_ANNOTATION_DIR / f"{wav_path.stem}_labels.npz"
        if annot_path.is_file():
            annot_data = load_annotation(annot_path)
            true_event_frames = np.where(annot_data["binary_labels"] == 1)[0]
            if true_count is None:
                true_count = len(true_event_frames)

        # 1. Baseline Inference
        res_base = sensor_baseline.process_offline_file(audio)
        
        # 2. CRNN Inference
        res_crnn = sensor_crnn.process_offline_file(audio) if sensor_crnn else None

        base_count = res_base["total_count"]
        crnn_count = res_crnn["total_count"] if res_crnn else None

        record = {
            "Filename": wav_path.name,
            "Duration (s)": round(len(audio) / config.SAMPLE_RATE, 2),
            "True Count": true_count,
            "Baseline Count": base_count,
            "CRNN Count": crnn_count
        }

        # Event-level F1 if frame-level labels available
        if len(true_event_frames) > 0:
            f1_base = compute_event_metrics(true_event_frames, res_base["event_frames"])["event_f1"]
            record["Baseline Event F1"] = round(f1_base, 4)
            if res_crnn:
                f1_crnn = compute_event_metrics(true_event_frames, res_crnn["event_frames"])["event_f1"]
                record["CRNN Event F1"] = round(f1_crnn, 4)

        results.append(record)

    df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("BẢNG KẾT QUẢ ĐỐI CHUẨN THỰC NGHIỆM TRÊN TẬP TEST")
    print("=" * 80)
    print(df.to_string(index=False))

    # Calculate aggregate count metrics if ground truth is available
    if any(r["True Count"] is not None for r in results):
        valid_results = [r for r in results if r["True Count"] is not None]
        y_true = [r["True Count"] for r in valid_results]
        base_counts = [r["Baseline Count"] for r in valid_results]

        m_base = compute_count_metrics(y_true, base_counts)
        print("\n" + "-" * 80)
        print(f"Tổng kết Baseline NMF + Peak Detection : Accuracy = {m_base['mean_accuracy']:.2f}% | MAE = {m_base['mae']:.2f} | MAPE = {m_base['mape']:.2f}%")

        if sensor_crnn:
            crnn_counts = [r["CRNN Count"] for r in valid_results]
            m_crnn = compute_count_metrics(y_true, crnn_counts)
            print(f"Tổng kết Proposed NMF + CRNN           : Accuracy = {m_crnn['mean_accuracy']:.2f}% | MAE = {m_crnn['mae']:.2f} | MAPE = {m_crnn['mape']:.2f}%")
        print("-" * 80)

    # Save to CSV
    csv_file = config.EVAL_RESULTS_CSV
    df.to_csv(str(csv_file), index=False)
    print(f"\n[✓] Đã xuất báo cáo so sánh chi tiết ra file: {csv_file}")


if __name__ == "__main__":
    run_benchmark()