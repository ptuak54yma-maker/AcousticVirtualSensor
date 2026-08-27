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


def run_benchmark():
    test_files = sorted(list(config.TEST_AUDIO_DIR.glob("*.wav")))
    if not test_files:
        print(f"[!] Không tìm thấy file WAV nào trong thư mục kiểm thử: {config.TEST_AUDIO_DIR}")
        return

    print(f"[*] Khởi tạo 2 cảm biến ảo đối chuẩn trên {len(test_files)} file test...")
    sensor_baseline = AcousticVirtualSensorCore(mode="baseline_nmf")
    sensor_crnn = AcousticVirtualSensorCore(mode="crnn")

    results = []

    for wav_path in test_files:
        audio, _ = load_audio(wav_path, target_sr=config.SAMPLE_RATE)
        
        # Load ground truth if exists
        annot_path = config.TEST_ANNOTATION_DIR / f"{wav_path.stem}_labels.npz"
        true_count = None
        true_event_frames = []
        if annot_path.is_file():
            annot_data = load_annotation(annot_path)
            true_event_frames = np.where(annot_data["binary_labels"] == 1)[0]
            true_count = len(true_event_frames)

        # 1. Baseline Inference
        res_base = sensor_baseline.process_offline_file(audio)
        
        # 2. CRNN Inference
        res_crnn = sensor_crnn.process_offline_file(audio)

        record = {
            "Filename": wav_path.name,
            "Duration (s)": round(len(audio) / config.SAMPLE_RATE, 2),
            "True Count": true_count,
            "Baseline Count": res_base["total_count"],
            "CRNN Count": res_crnn["total_count"]
        }

        # Event-level F1 if labels available
        if true_count is not None:
            f1_base = compute_event_metrics(true_event_frames, res_base["event_frames"])["event_f1"]
            f1_crnn = compute_event_metrics(true_event_frames, res_crnn["event_frames"])["event_f1"]
            record["Baseline Event F1"] = round(f1_base, 4)
            record["CRNN Event F1"] = round(f1_crnn, 4)

        results.append(record)

    df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("BẢNG KẾT QUẢ ĐỐI CHUẨN THỰC NGHIỆM TRÊN TẬP TEST")
    print("=" * 80)
    print(df.to_string(index=False))

    # Calculate aggregate count metrics if ground truth is available
    if all(r["True Count"] is not None for r in results):
        y_true = [r["True Count"] for r in results]
        base_counts = [r["Baseline Count"] for r in results]
        crnn_counts = [r["CRNN Count"] for r in results]

        m_base = compute_count_metrics(y_true, base_counts)
        m_crnn = compute_count_metrics(y_true, crnn_counts)

        print("\n" + "-" * 80)
        print(f"Tổng kết Baseline NMF + Peak Detection : Accuracy = {m_base['mean_accuracy']:.2f}% | MAE = {m_base['mae']:.2f} | MAPE = {m_base['mape']:.2f}%")
        print(f"Tổng kết Proposed NMF + CRNN           : Accuracy = {m_crnn['mean_accuracy']:.2f}% | MAE = {m_crnn['mae']:.2f} | MAPE = {m_crnn['mape']:.2f}%")
        print("-" * 80)

    # Save to CSV
    csv_file = config.EVAL_RESULTS_CSV
    df.to_csv(str(csv_file), index=False)
    print(f"\n[✓] Đã xuất báo cáo so sánh chi tiết ra file: {csv_file}")


if __name__ == "__main__":
    run_benchmark()