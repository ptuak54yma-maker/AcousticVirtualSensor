"""
===============================================================================
Script: Grid Search Tuning for Peak Detection (scripts/train_peak_detector.py)
===============================================================================

Optimizes Peak Detection parameters (Prominence P, Distance D) on the Training Set
against ground-truth counts from data/train/label.txt.
Saves optimal parameters P*, D* into models/peak_detection/peak_params.json.
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy.signal import find_peaks
import config
from src.audio.loader import load_audio
from src.preprocessing.stft import compute_stft
from src.preprocessing.log_compression import log_compression
from src.nmf.dictionary import load_dictionary
from src.nmf.nnls import solve_nnls_batch
from src.nmf.activation import compute_h_event_sum


def parse_label_txt(label_file_path: Path) -> dict:
    """Đọc file label.txt và chuyển thành dictionary {filename: count}."""
    ground_truth = {}
    if not label_file_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file nhãn: {label_file_path}")

    with open(label_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            parts = line.split(":")
            fname = parts[0].strip()
            # Tự động bổ sung đuôi .wav nếu thiếu
            if not fname.lower().endswith(".wav"):
                fname = f"{fname}.wav"
            try:
                cnt = int(parts[1].strip())
                ground_truth[fname] = cnt
            except ValueError:
                continue
    return ground_truth


def main():
    print("=" * 70)
    print("GIAI ĐOẠN 2 & 3: TỐI ƯU HÓA THAM SỐ PEAK DETECTION (GRID SEARCH P, D)")
    print("=" * 70)

    # 1. Đọc nhãn ground-truth của tập Train
    train_label_path = config.TRAIN_AUDIO_DIR / "label.txt"
    ground_truth_counts = parse_label_txt(train_label_path)
    print(f"[*] Đã đọc {len(ground_truth_counts)} nhãn ground-truth từ: {train_label_path}")

    # 2. Nạp ma trận W_standard
    if not config.W_STANDARD_PATH.is_file():
        print(f"[!] Lỗi: Không tìm thấy {config.W_STANDARD_PATH}. Hãy chạy scripts/train_nmf.py trước.")
        return
    w_standard = load_dictionary(config.W_STANDARD_PATH)

# 3. Trích xuất và Cache H_event_sum cho từng file Train
    train_samples = []
    print("\n[*] Đang tính toán và cache H_event_sum cho các file Train_*.wav...")
    
    for fname, true_count in ground_truth_counts.items():
        # Chỉ xử lý các file thuộc tập Train (bỏ qua Bowl_*)
        if not fname.startswith("Train_"):
            continue

        wav_path = config.TRAIN_AUDIO_DIR / fname
        if not wav_path.is_file():
            print(f"    [!] Cảnh báo: File {fname} trong label.txt không tồn tại trên ổ cứng. Bỏ qua.")
            continue

        audio, _ = load_audio(wav_path, target_sr=config.SAMPLE_RATE)
        stft_matrix = compute_stft(audio)
        v_matrix = log_compression(stft_matrix)
        h_matrix = solve_nnls_batch(v_matrix, w_standard)
        h_event_sum = compute_h_event_sum(h_matrix)

        train_samples.append({
            "name": fname,
            "h_event_sum": h_event_sum,
            "true_count": true_count
        })
        print(f"    [✓] Đã cache: {fname} | True Count: {true_count} | Khung phổ: {len(h_event_sum)}")

    if not train_samples:
        print("[!] Không có mẫu huấn luyện hợp lệ. Quá trình dừng lại.")
        return

    # 4. Thiết lập không gian tìm kiếm Grid Search
    prominences = np.linspace(
        config.GRID_SEARCH_PROMINENCES[0],
        config.GRID_SEARCH_PROMINENCES[1],
        int(config.GRID_SEARCH_PROMINENCES[2])
    )
    distances = np.arange(
        config.GRID_SEARCH_DISTANCES[0],
        config.GRID_SEARCH_DISTANCES[1],
        int(config.GRID_SEARCH_DISTANCES[2])
    )

    total_combinations = len(prominences) * len(distances)
    print(f"\n[*] Bắt đầu Grid Search trên {total_combinations} tổ hợp (P: 50 mức, D: 50 mức)...")

    best_mae = float("inf")
    best_p = None
    best_d = None

    # 5. Vòng lặp tìm kiếm tham số tối ưu
    for p in prominences:
        for d in distances:
            abs_errors = []
            for sample in train_samples:
                peaks, _ = find_peaks(sample["h_event_sum"], prominence=p, distance=d)
                pred_count = len(peaks)
                abs_errors.append(abs(pred_count - sample["true_count"]))

            mae = float(np.mean(abs_errors))
            if mae < best_mae:
                best_mae = mae
                best_p = float(p)
                best_d = int(d)

    print("\n" + "=" * 70)
    print("KẾT QUẢ TỐI ƯU HÓA THÀNH CÔNG:")
    print(f"  - Best Prominence (P*) : {best_p:.4f}")
    print(f"  - Best Distance   (D*) : {best_d} frames (~{best_d * config.HOP_LENGTH / config.SAMPLE_RATE * 1000:.1f} ms)")
    print(f"  - Minimum Train MAE    : {best_mae:.4f} phôi")
    print("=" * 70)

    # 6. Lưu kết quả vào file JSON chính thức
    out_payload = {
        "prominence": round(best_p, 4),
        "distance": int(best_d),
        "mae": round(best_mae, 4),
        "total_train_files": len(train_samples)
    }

    with open(config.PEAK_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=4)

    print(f"[✓] Đã lưu bộ tham số chuẩn vào: {config.PEAK_PARAMS_PATH}\n")


if __name__ == "__main__":
    main()