"""
===============================================================================
Script: Train Peak Detector with Persistent Feature Cache (scripts/train_peak_detector.py)
===============================================================================

Workflow:
1. Load Train_*.wav and parse data/train/label.txt.
2. Single-pass STFT -> Log-Mel & H_event_sum.
3. Save features persistently to data/features/train/Train_*_features.npz.
4. Run 2D Grid Search (P x D) on H_event_sum to minimize MAE.
5. Save optimal P*, D* to models/peak_detection/peak_params.json.
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
from src.nmf.dictionary import load_dictionary
from src.preprocessing.features import extract_features_from_audio, save_train_features

TRAIN_FEATURES_DIR = config.DATA_DIR / "features" / "train"


def parse_label_txt(label_file_path: Path) -> dict:
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
            if not fname.lower().endswith(".wav"):
                fname = f"{fname}.wav"
            try:
                ground_truth[fname] = int(parts[1].strip())
            except ValueError:
                continue
    return ground_truth


def main():
    print("=" * 70)
    print("TỐI ƯU HÓA PEAK DETECTION & LƯU TRỮ FEATURE CACHE CHO CRNN")
    print("=" * 70)

    # 1. Đọc nhãn ground truth
    train_label_path = config.TRAIN_AUDIO_DIR / "label.txt"
    ground_truth_counts = parse_label_txt(train_label_path)

    # 2. Nạp ma trận W_standard
    if not config.W_STANDARD_PATH.is_file():
        print(f"[!] Lỗi: Không tìm thấy {config.W_STANDARD_PATH}. Hãy chạy scripts/train_nmf.py trước.")
        return
    w_standard = load_dictionary(config.W_STANDARD_PATH)

    # 3. Trích xuất đặc trưng (1 pass STFT) và lưu Cache
    train_samples = []
    print("\n[*] Đang trích xuất đặc trưng hợp nhất và lưu vào data/features/train/...")

    for fname, true_count in ground_truth_counts.items():
        if not fname.startswith("Train_"):
            continue

        wav_path = config.TRAIN_AUDIO_DIR / fname
        if not wav_path.is_file():
            print(f"    [!] File {fname} không tồn tại trên ổ cứng. Bỏ qua.")
            continue

        audio, sr = load_audio(wav_path, target_sr=config.SAMPLE_RATE)
        
        # Gọi hàm xử lý hợp nhất (STFT 1 lần duy nhất)
        log_mel, h_event_sum, frame_times = extract_features_from_audio(audio, w_standard, sr)

        # Lưu đệm xuống đĩa cho CRNN
        feat_path = TRAIN_FEATURES_DIR / f"{wav_path.stem}_features.npz"
        save_train_features(
            output_path=feat_path,
            filename=fname,
            log_mel=log_mel,
            h_event_sum=h_event_sum,
            frame_times=frame_times,
            true_count=true_count
        )

        train_samples.append({
            "name": fname,
            "h_event_sum": h_event_sum,
            "true_count": true_count
        })
        print(f"    [✓] Đã cache: {feat_path.name} | Log-Mel: {log_mel.shape} | Khung: {len(h_event_sum)}")

    if not train_samples:
        print("[!] Không có mẫu huấn luyện hợp lệ. Dừng.")
        return

    # 4. Grid Search P, D (Thuật toán giữ nguyên hoàn toàn)
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

    print(f"\n[*] Bắt đầu Grid Search {len(prominences) * len(distances)} tổ hợp...")
    best_mae = float("inf")
    best_p = None
    best_d = None

    for p in prominences:
        for d in distances:
            abs_errors = []
            for sample in train_samples:
                peaks, _ = find_peaks(sample["h_event_sum"], prominence=p, distance=d)
                abs_errors.append(abs(len(peaks) - sample["true_count"]))

            mae = float(np.mean(abs_errors))
            if mae < best_mae:
                best_mae = mae
                best_p = float(p)
                best_d = int(d)

    print("\n" + "=" * 70)
    print(f"KẾT QUẢ TỐI ƯU: P* = {best_p:.4f} | D* = {best_d} frames | MAE = {best_mae:.4f}")
    print("=" * 70)

    # 5. Lưu kết quả ra JSON
    out_payload = {
        "prominence": round(best_p, 4),
        "distance": int(best_d),
        "mae": round(best_mae, 4),
        "total_train_files": len(train_samples)
    }
    with open(config.PEAK_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=4)
    print(f"[✓] Đã lưu tham số tối ưu vào: {config.PEAK_PARAMS_PATH}")


if __name__ == "__main__":
    main()