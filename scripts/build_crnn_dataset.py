"""
===============================================================================
Script: Build CRNN Training Dataset (scripts/build_crnn_dataset.py)
===============================================================================

Directly consumes precomputed features from data/features/train/
and manual annotations from data/annotations/train/.
Eliminates redundant STFT / Audio loading.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import config
from src.annotation.annotation_io import load_annotation, create_gaussian_label
from src.crnn.dataset import extract_hybrid_windows

TRAIN_FEATURES_DIR = config.DATA_DIR / "features" / "train"


def main():
    print("=" * 70)
    print("GIAI ĐOẠN 5: ĐÓNG GÓI TẬP DỮ LIỆU HUẤN LUYỆN CRNN (TỪ FEATURE CACHE)")
    print("=" * 70)

    feature_files = sorted(list(TRAIN_FEATURES_DIR.glob("Train_*_features.npz")))
    if not feature_files:
        print(f"[!] Không tìm thấy feature cache nào trong {TRAIN_FEATURES_DIR}.")
        print("    Vui lòng chạy 'python scripts/train_peak_detector.py' trước để sinh cache.")
        return

    all_x_mel = []
    all_x_he = []
    all_y = []

    print(f"[*] Tìm thấy {len(feature_files)} tệp đặc trưng. Tiến hành nạp và cắt cửa sổ...")

    for feat_path in feature_files:
        # Lấy tên file tương ứng (ví dụ: Train_0)
        stem = feat_path.name.replace("_features.npz", "")
        annot_path = config.TRAIN_ANNOTATION_DIR / f"{stem}_labels.npz"

        if not annot_path.is_file():
            print(f"    [!] Cảnh báo: Thiếu file nhãn {annot_path.name}. Hãy chạy scripts/annotate_train.py trước. Bỏ qua.")
            continue

        # 1. Nạp trực tiếp đặc trưng (Zero STFT re-computation)
        with np.load(str(feat_path)) as feat_data:
            log_mel = feat_data["log_mel"]          # (128, T)
            h_event_sum = feat_data["h_event_sum"]  # (T,)

        # 2. Nạp nhãn gán thủ công
        annot_data = load_annotation(annot_path)
        binary_labels = annot_data["binary_labels"]

        # Đồng bộ độ dài an toàn
        min_len = min(log_mel.shape[1], len(h_event_sum), len(binary_labels))
        log_mel = log_mel[:, :min_len]
        h_event_sum = h_event_sum[:min_len]
        binary_labels = binary_labels[:min_len]

        # 3. Làm mịn nhãn Gaussian
        if config.ANNOTATION_SMOOTHING_ENABLED:
            peak_frames = np.where(binary_labels == 1)[0]
            target_labels = create_gaussian_label(
                peak_frames=peak_frames,
                total_frames=min_len,
                sigma=config.ANNOTATION_GAUSSIAN_SIGMA
            )
        else:
            target_labels = binary_labels.astype(np.float32)

        # 4. Cắt cửa sổ trượt (L=128, Hop=64)
        x_mel_wins, x_he_wins, y_wins = extract_hybrid_windows(
            log_mel=log_mel,
            h_event_sum=h_event_sum,
            labels=target_labels,
            window_len=config.SEQUENCE_LENGTH,
            hop_size=config.SEQUENCE_HOP
        )

        all_x_mel.append(x_mel_wins)
        all_x_he.append(x_he_wins)
        all_y.append(y_wins)
        print(f"    [✓] {stem:<10} -> {len(y_wins)} windows ({config.SEQUENCE_LENGTH} frames)")

    if not all_y:
        print("[!] Không có dữ liệu hợp lệ để đóng gói. Dừng.")
        return

    # 5. Gom cụm toàn bộ dataset
    x_mel_train = np.concatenate(all_x_mel, axis=0)
    x_he_train = np.concatenate(all_x_he, axis=0)
    y_train = np.concatenate(all_y, axis=0)

    # 6. Tính pos_weight cho hàm mất mát
    pos_samples = np.sum(y_train > 0.5)
    total_elements = y_train.size
    neg_samples = total_elements - pos_samples
    pos_weight = float(neg_samples / max(1, pos_samples))

    output_dataset = config.CRNN_DATASET_DIR / "crnn_train.npz"
    config.CRNN_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        str(output_dataset),
        X_mel_train=x_mel_train,
        X_he_train=x_he_train,
        Y_train=y_train,
        pos_weight=pos_weight
    )

    print("\n" + "=" * 70)
    print(f"[✓] ĐÃ ĐÓNG GÓI THÀNH CÔNG: {output_dataset}")
    print(f"  - X_mel_train Shape : {x_mel_train.shape} (N, 1, 128, L)")
    print(f"  - X_he_train Shape  : {x_he_train.shape} (N, 1, L)")
    print(f"  - Y_train Shape     : {y_train.shape} (N, L)")
    print(f"  - Class pos_weight  : {pos_weight:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()