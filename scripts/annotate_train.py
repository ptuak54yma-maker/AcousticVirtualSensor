"""
===============================================================================
Script: Standalone Label Annotation Tool (scripts/annotate_train.py)
===============================================================================

Workflow:
1. Load W_standard.npy to compute H_event_sum (cached for high performance).
2. Read Ground Truth Count from data/train/label.txt for reference.
3. Load optimal peak parameters (P*, D*) from models/peak_detection/peak_params.json.
4. Launch Interactive GUI with Audio Playback, Real-time Cursor, and Tri-Count display.
5. Save validated frame-level labels into data/annotations/train/*_labels.npz.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import config
from src.audio.loader import load_audio
from src.audio.framing import frame_to_time
from src.preprocessing.stft import compute_stft
from src.preprocessing.log_compression import log_compression
from src.nmf.dictionary import load_dictionary
from src.nmf.nnls import solve_nnls_batch
from src.nmf.activation import compute_h_event_sum
from src.annotation.peak_reference import detect_reference_peaks, load_peak_parameters
from src.annotation.manual_label import InteractiveAnnotator
from src.annotation.annotation_io import save_annotation, load_annotation

# Thư mục lưu cache đặc trưng
FEATURE_CACHE_DIR = config.DATA_DIR / "features" / "train"
FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def parse_label_txt(label_file_path: Path) -> dict:
    """Đọc file label.txt và chuẩn hóa đuôi mở rộng .wav."""
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
                ground_truth[fname] = int(parts[1].strip())
            except ValueError:
                continue
    return ground_truth


def get_or_compute_features(wav_path: Path, w_matrix: np.ndarray):
    """Nạp đặc trưng NMF từ cache hoặc tính mới nếu chưa có."""
    feat_file = FEATURE_CACHE_DIR / f"{wav_path.stem}_feat.npz"

    if feat_file.is_file():
        data = np.load(str(feat_file))
        return data["h_event_sum"], data["frame_times"]

    print(f"    [*] Đang trích xuất đặc trưng NMF lần đầu cho {wav_path.name}...")
    audio, sr = load_audio(wav_path, target_sr=config.SAMPLE_RATE)
    stft_matrix = compute_stft(audio)
    v_matrix = log_compression(stft_matrix)
    h_matrix = solve_nnls_batch(v_matrix, w_matrix)
    h_event_sum = compute_h_event_sum(h_matrix)

    total_frames = len(h_event_sum)
    frame_times = frame_to_time(np.arange(total_frames), hop_length=config.HOP_LENGTH, sr=sr)

    np.savez_compressed(
        str(feat_file),
        h_event_sum=h_event_sum.astype(np.float32),
        frame_times=frame_times.astype(np.float32)
    )
    return h_event_sum, frame_times


def run_annotation():
    print("=" * 75)
    print("GIAI ĐOẠN 4: GÁN NHÃN THỦ CÔNG FRAME-LEVEL (AUDIO-VISUAL ANNOTATION)")
    print("=" * 75)

    # 1. Quét danh sách file âm thanh tập Train
    train_wav_files = sorted(list(config.TRAIN_AUDIO_DIR.glob(config.PATTERN_CRNN_TRAIN)))
    if not train_wav_files:
        train_wav_files = sorted(list(config.DATA_DIR.glob(f"**/{config.PATTERN_CRNN_TRAIN}")))

    if not train_wav_files:
        print(f"[!] Không tìm thấy file nào khớp '{config.PATTERN_CRNN_TRAIN}' trong {config.TRAIN_AUDIO_DIR}")
        return

    # 2. Đọc Ground Truth Count từ data/train/label.txt
    train_label_file = config.TRAIN_AUDIO_DIR / "label.txt"
    gt_counts = parse_label_txt(train_label_file)
    print(f"[*] Đã nạp {len(gt_counts)} mốc nhãn count từ: {train_label_file.name}")

    # 3. Kiểm tra và nạp ma trận từ điển W_standard
    if not config.W_STANDARD_PATH.is_file():
        print(f"[!] Lỗi: Không tìm thấy {config.W_STANDARD_PATH}. Hãy chạy scripts/train_nmf.py trước.")
        return
    w_matrix = load_dictionary(config.W_STANDARD_PATH)

    # 4. Nạp P*, D* tối ưu từ Grid Search
    opt_p, opt_d = load_peak_parameters()
    print(f"[*] Nạp cấu hình Peak Detection tối ưu: P* = {opt_p}, D* = {opt_d} frames")

    print(f"[*] Bắt đầu gán nhãn cho {len(train_wav_files)} files...")

    # 5. Lặp qua từng file
    for idx, wav_path in enumerate(train_wav_files, 1):
        output_npz = config.TRAIN_ANNOTATION_DIR / f"{wav_path.stem}_labels.npz"
        print(f"\n[{idx:02d}/{len(train_wav_files):02d}] Mở file: {wav_path.name}")

        # Nạp audio waveform cho engine phát âm thanh
        audio_samples, sr = load_audio(wav_path, target_sr=config.SAMPLE_RATE)

        # Lấy mảng H_event_sum
        h_event_sum, frame_times = get_or_compute_features(wav_path, w_matrix)
        total_frames = len(h_event_sum)

        # Mốc Ground Truth từ label.txt
        gt_count = gt_counts.get(wav_path.name, None)

        # Quyết định đỉnh hiển thị ban đầu
        if output_npz.is_file():
            print(f"    [i] Nạp lại nhãn đã gán trước đó từ: {output_npz.name}")
            annot_data = load_annotation(output_npz)
            initial_peaks = np.where(annot_data["binary_labels"] == 1)[0]
        else:
            initial_peaks, _ = detect_reference_peaks(
                h_event_sum,
                prominence=opt_p,
                distance=opt_d
            )
            print(f"    [*] Khởi tạo {len(initial_peaks)} đỉnh tham chiếu từ P*={opt_p}, D*={opt_d}")

        # Khởi chạy GUI tương tác
        annotator = InteractiveAnnotator(
            audio_filename=wav_path.name,
            h_event_sum=h_event_sum,
            reference_peaks=initial_peaks,
            prominence_threshold=opt_p,
            ground_truth_count=gt_count,
            audio_samples=audio_samples,
            sample_rate=sr
        )
        confirmed_peaks = annotator.show()

        # Tạo nhãn nhị phân
        binary_labels = np.zeros(total_frames, dtype=np.uint8)
        binary_labels[confirmed_peaks] = 1

        # Lưu file .npz
        save_annotation(
            output_path=output_npz,
            audio_filename=wav_path.name,
            h_event_sum=h_event_sum,
            binary_labels=binary_labels,
            frame_times=frame_times,
            metadata={
                "true_count": len(confirmed_peaks),
                "gt_label_count": gt_count,
                "opt_prominence": opt_p,
                "opt_distance": opt_d
            }
        )
        print(f"    [✓] Đã lưu nhãn thành công: {output_npz.name} (Số phôi xác nhận: {len(confirmed_peaks)})")

    print("\n" + "=" * 75)
    print("[✓] ĐÃ HOÀN TẤT TOÀN BỘ QUÁ TRÌNH GÁN NHÃN CHO TẬP TRAIN!")
    print("=" * 75)


if __name__ == "__main__":
    run_annotation()