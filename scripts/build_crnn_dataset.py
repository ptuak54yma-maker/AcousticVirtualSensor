"""
===============================================================================
Script: Build Multimodal CRNN Dataset (scripts/build_crnn_dataset.py)
===============================================================================

Reads WAV audio and annotated labels, computes Log-Mel Spectrogram and NMF 
H_event_sum, applies Gaussian label smoothing, extracts temporal context 
windows, and saves everything into data/datasets/crnn_train.npz.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import librosa
import config

from src.audio.loader import load_audio
from src.preprocessing.stft import compute_stft
from src.annotation.annotation_io import load_annotation, create_gaussian_label
from src.crnn.dataset import extract_hybrid_windows


def compute_log_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """
    Compute Log-Mel Spectrogram with project standard parameters.
    """
    stft_matrix = compute_stft(audio)
    magnitude_spectrogram = np.abs(stft_matrix)

    # Mel filterbank projection
    mel_basis = librosa.filters.mel(
        sr=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        n_mels=config.N_MELS,
        fmin=config.F_MIN,
        fmax=config.F_MAX
    )
    mel_spectrogram = np.dot(mel_basis, magnitude_spectrogram)
    log_mel = np.log1p(mel_spectrogram)
    return np.ascontiguousarray(log_mel, dtype=np.float32)


def build_train_dataset():
    annotation_files = sorted(list(config.TRAIN_ANNOTATION_DIR.glob("*_labels.npz")))
    if not annotation_files:
        print(f"[!] Không tìm thấy file gán nhãn nào trong: {config.TRAIN_ANNOTATION_DIR}")
        print("    Vui lòng chạy 'python scripts/annotate_train.py' trước để tạo nhãn ground truth.")
        return

    print(f"[*] Bắt đầu đóng gói Multimodal CRNN dataset từ {len(annotation_files)} file...")

    all_x_mel = []
    all_x_he = []
    all_y = []
    total_events_count = 0

    for file_idx, npz_path in enumerate(annotation_files, 1):
        data = load_annotation(npz_path)
        audio_name = str(data.get("audio_filename", f"{npz_path.stem.replace('_labels', '')}.wav"))
        wav_path = config.TRAIN_AUDIO_DIR / audio_name

        if not wav_path.is_file():
            print(f"    [!] Cảnh báo: Không tìm thấy file WAV gốc '{wav_path.name}'. Bỏ qua file này.")
            continue

        # 1. Nạp âm thanh và tính Log-Mel Spectrogram (128, T)
        audio, _ = load_audio(wav_path, target_sr=config.SAMPLE_RATE)
        log_mel = compute_log_mel_spectrogram(audio)

        # 2. Nạp H_event_sum (1D) và nhãn đã gán
        h_event_sum = data["h_event_sum"]
        binary_labels = data["binary_labels"]
        total_frames = len(h_event_sum)

        # Cân chỉnh độ dài nếu có sai lệch nhỏ ở frame cuối
        min_len = min(log_mel.shape[1], total_frames, len(binary_labels))
        log_mel = log_mel[:, :min_len]
        h_event_sum = h_event_sum[:min_len]
        binary_labels = binary_labels[:min_len]
        total_frames = min_len

        peak_frames = np.where(binary_labels == 1)[0]
        total_events_count += len(peak_frames)

        # 3. Làm mịn nhãn Gaussian
        if config.ANNOTATION_SMOOTHING_ENABLED:
            target_labels = create_gaussian_label(
                peak_frames=peak_frames,
                total_frames=total_frames,
                sigma=config.ANNOTATION_GAUSSIAN_SIGMA
            )
        else:
            target_labels = binary_labels.astype(np.float32)

        # 4. Cắt cửa sổ trượt đồng thời cả Mel và H_event_sum
        X_mel_file, X_he_file, Y_file = extract_hybrid_windows(
            log_mel_seq=log_mel,
            h_event_sum_seq=h_event_sum,
            label_seq=target_labels,
            window_length=config.SEQUENCE_LENGTH,
            hop_size=config.SEQUENCE_HOP
        )

        if len(X_mel_file) > 0:
            all_x_mel.append(X_mel_file)
            all_x_he.append(X_he_file)
            all_y.append(Y_file)

        print(f"  [{file_idx}/{len(annotation_files)}] {audio_name}: "
              f"{total_frames} frames -> {len(X_mel_file)} sequences (Events: {len(peak_frames)})")

    if not all_x_mel:
        print("[!] Không tạo được sample nào. Kiểm tra lại dữ liệu đầu vào.")
        return

    # Ghép toàn bộ samples
    X_mel_train = np.concatenate(all_x_mel, axis=0)  # Shape: (N, 1, 128, L)
    X_he_train = np.concatenate(all_x_he, axis=0)    # Shape: (N, 1, L)
    Y_train = np.concatenate(all_y, axis=0)          # Shape: (N, L)

    # Tính positive weight phục vụ hàm mất mát
    num_pos = np.sum(Y_train > 0.1)
    num_neg = Y_train.size - num_pos
    pos_weight = float(num_neg / max(1, num_pos))

    output_dataset_file = config.CRNN_DATASET_DIR / "crnn_train.npz"
    np.savez_compressed(
        str(output_dataset_file),
        X_mel_train=X_mel_train,
        X_he_train=X_he_train,
        Y_train=Y_train,
        pos_weight=pos_weight,
        sequence_length=config.SEQUENCE_LENGTH,
        sequence_hop=config.SEQUENCE_HOP,
        total_samples=len(Y_train),
        total_events=total_events_count
    )

    print("\n" + "=" * 60)
    print(f"[✓] ĐÃ ĐÓNG GÓI THÀNH CÔNG MULTIMODAL CRNN DATASET!")
    print(f"    - File lưu trữ: {output_dataset_file}")
    print(f"    - Shape X_mel_train: {X_mel_train.shape} (N, 1, N_mels, L)")
    print(f"    - Shape X_he_train:  {X_he_train.shape} (N, 1, L)")
    print(f"    - Shape Y_train:     {Y_train.shape} (N, L)")
    print(f"    - Tổng số mẫu (sequences): {len(Y_train)}")
    print(f"    - Pos Weight: {pos_weight:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    build_train_dataset()