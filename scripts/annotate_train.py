"""
===============================================================================
Script: Annotate Training Files (scripts/annotate_train.py)
===============================================================================

Iterates through data/train/*.wav, runs NMF feature extraction, launches
the interactive annotation tool, and saves validated ground truth into
data/annotations/train/*_labels.npz.
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
from src.annotation.peak_reference import detect_reference_peaks
from src.annotation.manual_label import InteractiveAnnotator
from src.annotation.annotation_io import save_annotation


def annotate_dataset():
    # Chỉ quét đúng các file Train_*.wav cho huấn luyện CRNN
    train_wav_files = sorted(list(config.TRAIN_AUDIO_DIR.glob(config.PATTERN_CRNN_TRAIN)))
    if not train_wav_files:
        # Dự phòng tìm ở thư mục data/ chung nếu chưa phân chia subfolder
        train_wav_files = sorted(list(config.DATA_DIR.glob(f"**/{config.PATTERN_CRNN_TRAIN}")))
    
    if not train_wav_files:
        print(f"[!] Không tìm thấy file nào khớp mẫu '{config.PATTERN_CRNN_TRAIN}'.")
        return

    print(f"[*] Tìm thấy {len(train_wav_files)} file trong tập train. Nạp W_standard...")
    w_matrix = load_dictionary(config.W_STANDARD_PATH)

    for idx, wav_path in enumerate(train_wav_files, 1):
        output_npz = config.TRAIN_ANNOTATION_DIR / f"{wav_path.stem}_labels.npz"

        print(f"\n[{idx}/{len(train_wav_files)}] Đang xử lý: {wav_path.name}")
        if output_npz.exists():
            print(f"    [i] Đã tồn tại nhãn: {output_npz.name}. Đang nạp để chỉnh sửa lại...")

        # 1. Pipeline Feature Extraction
        audio, sr = load_audio(wav_path, target_sr=config.SAMPLE_RATE)
        stft_matrix = compute_stft(audio)
        v_matrix = log_compression(stft_matrix)
        h_matrix = solve_nnls_batch(v_matrix, w_matrix)
        h_event_sum = compute_h_event_sum(h_matrix)

        total_frames = len(h_event_sum)
        frame_times = frame_to_time(np.arange(total_frames), hop_length=config.HOP_LENGTH, sr=sr)

        # 2. Sinh mốc đỉnh gợi ý ban đầu
        ref_peaks, _ = detect_reference_peaks(h_event_sum)

        # 3. Mở Interactive GUI gán nhãn
        annotator = InteractiveAnnotator(
            audio_filename=wav_path.name,
            h_event_sum=h_event_sum,
            reference_peaks=ref_peaks
        )
        confirmed_peak_frames = annotator.show()

        # 4. Sinh vector nhãn nhị phân Y (0 hoặc 1 cho từng frame)
        binary_labels = np.zeros(total_frames, dtype=np.uint8)
        binary_labels[confirmed_peak_frames] = 1

        # 5. Lưu kết quả Annotation vào file .npz
        save_annotation(
            output_path=output_npz,
            audio_filename=wav_path.name,
            h_event_sum=h_event_sum,
            binary_labels=binary_labels,
            frame_times=frame_times,
            metadata={
                "true_count": len(confirmed_peak_frames),
                "sampling_rate": sr,
                "n_fft": config.N_FFT,
                "hop_length": config.HOP_LENGTH
            }
        )
        print(f"    [✓] Đã lưu nhãn thành công: {output_npz.name} (Tổng số phôi: {len(confirmed_peak_frames)})")

    print("\n[✓] Hoàn tất gán nhãn cho toàn bộ tập Train!")


if __name__ == "__main__":
    annotate_dataset()