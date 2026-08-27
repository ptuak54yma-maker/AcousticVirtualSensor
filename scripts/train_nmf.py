"""
===============================================================================
Script: Train NMF Dictionary W_standard (scripts/train_nmf.py)
===============================================================================

Automatically aggregates audio files by prefix:
- phoi_*.wav      -> W_event (24 components, Coordinate Descent + L2 Regularization)
- noisebowl_*.wav -> W_bowl  (12 components)
- noiseevir_*.wav -> W_env   (12 components)

Resulting Matrix: W_standard of shape (1025, 48).
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.decomposition import NMF
import config
from src.audio.loader import load_audio
from src.preprocessing.stft import compute_stft
from src.preprocessing.log_compression import log_compression


def extract_aggregated_spectrum(file_list: list) -> np.ndarray:
    """
    Load multiple WAV files, compute Log-Magnitude Spectrogram for each,
    and concatenate them horizontally along the time axis.
    """
    v_blocks = []
    for fpath in sorted(file_list):
        audio, sr = load_audio(fpath, target_sr=config.SAMPLE_RATE)
        stft_matrix = compute_stft(audio)
        v = log_compression(stft_matrix)  # V = log(1 + |X|)
        v_blocks.append(v)

    # Concatenate along time axis (Frequency x Total_Frames)
    return np.concatenate(v_blocks, axis=1)


def train_single_source(
    file_list: list,
    n_components: int,
    alpha_h: float = 0.0,
    source_name: str = "Source"
) -> np.ndarray:
    """Train basis matrix W for a specific source group."""
    print(f"[*] Đang nạp {len(file_list)} files cho nhóm [{source_name}]...")
    for f in file_list:
        print(f"    - {f.name}")

    V_combined = extract_aggregated_spectrum(file_list)
    print(f"    -> Tổng số khung phổ V: {V_combined.shape} (1025 bins x {V_combined.shape[1]} frames)")

    # NMF Solver: Coordinate Descent with Frobenius cost
    model = NMF(
        n_components=n_components,
        init="random",
        solver="cd",
        beta_loss="frobenius",
        alpha_H=alpha_h,
        l1_ratio=0.0,  # Pure L2 penalty on H
        max_iter=config.NNLS_MAX_ITER,
        random_state=42
    )

    W = model.fit_transform(V_combined)
    return W.astype(np.float32)


def main():
    print("=" * 65)
    print("QUY TRÌNH TỰ ĐỘNG HUẤN LUYỆN FIXED DICTIONARY W_STANDARD")
    print("=" * 65)

    # Quét toàn bộ file theo tiền tố đã định nghĩa
    audio_dir = config.AUDIO_DIR if hasattr(config, "AUDIO_DIR") and config.AUDIO_DIR.exists() else config.TRAIN_AUDIO_DIR
    
    event_files = sorted(list(audio_dir.glob(config.PATTERN_NMF_EVENT)))
    bowl_files  = sorted(list(audio_dir.glob(config.PATTERN_NMF_BOWL)))
    env_files   = sorted(list(audio_dir.glob(config.PATTERN_NMF_ENV)))

    if not event_files:
        raise FileNotFoundError(f"Không tìm thấy file nào khớp mẫu '{config.PATTERN_NMF_EVENT}' trong {audio_dir}")
    if not bowl_files:
        raise FileNotFoundError(f"Không tìm thấy file nào khớp mẫu '{config.PATTERN_NMF_BOWL}' trong {audio_dir}")
    if not env_files:
        raise FileNotFoundError(f"Không tìm thấy file nào khớp mẫu '{config.PATTERN_NMF_ENV}' trong {audio_dir}")

    # 1. Huấn luyện W_event (24 components, L2 penalty alpha_H=0.2)
    w_event = train_single_source(event_files, config.EVENT_COMPONENTS, alpha_h=0.2, source_name="EVENT (Phoi)")

    # 2. Huấn luyện W_bowl (12 components)
    w_bowl = train_single_source(bowl_files, config.BOWL_COMPONENTS, alpha_h=0.0, source_name="BOWL (Noise Bowl)")

    # 3. Huấn luyện W_env (12 components)
    w_env = train_single_source(env_files, config.ENV_COMPONENTS, alpha_h=0.0, source_name="ENV (Noise Environment)")

    # 4. Ghép nối thành W_standard (1025 x 48)
    w_standard = np.concatenate([w_event, w_bowl, w_env], axis=1)

    # 5. Lưu ma trận
    output_path = config.W_STANDARD_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), w_standard)

    print("\n" + "=" * 65)
    print(f"[✓] HOÀN TẤT HUẤN LUYỆN W_STANDARD!")
    print(f"    - Kích thước ma trận : {w_standard.shape} (Freq Bins x Components)")
    print(f"    - File lưu trữ       : {output_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()