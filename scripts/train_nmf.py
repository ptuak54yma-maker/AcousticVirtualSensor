"""
===============================================================================
Script: Train NMF Dictionary (scripts/train_nmf.py)
===============================================================================

Trains the fixed dictionary W_standard (1025 x 48) from isolated sound recordings:
- W_event: 24 components (Part impact sounds)
- W_bowl: 12 components (Bowl drive vibrations)
- W_env: 12 components (Ambient noise)
Algorithm: Coordinate Descent with Frobenius cost and L2 penalty on H.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.decomposition import NMF
import config
from src.audio.loader import load_audio
from src.preprocessing.stft import compute_stft
from src.preprocessing.log_compression import log_compression


def train_single_dictionary(
    audio_path: Path,
    n_components: int,
    alpha_h: float,
    l1_ratio: float = config.NMF_L1_RATIO,
    max_iter: int = config.NMF_MAX_ITER
) -> np.ndarray:
    """Train an individual source basis matrix W."""
    audio, _ = load_audio(audio_path, target_sr=config.SAMPLE_RATE)
    stft_matrix = compute_stft(audio)
    v_matrix = log_compression(stft_matrix)

    model = NMF(
        n_components=n_components,
        init=config.NMF_INIT,
        solver=config.NMF_SOLVER,
        beta_loss=config.NMF_BETA_LOSS,
        alpha_H=alpha_h,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        random_state=config.NMF_RANDOM_SEED
    )
    w = model.fit_transform(v_matrix)
    return w.astype(np.float32)


def main():
    print("[*] Bắt đầu quy trình huấn luyện Fixed Dictionary W_standard...")
    
    event_wav = config.DATA_DIR / "train_sources" / "event_source.wav"
    bowl_wav = config.DATA_DIR / "train_sources" / "bowl_source.wav"
    env_wav = config.DATA_DIR / "train_sources" / "env_source.wav"

    if not (event_wav.is_file() and bowl_wav.is_file() and env_wav.is_file()):
        print(f"[!] Chưa tìm thấy đủ 3 file nguồn âm thanh trong thư mục data/train_sources/.")
        print(f"    Yêu cầu: {event_wav.name}, {bowl_wav.name}, {env_wav.name}")
        return

    print(f"  -> Huấn luyện W_event ({config.EVENT_COMPONENTS} components, alpha_H={config.NMF_EVENT_ALPHA_H})...")
    w_event = train_single_dictionary(event_wav, config.EVENT_COMPONENTS, alpha_h=config.NMF_EVENT_ALPHA_H)

    print(f"  -> Huấn luyện W_bowl ({config.BOWL_COMPONENTS} components, alpha_H={config.NMF_BOWL_ALPHA_H})...")
    w_bowl = train_single_dictionary(bowl_wav, config.BOWL_COMPONENTS, alpha_h=config.NMF_BOWL_ALPHA_H)

    print(f"  -> Huấn luyện W_env ({config.ENV_COMPONENTS} components, alpha_H={config.NMF_ENV_ALPHA_H})...")
    w_env = train_single_dictionary(env_wav, config.ENV_COMPONENTS, alpha_h=config.NMF_ENV_ALPHA_H)

    # Ghép 3 từ điển thành W_standard (1025 x 48)
    w_standard = np.concatenate([w_event, w_bowl, w_env], axis=1)
    
    output_w_path = config.W_STANDARD_PATH
    output_w_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_w_path), w_standard)

    print("\n" + "=" * 60)
    print(f"[✓] HUẤN LUYỆN THÀNH CÔNG W_standard!")
    print(f"    - Kích thước ma trận : {w_standard.shape} (Freq Bins x Components)")
    print(f"    - Lưu trữ tại        : {output_w_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()