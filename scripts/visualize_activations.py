"""
scripts/visualize_activations.py

Hiển thị ma trận kích hoạt H (48 x T) cho tất cả các file Train_*.wav và Bowl_*.wav.
Pipeline: load_audio -> compute_stft -> log_compression -> NNLS.
Phân định 3 nhóm thành phần rõ ràng:
- Event: 1 -> 24
- Bowl Noise: 25 -> 36
- Environmental Noise: 37 -> 48
"""

import sys
from pathlib import Path

# Đảm bảo đường dẫn gốc được nhận diện chính xác
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls

import config
from src.audio.loader import load_audio
from src.preprocessing.stft import compute_stft
from src.preprocessing.log_compression import log_compression
from src.nmf.dictionary import load_dictionary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize NMF Activation Matrix H(k, t) for Train and Bowl audio files."
    )
    parser.add_argument(
        "--group",
        type=str,
        choices=["train", "bowl", "all"],
        default="all",
        help="Nhóm file cần xuất ảnh: 'train', 'bowl', hoặc 'all' (mặc định: 'all')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(config.FIGURES_DIR / "activations"),
        help="Thư mục lưu ảnh kết quả (mặc định: outputs/figures/activations/)",
    )
    return parser.parse_args()


def solve_nnls_matrix(W, V):
    """
    Giải bài toán Non-Negative Least Squares cho từng frame:
    min ||V[:, t] - W * H[:, t]||_2 với H[:, t] >= 0
    W: (n_bins, n_components) = (1025, 48)
    V: (n_bins, n_frames)
    Trả về H: (n_components, n_frames) = (48, n_frames)
    """
    n_components = W.shape[1]
    n_frames = V.shape[1]
    H = np.zeros((n_components, n_frames), dtype=np.float32)

    for t in range(n_frames):
        h_t, _ = nnls(W, V[:, t])
        H[:, t] = h_t

    return H


def plot_activation_spectrogram(H, filename, output_path, sr=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH):
    """
    Vẽ phổ ma trận kích hoạt H(k, t) với đường phân vùng 3 nhóm rõ rệt.
    Trục X: Thời gian (s).
    Trục Y: NMF Components (1 -> 48).
    """
    n_components, n_frames = H.shape
    duration = n_frames * hop_length / sr

    fig, ax = plt.subplots(figsize=(15, 8), dpi=300)

    # Hiển thị heatmap: component 1 ở trên cùng, component 48 ở dưới cùng
    im = ax.imshow(
        H,
        aspect="auto",
        origin="upper",
        cmap="inferno",
        extent=[0, duration, 48.5, 0.5],
        interpolation="nearest"
    )

    # Đường phân cách ngang giữa 3 nhóm
    ax.axhline(24.5, color="#00FFFF", linestyle="-", linewidth=2.0, alpha=0.95)
    ax.axhline(36.5, color="#39FF14", linestyle="-", linewidth=2.0, alpha=0.95)

    # Nhãn vùng ở trục Y phụ bên phải (Secondary Y-axis)
    ax_labels = ax.twinx()
    ax_labels.set_ylim(ax.get_ylim())
    ax_labels.set_yticks([12.5, 30.5, 42.5])
    ax_labels.set_yticklabels([
        "EVENT / Workpiece Impact\n(Components 1 - 24)",
        "BOWL NOISE\n(Components 25 - 36)",
        "ENVIRONMENT NOISE\n(Components 37 - 48)"
    ], fontsize=11, fontweight="bold")
    ax_labels.tick_params(length=0)

    # Cấu hình trục Y bên trái và trục X
    ax.set_ylabel("NMF Component Index", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontsize=12, fontweight="bold")
    ax.set_yticks([1, 6, 12, 18, 24, 25, 30, 36, 37, 42, 48])
    ax.set_title(f"NMF Activation Spectrogram $H(k, t)$ — {filename}", fontsize=14, pad=12, fontweight="bold")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, pad=0.14, fraction=0.046)
    cbar.set_label("Activation Magnitude $H_{k}(t)$", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {output_path}")


def process_file(audio_path, W_standard, output_dir):
    """
    Chạy pipeline: Audio -> STFT -> Log Compression -> NNLS -> Vẽ hình
    """
    # 1. Đọc audio
    signal, sr = load_audio(audio_path, target_sr=config.SAMPLE_RATE)

    # 2. Tính STFT magnitude spectrum
    mag_spec = compute_stft(
        signal,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        window=config.WINDOW
    )

    # 3. Log compression
    V = log_compression(mag_spec, gamma=config.GAMMA)

    # 4. NNLS với W_standard (1025 x 48) -> H (48, T)
    H = solve_nnls_matrix(W_standard, V)

    # 5. Vẽ và lưu ảnh
    out_filename = f"{audio_path.stem}_H.png"
    out_path = Path(output_dir) / out_filename
    plot_activation_spectrogram(H, audio_path.name, out_path, sr=sr, hop_length=config.HOP_LENGTH)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    w_path = config.W_STANDARD_PATH
    if not w_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file {w_path}. Hãy chạy `python scripts/train_nmf.py` trước!")

    W_standard = load_dictionary(w_path)
    print(f" Loaded W_standard shape: {W_standard.shape}")

    train_dir = config.DATA_TRAIN_DIR
    files_to_process = []

    if args.group in ["train", "all"]:
        files_to_process.extend(sorted(list(train_dir.glob("Train_*.wav"))))

    if args.group in ["bowl", "all"]:
        files_to_process.extend(sorted(list(train_dir.glob("Bowl_*.wav"))))

    if not files_to_process:
        print(f"Không tìm thấy file Train_*.wav hoặc Bowl_*.wav nào trong {train_dir}")
        return

    print(f" Bắt đầu xử lý {len(files_to_process)} file âm thanh...")
    for idx, fpath in enumerate(files_to_process, 1):
        print(f"[{idx}/{len(files_to_process)}] Đang xử lý: {fpath.name}")
        process_file(fpath, W_standard, out_dir)

    print(f"\n Hoàn tất! Tất cả hình ảnh phổ activation đã được lưu tại:\n {out_dir.resolve()}")


if __name__ == "__main__":
    main()