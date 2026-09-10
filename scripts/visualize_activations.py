"""
scripts/visualize_activations.py

Hiển thị ma trận kích hoạt H (48 x T) cho tất cả các file Train_*.wav và Bowl_*.wav.
Pipeline: load_audio -> compute_stft -> log_compression -> solve_nnls_batch.
Phân định 3 nhóm thành phần rõ ràng:
- Event: 1 -> 24
- Bowl Noise: 25 -> 36
- Environmental Noise: 37 -> 48
"""

import sys
from pathlib import Path

# Tự động nhận diện thư mục gốc của project
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import config
from src.audio.loader import load_audio
from src.preprocessing.stft import compute_stft
from src.preprocessing.log_compression import log_compression
from src.nmf.dictionary import load_dictionary
from src.nmf.nnls import solve_nnls_batch


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
        default=str(config.FIGURES_DIR / "activations") if hasattr(config, "FIGURES_DIR") else str(PROJECT_ROOT / "outputs" / "figures" / "activations"),
        help="Thư mục lưu ảnh kết quả (mặc định: outputs/figures/activations/)",
    )
    return parser.parse_args()


def plot_activation_spectrogram(H, filename, output_path, total_duration):
    """
    Vẽ phổ ma trận kích hoạt H(k, t) bằng pcolormesh kết hợp LogNorm:
    - Trục thời gian trải đều đúng tổng thời lượng thực tế của file âm thanh (total_duration).
    - LogNorm giúp phân bổ dải động rộng: H=0 là nền tối, kích hoạt yếu có màu tím/cam, kích hoạt mạnh rực sáng.
    - Mỗi ô component dày chuẩn xác đúng 1/48 chiều cao phổ.
    """
    n_components, n_frames = H.shape

    # 1. Trục thời gian (giây) và trục Component (Y)
    time_edges = np.linspace(0.0, total_duration, n_frames + 1)
    comp_edges = np.arange(0.5, n_components + 1.5, 1.0)

    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)

    # 2. Thiết lập Log Normalization trên tập giá trị dương H > 0
    positive_values = H[H > 0]

    if positive_values.size > 0:
        v_min = np.percentile(positive_values, 5)
        v_max = np.percentile(positive_values, 99.5)

        if v_min >= v_max:
            v_min = positive_values.min()
            v_max = positive_values.max()
        if v_min <= 0:
            v_min = 1e-6
    else:
        v_min = 1e-6
        v_max = 1.0

    # Cấu hình cmap để các giá trị <= 0 hoặc dưới v_min ăn khớp màu đen nền tối
    cmap = plt.get_cmap("magma").copy()
    cmap.set_under(cmap(0.0))

    # 3. Vẽ bằng pcolormesh với norm=LogNorm
    mesh = ax.pcolormesh(
        time_edges,
        comp_edges,
        H,
        cmap=cmap,
        norm=LogNorm(vmin=v_min, vmax=v_max),
        shading="flat",
        edgecolors="none"
    )

    # Đảo ngược trục Y: Component 1 ở trên đỉnh, Component 48 ở đáy
    ax.set_ylim(48.5, 0.5)
    ax.set_xlim(0.0, total_duration)

    # 4. Đường phân cách ngang giữa 3 nhóm
    ax.axhline(24.5, color="#00E5FF", linestyle="--", linewidth=1.8, alpha=0.95)
    ax.axhline(36.5, color="#76FF03", linestyle="--", linewidth=1.8, alpha=0.95)

    # 5. Cấu hình nhãn vùng ở trục Y phụ bên phải (Secondary Y-axis)
    ax_labels = ax.twinx()
    ax_labels.set_ylim(ax.get_ylim())
    ax_labels.set_yticks([12.5, 30.5, 42.5])
    ax_labels.set_yticklabels([
        "EVENT / Workpiece Impact\n(Components 1 - 24)",
        "BOWL NOISE\n(Components 25 - 36)",
        "ENVIRONMENT NOISE\n(Components 37 - 48)"
    ], fontsize=11, fontweight="bold")
    ax_labels.tick_params(length=0)

    # 6. Cấu hình trục chính
    ax.set_ylabel("NMF Component Index", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontsize=12, fontweight="bold")
    ax.set_yticks([1, 6, 12, 18, 24, 25, 30, 36, 37, 42, 48])
    ax.tick_params(axis="both", labelsize=10)
    ax.set_title(f"NMF Activation Spectrogram $H(k, t)$ — {filename}", fontsize=14, pad=12, fontweight="bold")

    # Colorbar với log format
    cbar = fig.colorbar(mesh, ax=ax, pad=0.14, fraction=0.046)
    cbar.set_label("Activation Magnitude $H_{k}(t)$ (Log Scale)", fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {output_path}")


def process_file(audio_path, W_standard, output_dir):
    """
    Chạy pipeline đồng bộ: Audio -> STFT -> Log Compression -> solve_nnls_batch -> Vẽ hình
    """
    signal, sr = load_audio(audio_path, target_sr=config.SAMPLE_RATE)
    total_duration = len(signal) / sr

    mag_spec = compute_stft(
        signal,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        window=config.WINDOW_FUNCTION
    )

    V = log_compression(
        mag_spec,
        spec_type=config.SPECTROGRAM_TYPE,
        epsilon=config.LOG_EPSILON
    )

    H = solve_nnls_batch(W_standard, V)

    out_filename = f"{audio_path.stem}_H.png"
    out_path = Path(output_dir) / out_filename
    plot_activation_spectrogram(H, audio_path.name, out_path, total_duration=total_duration)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    w_path = config.W_STANDARD_PATH
    if not w_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file {w_path}. Hãy chạy `python scripts/train_nmf.py` trước!")

    W_standard = load_dictionary(w_path)
    print(f"Loaded W_standard shape: {W_standard.shape}")

    if hasattr(config, "TRAIN_DATA_DIR"):
        train_dir = Path(config.TRAIN_DATA_DIR)
    elif hasattr(config, "DATA_DIR"):
        train_dir = Path(config.DATA_DIR) / "train"
    else:
        train_dir = PROJECT_ROOT / "data" / "train"

    files_to_process = []

    train_pattern = getattr(config, "PATTERN_CRNN_TRAIN", "Train_*.wav")
    bowl_pattern = getattr(config, "PATTERN_TEST_BOWL", "Bowl_*.wav")

    if args.group in ["train", "all"]:
        files_to_process.extend(sorted(list(train_dir.glob(train_pattern))))

    if args.group in ["bowl", "all"]:
        files_to_process.extend(sorted(list(train_dir.glob(bowl_pattern))))

    if not files_to_process:
        print(f"Không tìm thấy file nào khớp với '{train_pattern}' hoặc '{bowl_pattern}' trong {train_dir}")
        return

    print(f"Bắt đầu xử lý {len(files_to_process)} file âm thanh...")
    for idx, fpath in enumerate(files_to_process, 1):
        print(f"[{idx}/{len(files_to_process)}] Đang xử lý: {fpath.name}")
        process_file(fpath, W_standard, out_dir)

    print(f"\nHoàn tất! Tất cả hình ảnh phổ activation đã được lưu tại:\n{out_dir.resolve()}")


if __name__ == "__main__":
    main()