"""
===============================================================================
Script: Train Hybrid CRNN Model (scripts/train_crnn.py)
===============================================================================

Loads the packed dataset from data/datasets/crnn_train.npz,
splits into train/val subsets using config.CRNN_VALIDATION_SPLIT,
and trains the HybridCRNN model saving weights to config.CRNN_MODEL_PATH.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

import config
from src.crnn.dataset import HybridCRNNDataset
from src.crnn.model import HybridCRNN
from src.crnn.trainer import CRNNTrainer


def main():
    print("=" * 70)
    print("GIAI ĐOẠN 6: HUẤN LUYỆN MÔ HÌNH HYBRID CRNN")
    print("=" * 70)

    dataset_file = config.CRNN_DATASET_DIR / "crnn_train.npz"
    if not dataset_file.is_file():
        print(f"[!] Không tìm thấy tập dataset: {dataset_file}")
        print("    Vui lòng chạy 'python scripts/build_crnn_dataset.py' trước.")
        return

    print(f"[*] Nạp dữ liệu huấn luyện từ: {dataset_file.name}")
    data = np.load(str(dataset_file))
    x_mel = data["X_mel_train"]
    x_he = data["X_he_train"]
    y = data["Y_train"]
    pos_weight = float(data.get("pos_weight", 1.0))

    full_dataset = HybridCRNNDataset(x_mel, x_he, y)
    val_size = max(1, int(config.CRNN_VALIDATION_SPLIT * len(full_dataset)))
    train_size = len(full_dataset) - val_size

    torch.manual_seed(config.CRNN_RANDOM_SEED)
    generator = torch.Generator().manual_seed(config.CRNN_RANDOM_SEED)
    train_set, val_set = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=config.CRNN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.CRNN_BATCH_SIZE, shuffle=False)

    print(f"[*] Khởi tạo Hybrid CRNN (Train samples: {train_size}, Val samples: {val_size})...")
    model = HybridCRNN()
    trainer = CRNNTrainer(model=model, pos_weight=pos_weight)

    print(f"[*] Bắt đầu huấn luyện {config.CRNN_EPOCHS} epochs...")
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.CRNN_EPOCHS,
        save_path=config.CRNN_MODEL_PATH
    )

    print("\n" + "=" * 70)
    print(f"[✓] Huấn luyện thành công! Trọng số đã lưu tại: {config.CRNN_MODEL_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()