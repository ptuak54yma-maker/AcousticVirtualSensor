"""
===============================================================================
CRNN Model Trainer Module
===============================================================================

Handles the training and validation loops for the Hybrid CRNN model.
Uses BCEWithLogitsLoss with positive class weighting to counteract class imbalance.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import config
from src.crnn.dataset import HybridCRNNDataset
from src.crnn.model import HybridCRNN


class CRNNTrainer:
    """
    Orchestrates the training process of the HybridCRNN model.
    """

    def __init__(
        self,
        model: HybridCRNN,
        learning_rate: float = config.CRNN_LEARNING_RATE,
        weight_decay: float = config.CRNN_WEIGHT_DECAY,
        pos_weight: Optional[float] = None,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # Loss function with positive class weighting
        if pos_weight is not None and config.USE_POSITIVE_CLASS_WEIGHT:
            weight_tensor = torch.tensor([pos_weight], device=self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.train_history = {"train_loss": [], "val_loss": []}

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0

        for x_mel, x_he, y in dataloader:
            x_mel = x_mel.to(self.device)
            x_he = x_he.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x_mel, x_he)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(y)

        return total_loss / len(dataloader.dataset)

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        """Run one validation epoch."""
        self.model.eval()
        total_loss = 0.0

        for x_mel, x_he, y in dataloader:
            x_mel = x_mel.to(self.device)
            x_he = x_he.to(self.device)
            y = y.to(self.device)

            logits = self.model(x_mel, x_he)
            loss = self.criterion(logits, y)
            total_loss += loss.item() * len(y)

        return total_loss / len(dataloader.dataset)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = config.CRNN_EPOCHS,
        save_path: Optional[Path] = config.CRNN_MODEL_PATH
    ) -> Dict[str, list]:
        """Execute the full training loop and save the best model weights."""
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            self.train_history["train_loss"].append(train_loss)

            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.train_history["val_loss"].append(val_loss)
                print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

                if val_loss < best_val_loss and save_path is not None:
                    best_val_loss = val_loss
                    self.save_checkpoint(save_path)
            else:
                print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f}")
                if save_path is not None:
                    self.save_checkpoint(save_path)

        return self.train_history

    def save_checkpoint(self, path: Path):
        """Save model weights to disk."""
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(path))