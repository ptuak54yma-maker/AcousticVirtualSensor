"""
===============================================================================
Hybrid Multimodal CRNN Model (Log-Mel + H_event_sum)
===============================================================================

Defines the neural network architecture for frame-wise acoustic event detection.
Features:
- 2D-CNN with Frequency-only downsampling to compress Log-Mel from (128, L) to (C_mel, L).
- 1D-CNN branch to extract temporal dynamics from H_event_sum.
- Feature Fusion along channel axis preserving exact temporal length L.
- Bidirectional GRU (BiGRU) to model bidirectional temporal context.
- Linear Projection + Sigmoid producing frame-wise peak probabilities.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class FrequencyDownsampler2D(nn.Module):
    """
    2D CNN branch to extract spectrogram texture and compress frequency axis to 1,
    strictly preserving the temporal length L.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 32):
        super().__init__()
        
        # Layer 1: (B, 1, 128, L) -> (B, 16, 64, L)
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        # Layer 2: (B, 16, 64, L) -> (B, 32, 32, L)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        # Layer 3: (B, 32, 32, L) -> (B, 32, 16, L)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        # Layer 4: (B, 32, 16, L) -> (B, out_channels, 8, L)
        self.conv4 = nn.Conv2d(32, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        # Frequency Collapse: Adaptive Pool on Frequency axis only -> (B, out_channels, 1, L)
        self.global_freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.dropout = nn.Dropout2d(config.CRNN_DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B, 1, N_mels, L) = (B, 1, 128, L)
        Output: (B, out_channels, L)
        """
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.global_freq_pool(x)
        x = self.dropout(x)
        
        # Squeeze the collapsed frequency dimension (dim 2)
        x = x.squeeze(2)  # Shape: (B, out_channels, L)
        return x


class EventEnvelopeEncoder1D(nn.Module):
    """
    1D CNN branch to extract local dynamics from the H_event_sum envelope.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 16):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(config.CRNN_DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B, 1, L)
        Output: (B, out_channels, L)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        return x


class HybridCRNN(nn.Module):
    """
    Complete Hybrid CRNN Architecture:
    Mel (2D) + He (1D) -> CNNs -> Concatenation -> BiGRU -> Linear -> Probability Sequence
    """

    def __init__(
        self,
        mel_channels: int = 32,
        he_channels: int = 16,
        rnn_hidden_size: int = config.RNN_HIDDEN_SIZE,
        rnn_layers: int = config.RNN_LAYERS,
        bidirectional: bool = config.RNN_BIDIRECTIONAL,
        dropout: float = config.CRNN_DROPOUT
    ):
        super().__init__()
        
        # 1. Feature Extraction Branches
        self.mel_branch = FrequencyDownsampler2D(in_channels=1, out_channels=mel_channels)
        self.he_branch = EventEnvelopeEncoder1D(in_channels=config.CRNN_INPUT_FEATURES, out_channels=he_channels)

        # 2. Recurrent Temporal Modeling
        # Total feature dimension at each frame t after concatenation
        total_feat_dim = mel_channels + he_channels  # e.g., 32 + 16 = 48
        
        self.gru = nn.GRU(
            input_size=total_feat_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if rnn_layers > 1 else 0.0
        )

        # 3. Output Classification Head
        num_directions = 2 if bidirectional else 1
        gru_out_dim = rnn_hidden_size * num_directions

        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)  # 1 logit per frame
        )

    def forward(self, x_mel: torch.Tensor, x_he: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the hybrid model.

        Parameters
        ----------
        x_mel : torch.Tensor
            Log-Mel Spectrogram batch of shape (B, 1, N_mels, L).
        x_he : torch.Tensor
            H_event_sum envelope batch of shape (B, 1, L).

        Returns
        -------
        logits : torch.Tensor
            Unnormalized log-odds of shape (B, L) for BCEWithLogitsLoss.
        """
        # Step 1: Forward feature branches
        feat_mel = self.mel_branch(x_mel)  # Shape: (B, C_mel, L)
        feat_he = self.he_branch(x_he)     # Shape: (B, C_he, L)

        # Step 2: Concatenate features along channel axis
        fused = torch.cat([feat_mel, feat_he], dim=1)  # Shape: (B, C_mel + C_he, L)

        # Step 3: Permute to (B, L, Features) for GRU batch_first=True
        fused = fused.permute(0, 2, 1)  # Shape: (B, L, 48)

        # Step 4: Temporal Modeling with BiGRU
        gru_out, _ = self.gru(fused)    # Shape: (B, L, hidden_size * num_directions)

        # Step 5: Frame-wise Logits Prediction
        logits = self.classifier(gru_out)  # Shape: (B, L, 1)
        logits = logits.squeeze(-1)        # Shape: (B, L)

        return logits

    @torch.no_grad()
    def predict_probability(self, x_mel: torch.Tensor, x_he: torch.Tensor) -> torch.Tensor:
        """
        Inference helper to return calibrated probabilities in [0.0, 1.0].
        """
        self.eval()
        logits = self.forward(x_mel, x_he)
        return torch.sigmoid(logits)