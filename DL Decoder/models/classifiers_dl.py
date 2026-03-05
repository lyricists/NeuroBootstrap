import os, pickle, time
import numpy as np
from typing import Tuple, Dict, List

# Torch / DL
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Metrics
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight


# ----------------------------
# Models
# ----------------------------


class EEGNet(nn.Module):
    # Lawhern et al. 2018 (compact CNN)
    def __init__(
        self, n_channels: int, n_samples: int, n_classes: int, dropout: float = 0.5
    ):
        super().__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 8, (1, 16), padding=(0, 8), bias=False), nn.BatchNorm2d(8)
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(8, 16, (n_channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(16, 16, (1, 8), padding=(0, 4), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        # time downsample factor ~ 4*4 = 16 (depending on padding/valid)
        # use adaptive pooling to avoid hand-calculating fc size:
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, n_classes)

    def forward(self, x):
        # x: (B, 1, C, T)
        x = self.firstconv(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = self.gap(x)  # (B, 16, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 16)
        return self.fc(x)


class ShallowConvNet(nn.Module):
    # Schirrmeister et al. 2017 (JNE)
    def __init__(
        self, n_channels: int, n_samples: int, n_classes: int, dropout: float = 0.5
    ):
        super().__init__()
        self.conv_time = nn.Conv2d(
            1, 40, (1, 13), bias=True
        )  # smaller kernel for short windows
        self.conv_spat = nn.Conv2d(40, 40, (n_channels, 1), bias=True)
        self.batchnorm = nn.BatchNorm2d(40)
        self.pool = nn.AvgPool2d((1, 15), stride=(1, 5))
        self.drop = nn.Dropout(dropout)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(40, n_classes)

    def forward(self, x):
        # x: (B, 1, C, T)
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.batchnorm(x)
        x = x**2
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.drop(x)
        x = self.gap(x)  # (B, 40, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 40)
        return self.fc(x)


class DeepConvNet(nn.Module):
    """Schirrmeister et al. 2017 (deeper CNN)
    Adapted for short EEG windows (e.g., 3×50)
    """

    def __init__(
        self, n_channels: int, n_samples: int, n_classes: int, dropout: float = 0.5
    ):
        super().__init__()

        # Dynamically adapt kernel size based on time window
        # make sure it never exceeds n_samples
        ksize = min(5, max(3, n_samples // 10))  # typical: 3–5 for short windows

        # Block 1: temporal + spatial convolutions
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 25, (1, ksize), bias=False, padding=(0, ksize // 2)),
            nn.Conv2d(25, 25, (n_channels, 1), bias=False),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropout),
        )

        # Use smaller kernels in deeper layers (3 instead of 5)
        self.block2 = self._block(25, 50, ksize=3, dropout=dropout)
        self.block3 = self._block(50, 100, ksize=3, dropout=dropout)

        # Optional 4th block only if enough timepoints remain
        if n_samples >= 75:
            self.block4 = self._block(100, 200, ksize=3, dropout=dropout)
            final_channels = 200
        else:
            self.block4 = nn.Identity()
            final_channels = 100

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(final_channels, n_classes)

    def _block(self, in_ch, out_ch, ksize, dropout):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (1, ksize), bias=False, padding=(0, ksize // 2)),
            nn.BatchNorm2d(out_ch),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, 1, C, T)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)  # (B, final_channels, 1, 1)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class MLP(nn.Module):
    def __init__(
        self, input_dim: int, n_classes: int, hidden: int = 128, dropout: float = 0.3
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        # x: (B, input_dim)
        return self.net(x)
