"""Audio-to-intent CRNN."""

from __future__ import annotations

import torch
import torch.nn as nn

from car_tse.models.kws_model import LogMelFront


class IntentClassifier(nn.Module):
    def __init__(
        self,
        sample_rate: int = 8000,
        n_mels: int = 80,
        hidden: int = 256,
        num_layers: int = 2,
        num_classes: int = 31,
    ):
        super().__init__()
        self.mel = LogMelFront(sample_rate, n_mels, n_fft=512, hop=160)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.lstm = nn.LSTM(128 * (n_mels // 4), hidden, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        mel = self.mel(wav).unsqueeze(1)
        x = self.conv(mel)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
