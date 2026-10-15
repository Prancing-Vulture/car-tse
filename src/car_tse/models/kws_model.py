"""Lightweight KWS on log-mel."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogMelFront(nn.Module):
    def __init__(
        self,
        sample_rate: int = 8000,
        n_mels: int = 40,
        n_fft: int = 256,
        hop: int = 80,
        exportable_frontend: bool = False,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop = hop
        self.exportable_frontend = bool(exportable_frontend)
        self.register_buffer("mel_fb", torch.tensor(self._mel_filterbank(sample_rate, n_fft, n_mels)))
        self.register_buffer("window", torch.hann_window(n_fft))
        if self.exportable_frontend:
            frequencies = torch.arange(n_fft // 2 + 1, dtype=torch.float32).view(-1, 1)
            samples = torch.arange(n_fft, dtype=torch.float32).view(1, -1)
            phase = 2.0 * torch.pi * frequencies * samples / float(n_fft)
            real = torch.cos(phase) * self.window.view(1, -1)
            imag = -torch.sin(phase) * self.window.view(1, -1)
            # Interleave real/imaginary rows so a single fixed Conv1d produces
            # the same one-sided DFT coefficients as torch.stft.
            kernel = torch.stack((real, imag), dim=1).reshape(-1, 1, n_fft)
            self.register_buffer("dft_kernel", kernel)

    @staticmethod
    def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> list[list[float]]:
        import numpy as np

        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        fmax = sr / 2
        mels = np.linspace(hz_to_mel(0), hz_to_mel(fmax), n_mels + 2)
        hz = mel_to_hz(mels)
        bins = np.floor((n_fft + 1) * hz / sr).astype(int)
        fb = np.zeros((n_mels, n_fft // 2 + 1))
        for i in range(n_mels):
            for j in range(bins[i], bins[i + 1]):
                if j < fb.shape[1]:
                    fb[i, j] = (j - bins[i]) / max(bins[i + 1] - bins[i], 1)
            for j in range(bins[i + 1], bins[i + 2]):
                if j < fb.shape[1]:
                    fb[i, j] = (bins[i + 2] - j) / max(bins[i + 2] - bins[i + 1], 1)
        return fb.tolist()

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if self.exportable_frontend:
            return self._forward_exportable(wav)
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.n_fft,
            window=self.window,
            return_complex=True,
        )
        power = spec.abs() ** 2
        mel = torch.matmul(self.mel_fb, power)
        return torch.log(mel + 1e-6)

    def _forward_exportable(self, wav: torch.Tensor) -> torch.Tensor:
        """Real-valued STFT equivalent that can be lowered to ONNX.

        The ordinary path uses ``torch.stft(return_complex=True)``, which is
        numerically convenient but is not accepted by the legacy ONNX
        exporter used by several embedded toolchains.  A fixed real/imaginary
        DFT convolution preserves the same centered, reflect-padded framing
        without complex tensors.
        """

        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        padded = F.pad(wav.unsqueeze(1), (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        coefficients = F.conv1d(padded, self.dft_kernel, stride=self.hop)
        real = coefficients[:, 0::2, :]
        imag = coefficients[:, 1::2, :]
        power = real.square() + imag.square()
        mel = torch.matmul(self.mel_fb, power)
        return torch.log(mel + 1e-6)


class KWSModel(nn.Module):
    def __init__(
        self,
        sample_rate: int = 8000,
        n_mels: int = 40,
        channels: list[int] | None = None,
        num_classes: int = 36,
        exportable_frontend: bool = False,
    ):
        super().__init__()
        channels = channels or [32, 64, 128]
        self.mel = LogMelFront(sample_rate, n_mels, exportable_frontend=exportable_frontend)
        layers: list[nn.Module] = []
        in_ch = 1
        for ch in channels:
            layers.extend(
                [
                    nn.Conv2d(in_ch, ch, 3, padding=1),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                ]
            )
            in_ch = ch
        self.cnn = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_ch, num_classes)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(wav))

    def forward_features(self, wav: torch.Tensor) -> torch.Tensor:
        """Return the pooled acoustic representation before classification."""

        mel = self.mel(wav).unsqueeze(1)
        x = self.cnn(mel)
        return x.mean(dim=(-1, -2))


class TargetSpeakerKWSModel(nn.Module):
    """Wake-word detector conditioned on the enrolled target speaker.

    The ordinary KWS model only sees the extracted waveform.  If an off-target
    passenger leaks through Stage 1, it has no identity signal with which to
    reject that passenger.  This head keeps the same compact log-mel CNN and
    augments it with a learned relation between the enrollment and extracted
    speaker embeddings.  Embeddings are produced by the frozen/transfer SpEx+
    encoder upstream and are cached during training for speed.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 40,
        channels: list[int] | None = None,
        num_classes: int = 2,
        speaker_embedding_dim: int = 256,
        hidden: int = 256,
        exportable_frontend: bool = False,
    ):
        super().__init__()
        self.audio = KWSModel(
            sample_rate=sample_rate,
            n_mels=n_mels,
            channels=channels,
            num_classes=num_classes,
            exportable_frontend=exportable_frontend,
        )
        audio_dim = int((channels or [32, 64, 128])[-1])
        relation_dim = int(speaker_embedding_dim) * 4 + 1
        self.classifier = nn.Sequential(
            nn.Linear(audio_dim + relation_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, num_classes),
        )

    def forward(
        self,
        wav: torch.Tensor,
        enrollment_embedding: torch.Tensor,
        extracted_embedding: torch.Tensor,
    ) -> torch.Tensor:
        audio_features = self.audio.forward_features(wav)
        enrollment_embedding = F.normalize(enrollment_embedding.float(), dim=-1)
        extracted_embedding = F.normalize(extracted_embedding.float(), dim=-1)
        relation = torch.cat(
            [
                enrollment_embedding,
                extracted_embedding,
                torch.abs(enrollment_embedding - extracted_embedding),
                enrollment_embedding * extracted_embedding,
                F.cosine_similarity(
                    enrollment_embedding, extracted_embedding, dim=-1
                ).unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.classifier(torch.cat([audio_features, relation], dim=-1))
