"""Manifest-backed PyTorch datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from car_tse.data.segment import segment_waveform
from car_tse.io.audio import read_mono, read_multichannel
from car_tse.io.manifest import read_jsonl


class TripletExtractionDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path | str,
        sample_rate: int = 8000,
        segment_seconds: float = 4.0,
        training: bool = True,
        multichannel_key: str = "mixture_wav",
    ):
        self.rows = read_jsonl(manifest_path)
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * segment_seconds)
        self.training = training
        self.multichannel_key = multichannel_key
        self.rng = np.random.default_rng(42)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        mix, _ = read_mono(row["mixture_wav"])
        enroll, _ = read_mono(row["enroll_wav"])
        target, _ = read_mono(row["target_wav"])
        mix = segment_waveform(mix, self.num_samples, self.training, self.rng)
        enroll = segment_waveform(enroll, self.num_samples // 2, self.training, self.rng)
        target = segment_waveform(target, self.num_samples, self.training, self.rng)
        item = {
            "mixture": torch.from_numpy(mix),
            "enrollment": torch.from_numpy(enroll),
            "target": torch.from_numpy(target),
            "target_present": float(row.get("target_present", 1)),
            "speaker_id": row.get("speaker_id", ""),
        }
        if "speaker_label" in row:
            item["speaker_label"] = int(row["speaker_label"])
        mc_path = row.get("mixture_multich_wav") or row.get(self.multichannel_key)
        if mc_path and Path(mc_path).exists() and Path(mc_path).suffix:
            try:
                mc, _ = read_multichannel(mc_path)
                if mc.ndim == 2:
                    mc = mc.T
                item["multichannel"] = torch.from_numpy(
                    segment_waveform(mc.mean(axis=0), self.num_samples, self.training, self.rng)
                )
            except OSError:
                pass
        if row.get("playback_ref_wav"):
            pb, _ = read_mono(row["playback_ref_wav"])
            item["playback_ref"] = torch.from_numpy(
                segment_waveform(pb, self.num_samples, self.training, self.rng)
            )
        return item


class PresenceDataset(TripletExtractionDataset):
    pass


class KWSDataset(Dataset):
    def __init__(self, manifest_path: Path | str, sample_rate: int = 8000, segment_seconds: float = 1.0, training: bool = True):
        self.rows = read_jsonl(manifest_path)
        self.num_samples = int(sample_rate * segment_seconds)
        self.training = training
        self.rng = np.random.default_rng(43)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        wav, _ = read_mono(row["wav"])
        wav = segment_waveform(wav, self.num_samples, self.training, self.rng)
        return {"wav": torch.from_numpy(wav), "label": int(row["label"])}


class IntentDataset(Dataset):
    def __init__(self, manifest_path: Path | str, sample_rate: int = 8000, segment_seconds: float = 2.0, training: bool = True):
        self.rows = read_jsonl(manifest_path)
        self.num_samples = int(sample_rate * segment_seconds)
        self.training = training
        self.rng = np.random.default_rng(44)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        path = row.get("wav") or row.get("mixture_wav")
        wav, _ = read_mono(path)
        wav = segment_waveform(wav, self.num_samples, self.training, self.rng)
        return {"wav": torch.from_numpy(wav), "label": int(row["label"])}
