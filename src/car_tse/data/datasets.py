"""Manifest-backed PyTorch datasets."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from car_tse.data.segment import segment_aligned_group, segment_waveform
from car_tse.io.audio import read_mono, read_multichannel
from car_tse.io.manifest import read_jsonl


def read_mono_segment(path: Path | str, sample_rate: int, samples: int, start_frame: int) -> np.ndarray:
    """Read one aligned mono window without loading a long source recording."""
    with sf.SoundFile(str(path), mode="r") as handle:
        if handle.samplerate != sample_rate:
            raise ValueError(f"Expected {sample_rate} Hz, got {handle.samplerate} Hz for {path}")
        total = len(handle)
        if total <= 0:
            return np.zeros(samples, dtype=np.float32)
        if total > samples:
            start = min(max(0, int(start_frame)), total - samples)
        else:
            start = 0
        handle.seek(start)
        data = handle.read(min(samples, total - start), dtype="float32", always_2d=True)
    out = np.zeros(samples, dtype=np.float32)
    if len(data):
        out[: len(data)] = data.mean(axis=1)
    return out


class TripletExtractionDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path | str,
        sample_rate: int = 8000,
        segment_seconds: float = 4.0,
        training: bool = True,
        multichannel_key: str = "mixture_wav",
        cabin_noise: bool = False,
        num_mics: int = 4,
        cabin_noise_channels: list[int] | tuple[int, ...] | None = None,
        cabin_noise_snr_db: list[float] | tuple[float, ...] | None = None,
        cabin_noise_cache_size: int = 32,
        cabin_noise_cache_dir: Path | str | None = None,
    ):
        self.rows = read_jsonl(manifest_path)
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * segment_seconds)
        self.training = training
        self.multichannel_key = multichannel_key
        self.cabin_noise = cabin_noise
        self.num_mics = num_mics
        self.cabin_noise_channels = tuple(
            cabin_noise_channels if cabin_noise_channels is not None else range(num_mics)
        )
        self.cabin_noise_snr_db = tuple(
            float(x) for x in (cabin_noise_snr_db or (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0))
        )
        self.cabin_noise_cache_size = max(0, int(cabin_noise_cache_size))
        self.cabin_noise_cache: dict[str, np.ndarray] = {}
        self.cabin_noise_cache_order: list[str] = []
        self.cabin_noise_cache_dir = Path(cabin_noise_cache_dir) if cabin_noise_cache_dir else None
        self.rng = np.random.default_rng(42)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        segment_start = row.get("segment_start_seconds")
        segment_start_frame = (
            None if segment_start is None else int(round(float(segment_start) * self.sample_rate))
        )
        enroll_start = row.get("enroll_start_seconds")
        enroll_start_frame = (
            None if enroll_start is None else int(round(float(enroll_start) * self.sample_rate))
        )
        if segment_start_frame is None:
            mix, _ = read_mono(row["mixture_wav"], self.sample_rate)
            target, _ = read_mono(row["target_wav"], self.sample_rate)
        else:
            mix = read_mono_segment(row["mixture_wav"], self.sample_rate, self.num_samples, segment_start_frame)
            target = read_mono_segment(row["target_wav"], self.sample_rate, self.num_samples, segment_start_frame)
        if enroll_start_frame is None:
            enroll, _ = read_mono(row["enroll_wav"], self.sample_rate)
            enroll = segment_waveform(enroll, self.num_samples // 2, self.training, self.rng)
        else:
            enroll = read_mono_segment(
                row["enroll_wav"], self.sample_rate, self.num_samples // 2, enroll_start_frame
            )

        # Load optional signals before choosing the crop so every waveform
        # that shares the mixture timeline receives the same offset.
        multichannel: np.ndarray | None = None
        cabin_noise_path = row.get("cabin_noise_wav")
        if self.cabin_noise and cabin_noise_path and Path(cabin_noise_path).exists():
            try:
                cache_key = f"{cabin_noise_path}|{self.sample_rate}"
                noise = self.cabin_noise_cache.get(cache_key)
                if noise is None:
                    preprocessed = None
                    if self.cabin_noise_cache_dir is not None:
                        digest = hashlib.sha1(str(Path(cabin_noise_path)).encode("utf-8")).hexdigest()
                        candidate = self.cabin_noise_cache_dir / f"{digest}.npy"
                        if candidate.exists():
                            try:
                                preprocessed = np.load(candidate, mmap_mode="r", allow_pickle=False)
                            except (OSError, ValueError):
                                preprocessed = None
                    noise = preprocessed if preprocessed is not None else read_multichannel(cabin_noise_path, self.sample_rate)[0]
                    if self.cabin_noise_cache_size:
                        self.cabin_noise_cache[cache_key] = noise
                        self.cabin_noise_cache_order.append(cache_key)
                        if len(self.cabin_noise_cache_order) > self.cabin_noise_cache_size:
                            oldest = self.cabin_noise_cache_order.pop(0)
                            self.cabin_noise_cache.pop(oldest, None)
                if noise.ndim == 1:
                    noise = noise[:, None]
                requested_channels = row.get("cabin_noise_channels", self.cabin_noise_channels)
                requested_channels = [int(x) for x in requested_channels]
                valid_channels = [x for x in requested_channels if 0 <= x < noise.shape[1]]
                if not valid_channels:
                    valid_channels = list(range(min(self.num_mics, noise.shape[1])))
                noise = noise[:, valid_channels]
                if noise.shape[1] < self.num_mics:
                    repeats = int(np.ceil(self.num_mics / noise.shape[1]))
                    noise = np.tile(noise, (1, repeats))[:, : self.num_mics]
                elif noise.shape[1] > self.num_mics:
                    noise = noise[:, : self.num_mics]

                # CAVEMOVE clips are longer than a training segment in most
                # cases.  Crop/loop them to the complete mixture timeline;
                # the common speech/target crop below then remains aligned.
                if noise.shape[0] >= len(mix):
                    if self.training:
                        start = int(self.rng.integers(0, noise.shape[0] - len(mix) + 1))
                    else:
                        start = (noise.shape[0] - len(mix)) // 2
                    noise = noise[start : start + len(mix)]
                else:
                    repeats = int(np.ceil(len(mix) / max(1, noise.shape[0])))
                    noise = np.tile(noise, (repeats, 1))[: len(mix)]

                row_snr = row.get("cabin_noise_snr_db")
                if self.training:
                    snr_db = float(self.rng.choice(self.cabin_noise_snr_db))
                elif row_snr is not None:
                    snr_db = float(row_snr)
                else:
                    snr_db = float(self.cabin_noise_snr_db[0])
                speech_rms = float(np.sqrt(np.mean(mix * mix) + 1e-8))
                noise_rms = float(np.sqrt(np.mean(noise * noise) + 1e-8))
                if noise_rms > 1e-6 and speech_rms > 1e-6:
                    scale = speech_rms * (10.0 ** (-snr_db / 20.0)) / noise_rms
                    multichannel = mix[:, None] + noise * scale
            except (OSError, RuntimeError, ValueError):
                multichannel = None

        mc_paths = row.get("mixture_multich_wavs")
        mc_path = row.get("mixture_multich_wav") or row.get(self.multichannel_key)
        if multichannel is None and mc_paths and all(Path(p).exists() for p in mc_paths):
            try:
                if segment_start_frame is None:
                    channels = [read_mono(p, self.sample_rate)[0] for p in mc_paths]
                    length = min(len(channel) for channel in channels)
                    multichannel = np.stack([channel[:length] for channel in channels], axis=1)
                else:
                    channels = [
                        read_mono_segment(p, self.sample_rate, self.num_samples, segment_start_frame)
                        for p in mc_paths
                    ]
                    multichannel = np.stack(channels, axis=1)
                if multichannel.shape[1] < self.num_mics:
                    repeats = int(np.ceil(self.num_mics / multichannel.shape[1]))
                    multichannel = np.tile(multichannel, (1, repeats))[:, : self.num_mics]
                elif multichannel.shape[1] > self.num_mics:
                    multichannel = multichannel[:, : self.num_mics]
            except OSError:
                multichannel = None
        elif multichannel is None and mc_path and Path(mc_path).exists() and Path(mc_path).suffix:
            try:
                multichannel, _ = read_multichannel(mc_path, self.sample_rate)
            except OSError:
                multichannel = None

        if multichannel is not None:
            if multichannel.ndim == 1:
                multichannel = multichannel[:, None]
            if multichannel.shape[1] < self.num_mics:
                repeats = int(np.ceil(self.num_mics / multichannel.shape[1]))
                multichannel = np.tile(multichannel, (1, repeats))[:, : self.num_mics]
            elif multichannel.shape[1] > self.num_mics:
                multichannel = multichannel[:, : self.num_mics]

        playback: np.ndarray | None = None
        if row.get("playback_ref_wav"):
            if segment_start_frame is None:
                playback, _ = read_mono(row["playback_ref_wav"], self.sample_rate)
            else:
                playback = read_mono_segment(
                    row["playback_ref_wav"], self.sample_rate, self.num_samples, segment_start_frame
                )

        aligned_values: list[np.ndarray] = [mix, target]
        aligned_names = ["mixture", "target"]
        if multichannel is not None:
            aligned_values.append(multichannel)
            aligned_names.append("multichannel")
        if playback is not None:
            aligned_values.append(playback)
            aligned_names.append("playback")
        aligned = segment_aligned_group(
            aligned_values,
            self.num_samples,
            self.training,
            self.rng,
            start_frame=segment_start_frame,
        )
        aligned_map = dict(zip(aligned_names, aligned))
        item = {
            "mixture": torch.from_numpy(aligned_map["mixture"]),
            "enrollment": torch.from_numpy(enroll),
            "target": torch.from_numpy(aligned_map["target"]),
            "target_present": float(row.get("target_present", 1)),
            # Binary presence manifests predate the personal-activity head.
            # Keep their behavior (present -> target class 1, absent -> class
            # 0) while allowing new manifests to distinguish target speech
            # (1), non-target speech (2), and non-speech (0).
            "target_activity_class": int(
                row.get("target_activity_class", 1 if row.get("target_present", 1) else 0)
            ),
            "speaker_id": row.get("speaker_id", ""),
        }
        if "speaker_label" in row:
            item["speaker_label"] = int(row["speaker_label"])
        if "multichannel" in aligned_map:
            # Dataset convention is [channels, time]; aligned_group works in
            # [time, channels] so the common crop is explicit.
            item["multichannel"] = torch.from_numpy(aligned_map["multichannel"].T)
        if "playback" in aligned_map:
            item["playback_ref"] = torch.from_numpy(aligned_map["playback"])
        return item


class PresenceDataset(TripletExtractionDataset):
    pass


class KWSDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path | str,
        sample_rate: int = 8000,
        segment_seconds: float = 1.0,
        training: bool = True,
        cabin_noise: bool = False,
        cabin_noise_snr_db: list[float] | tuple[float, ...] | None = None,
        cabin_noise_cache_size: int = 32,
        audio_cache_size: int = 2048,
        cabin_noise_cache_dir: Path | str | None = None,
        waveform_cache_path: Path | str | None = None,
        waveform_cache_is_noisy: bool = False,
    ):
        self.rows = read_jsonl(manifest_path)
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * segment_seconds)
        self.training = training
        self.cabin_noise = cabin_noise
        self.cabin_noise_snr_db = tuple(
            float(x) for x in (cabin_noise_snr_db or (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0))
        )
        self.cabin_noise_cache_size = max(0, int(cabin_noise_cache_size))
        self.cabin_noise_cache: dict[str, np.ndarray] = {}
        self.cabin_noise_cache_order: list[str] = []
        self.cabin_noise_cache_dir = Path(cabin_noise_cache_dir) if cabin_noise_cache_dir else None
        self.waveform_cache = None
        self.waveform_cache_is_noisy = bool(waveform_cache_is_noisy)
        if waveform_cache_path:
            cache_path = Path(waveform_cache_path)
            if cache_path.exists():
                try:
                    cached = np.load(cache_path, mmap_mode="r", allow_pickle=False)
                    expected_shape = (len(self.rows), self.num_samples)
                    if tuple(cached.shape) == expected_shape:
                        self.waveform_cache = cached
                except (OSError, ValueError):
                    self.waveform_cache = None
        self.audio_cache_size = max(0, int(audio_cache_size))
        self.audio_cache: dict[str, np.ndarray] = {}
        self.audio_cache_order: list[str] = []
        self.rng = np.random.default_rng(43)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        wav_path = row["wav"]
        if self.waveform_cache is not None:
            wav = np.array(self.waveform_cache[idx], dtype=np.float32, copy=True)
        else:
            wav = self.audio_cache.get(wav_path)
            if wav is None:
                wav, _ = read_mono(wav_path, self.sample_rate)
                if self.audio_cache_size and not row.get("noise_only", False):
                    self.audio_cache[wav_path] = wav
                    self.audio_cache_order.append(wav_path)
                    if len(self.audio_cache_order) > self.audio_cache_size:
                        oldest = self.audio_cache_order.pop(0)
                        self.audio_cache.pop(oldest, None)
        cabin_noise_path = row.get("cabin_noise_wav")
        if (
            self.cabin_noise
            and not self.waveform_cache_is_noisy
            and cabin_noise_path
            and Path(cabin_noise_path).exists()
        ):
            try:
                cache_key = f"{cabin_noise_path}|{self.sample_rate}"
                noise = self.cabin_noise_cache.get(cache_key)
                if noise is None:
                    preprocessed = None
                    if self.cabin_noise_cache_dir is not None:
                        digest = hashlib.sha1(str(Path(cabin_noise_path)).encode("utf-8")).hexdigest()
                        candidate = self.cabin_noise_cache_dir / f"{digest}.npy"
                        if candidate.exists():
                            try:
                                preprocessed = np.load(candidate, mmap_mode="r", allow_pickle=False)
                            except (OSError, ValueError):
                                preprocessed = None
                    noise = preprocessed if preprocessed is not None else read_mono(cabin_noise_path, self.sample_rate)[0]
                    if self.cabin_noise_cache_size:
                        self.cabin_noise_cache[cache_key] = noise
                        self.cabin_noise_cache_order.append(cache_key)
                        if len(self.cabin_noise_cache_order) > self.cabin_noise_cache_size:
                            oldest = self.cabin_noise_cache_order.pop(0)
                            self.cabin_noise_cache.pop(oldest, None)
                if len(noise) >= len(wav):
                    if self.training:
                        start = int(self.rng.integers(0, len(noise) - len(wav) + 1))
                    else:
                        start = (len(noise) - len(wav)) // 2
                    noise = noise[start : start + len(wav)]
                else:
                    repeats = int(np.ceil(len(wav) / max(1, len(noise))))
                    noise = np.tile(noise, repeats)[: len(wav)]
                row_snr = row.get("cabin_noise_snr_db")
                snr_db = (
                    float(self.rng.choice(self.cabin_noise_snr_db))
                    if self.training
                    else float(row_snr if row_snr is not None else self.cabin_noise_snr_db[0])
                )
                speech_rms = float(np.sqrt(np.mean(wav * wav) + 1e-8))
                noise_rms = float(np.sqrt(np.mean(noise * noise) + 1e-8))
                if speech_rms > 1e-6 and noise_rms > 1e-6:
                    wav = wav + noise * (speech_rms * (10.0 ** (-snr_db / 20.0)) / noise_rms)
            except (OSError, RuntimeError, ValueError):
                pass
        wav = segment_waveform(wav, self.num_samples, self.training, self.rng)
        return {"wav": torch.from_numpy(wav), "label": int(row["label"])}


class IntentDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path | str,
        sample_rate: int = 8000,
        segment_seconds: float = 2.0,
        training: bool = True,
        cabin_noise: bool = False,
        cabin_noise_snr_db: list[float] | tuple[float, ...] | None = None,
        cabin_noise_cache_size: int = 32,
        audio_cache_size: int = 2048,
        waveform_cache_path: Path | str | None = None,
        cabin_noise_cache_dir: Path | str | None = None,
        waveform_cache_is_noisy: bool = False,
    ):
        self.rows = read_jsonl(manifest_path)
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * segment_seconds)
        self.training = training
        self.cabin_noise = cabin_noise
        self.cabin_noise_snr_db = tuple(
            float(x) for x in (cabin_noise_snr_db or (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0))
        )
        self.cabin_noise_cache_size = max(0, int(cabin_noise_cache_size))
        self.cabin_noise_cache: dict[str, np.ndarray] = {}
        self.cabin_noise_cache_order: list[str] = []
        self.cabin_noise_cache_dir = Path(cabin_noise_cache_dir) if cabin_noise_cache_dir else None
        self.waveform_cache_is_noisy = bool(waveform_cache_is_noisy)
        self.audio_cache_size = max(0, int(audio_cache_size))
        self.audio_cache: dict[str, np.ndarray] = {}
        self.audio_cache_order: list[str] = []
        self.waveform_cache = None
        if waveform_cache_path:
            cache_path = Path(waveform_cache_path)
            if cache_path.exists():
                try:
                    cached = np.load(cache_path, mmap_mode="r", allow_pickle=False)
                    expected_shape = (len(self.rows), self.num_samples)
                    if tuple(cached.shape) == expected_shape:
                        self.waveform_cache = cached
                except (OSError, ValueError):
                    self.waveform_cache = None
        self.rng = np.random.default_rng(44)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        path = row.get("wav") or row.get("mixture_wav")
        if self.waveform_cache is not None:
            # The cache is a read-only memmap; make a small mutable copy before
            # applying cabin noise so worker processes never modify it.
            wav = np.array(self.waveform_cache[idx], dtype=np.float32, copy=True)
        else:
            wav = self.audio_cache.get(path)
            if wav is None:
                wav, _ = read_mono(path, self.sample_rate)
                if self.audio_cache_size:
                    self.audio_cache[path] = wav
                    self.audio_cache_order.append(path)
                    if len(self.audio_cache_order) > self.audio_cache_size:
                        oldest = self.audio_cache_order.pop(0)
                        self.audio_cache.pop(oldest, None)
        cabin_noise_path = row.get("cabin_noise_wav")
        if (
            self.cabin_noise
            and not self.waveform_cache_is_noisy
            and cabin_noise_path
            and Path(cabin_noise_path).exists()
        ):
            try:
                cache_key = f"{cabin_noise_path}|{self.sample_rate}"
                noise = self.cabin_noise_cache.get(cache_key)
                if noise is None:
                    preprocessed = None
                    if self.cabin_noise_cache_dir is not None:
                        digest = hashlib.sha1(str(Path(cabin_noise_path)).encode("utf-8")).hexdigest()
                        candidate = self.cabin_noise_cache_dir / f"{digest}.npy"
                        if candidate.exists():
                            try:
                                preprocessed = np.load(candidate, mmap_mode="r", allow_pickle=False)
                            except (OSError, ValueError):
                                preprocessed = None
                    noise = preprocessed if preprocessed is not None else read_mono(cabin_noise_path, self.sample_rate)[0]
                    if self.cabin_noise_cache_size:
                        self.cabin_noise_cache[cache_key] = noise
                        self.cabin_noise_cache_order.append(cache_key)
                        if len(self.cabin_noise_cache_order) > self.cabin_noise_cache_size:
                            oldest = self.cabin_noise_cache_order.pop(0)
                            self.cabin_noise_cache.pop(oldest, None)
                if len(noise) >= len(wav):
                    if self.training:
                        start = int(self.rng.integers(0, len(noise) - len(wav) + 1))
                    else:
                        start = (len(noise) - len(wav)) // 2
                    noise = noise[start : start + len(wav)]
                else:
                    repeats = int(np.ceil(len(wav) / max(1, len(noise))))
                    noise = np.tile(noise, repeats)[: len(wav)]
                row_snr = row.get("cabin_noise_snr_db")
                snr_db = (
                    float(self.rng.choice(self.cabin_noise_snr_db))
                    if self.training
                    else float(row_snr if row_snr is not None else self.cabin_noise_snr_db[0])
                )
                wav_rms = float(np.sqrt(np.mean(wav * wav) + 1e-8))
                noise_rms = float(np.sqrt(np.mean(noise * noise) + 1e-8))
                if wav_rms > 1e-6 and noise_rms > 1e-6:
                    wav = wav + noise * (wav_rms * (10.0 ** (-snr_db / 20.0)) / noise_rms)
            except (OSError, RuntimeError, ValueError):
                pass
        wav = segment_waveform(wav, self.num_samples, self.training, self.rng)
        return {"wav": torch.from_numpy(wav), "label": int(row["label"])}
