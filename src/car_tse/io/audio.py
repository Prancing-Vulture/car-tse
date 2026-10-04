"""Audio I/O helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def _resample_linear(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return data
    n_out = max(1, int(round(data.shape[0] * target_sr / orig_sr)))
    x_old = np.linspace(0.0, 1.0, data.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    if data.ndim == 1:
        return np.interp(x_new, x_old, data).astype(np.float32)
    channels = [np.interp(x_new, x_old, data[:, i]) for i in range(data.shape[1])]
    return np.stack(channels, axis=1).astype(np.float32)


def read_mono(path: Path | str, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), always_2d=True)
    mono = data.mean(axis=1).astype(np.float32)
    if target_sr is not None and sr != target_sr:
        mono = _resample_linear(mono, sr, target_sr)
        sr = target_sr
    return mono, sr


def read_multichannel(path: Path | str, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), always_2d=True)
    data = data.astype(np.float32)
    if target_sr is not None and sr != target_sr:
        data = _resample_linear(data, sr, target_sr)
        sr = target_sr
    return data, sr


def write_mono(path: Path | str, wav: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav.astype(np.float32), sample_rate)
