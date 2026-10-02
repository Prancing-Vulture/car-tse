"""Audio I/O helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def read_mono(path: Path | str) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), always_2d=True)
    mono = data.mean(axis=1).astype(np.float32)
    return mono, sr


def read_multichannel(path: Path | str) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), always_2d=True)
    return data.astype(np.float32), sr


def write_mono(path: Path | str, wav: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav.astype(np.float32), sample_rate)
