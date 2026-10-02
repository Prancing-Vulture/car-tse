"""Fixed-length segment crop/pad."""

from __future__ import annotations

import numpy as np


def segment_waveform(
    wav: np.ndarray,
    num_samples: int,
    training: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    n = wav.shape[0]
    if n == num_samples:
        return wav.astype(np.float32)
    if n > num_samples:
        if training:
            start = int(rng.integers(0, n - num_samples + 1))
        else:
            start = (n - num_samples) // 2
        return wav[start : start + num_samples].astype(np.float32)
    out = np.zeros(num_samples, dtype=np.float32)
    out[:n] = wav
    return out
