"""Fixed-length segment crop/pad."""

from __future__ import annotations

import numpy as np


def segment_waveform(
    wav: np.ndarray,
    num_samples: int,
    training: bool = True,
    rng: np.random.Generator | None = None,
    start_frame: int | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    n = wav.shape[0]
    if n == num_samples:
        return wav.astype(np.float32)
    if n > num_samples:
        if start_frame is not None:
            start = min(max(0, int(start_frame)), n - num_samples)
        elif training:
            start = int(rng.integers(0, n - num_samples + 1))
        else:
            start = (n - num_samples) // 2
        return wav[start : start + num_samples].astype(np.float32)
    out = np.zeros(num_samples, dtype=np.float32)
    out[:n] = wav
    return out


def segment_aligned_pair(
    first: np.ndarray,
    second: np.ndarray,
    num_samples: int,
    training: bool = True,
    rng: np.random.Generator | None = None,
    start_frame: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop two time-aligned waveforms with exactly the same offset.

    Mixtures and clean sources share a timeline.  Cropping them through two
    independent calls to ``segment_waveform`` silently destroys that
    alignment whenever the utterance is longer than the requested segment.
    """

    rng = rng or np.random.default_rng()
    first = np.asarray(first)
    second = np.asarray(second)
    cropped = segment_aligned_group([first, second], num_samples, training, rng, start_frame=start_frame)
    return cropped[0], cropped[1]


def segment_aligned_group(
    values: list[np.ndarray],
    num_samples: int,
    training: bool = True,
    rng: np.random.Generator | None = None,
    start_frame: int | None = None,
) -> list[np.ndarray]:
    """Crop/pad several waveforms that share the same timeline.

    Arrays may be mono ``[time]`` or multichannel ``[time, channels]``.  The
    returned arrays all use one common crop offset, which is required when a
    multichannel degraded mixture must remain aligned with its clean target.
    """

    if not values:
        raise ValueError("values must contain at least one waveform")
    rng = rng or np.random.default_rng()
    shared_length = min(value.shape[0] for value in values)
    if shared_length > num_samples and start_frame is not None:
        start = min(max(0, int(start_frame)), shared_length - num_samples)
    elif shared_length > num_samples and training:
        start = int(rng.integers(0, shared_length - num_samples + 1))
    elif shared_length > num_samples:
        start = (shared_length - num_samples) // 2
    else:
        start = 0

    result: list[np.ndarray] = []
    for value in values:
        chunk = value[start : start + num_samples]
        if chunk.shape[0] == num_samples:
            result.append(chunk.astype(np.float32))
            continue
        shape = (num_samples,) + value.shape[1:]
        out = np.zeros(shape, dtype=np.float32)
        out[: chunk.shape[0]] = chunk
        result.append(out)
    return result


def segment_multichannel(
    wav: np.ndarray,
    num_samples: int,
    training: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Crop/pad an aligned ``[channels, samples]`` array without averaging."""
    rng = rng or np.random.default_rng()
    if wav.ndim == 1:
        wav = wav[None, :]
    n = wav.shape[1]
    if n == num_samples:
        return wav.astype(np.float32)
    if n > num_samples:
        if training:
            start = int(rng.integers(0, n - num_samples + 1))
        else:
            start = (n - num_samples) // 2
        return wav[:, start : start + num_samples].astype(np.float32)
    out = np.zeros((wav.shape[0], num_samples), dtype=np.float32)
    out[:, :n] = wav
    return out
