"""Deterministic spatial front-ends for fixed microphone arrays.

The production plan calls for a cheap seat-zone baseline before committing to
another learned separator.  This module intentionally keeps that baseline
small and explicit: integer-sample delay-and-sum for low-cost streaming use,
and an offline frequency-domain MVDR implementation for comparison studies.

The functions operate on channel-first audio ``[channels, samples]`` by
default.  They do not infer a vehicle's geometry.  Delays must come from a
measured microphone layout or a declared calibration experiment; arbitrary
delays must not be treated as a production seat-zone configuration.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

try:  # Torch is already a project dependency, but keep the module importable for tools.
    import torch
except ImportError:  # pragma: no cover - exercised only outside the project environment.
    torch = None  # type: ignore[assignment]


def _channel_first_numpy(audio: np.ndarray, channel_axis: int) -> np.ndarray:
    array = np.asarray(audio)
    if array.ndim != 2:
        raise ValueError(f"audio must be rank-2, got shape {array.shape}")
    axis = channel_axis if channel_axis >= 0 else array.ndim + channel_axis
    if axis not in (0, 1):
        raise ValueError(f"channel_axis must be 0 or 1, got {channel_axis}")
    return np.moveaxis(array, axis, 0) if axis else array


def _integer_delays(delays_samples: Sequence[int | float], channels: int) -> np.ndarray:
    delays = np.asarray(list(delays_samples), dtype=np.float64)
    if delays.ndim != 1 or len(delays) != channels:
        raise ValueError(f"expected {channels} delay values, got {delays_samples}")
    rounded = np.rint(delays)
    if not np.allclose(delays, rounded):
        raise ValueError("delay-and-sum currently requires integer-sample delays")
    return rounded.astype(np.int64)


def _normalized_weights(weights: Sequence[float] | None, channels: int) -> np.ndarray:
    if weights is None:
        result = np.ones(channels, dtype=np.float64)
    else:
        result = np.asarray(list(weights), dtype=np.float64)
        if result.ndim != 1 or len(result) != channels:
            raise ValueError(f"expected {channels} weights, got {weights}")
    scale = float(np.abs(result).sum())
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("beamformer weights must contain a non-zero finite value")
    return result / scale


def _shift_numpy(channel: np.ndarray, delay: int) -> np.ndarray:
    result = np.zeros_like(channel)
    if delay == 0:
        result[...] = channel
    elif abs(delay) < len(channel):
        if delay > 0:
            result[delay:] = channel[:-delay]
        else:
            result[:delay] = channel[-delay:]
    return result


def delay_and_sum(
    audio: np.ndarray,
    delays_samples: Sequence[int | float],
    weights: Sequence[float] | None = None,
    *,
    channel_axis: int = 0,
) -> np.ndarray:
    """Apply an integer-sample delay-and-sum beamformer.

    A positive delay shifts that microphone's waveform later in time.  The
    returned value is mono and has the same number of samples and floating
    dtype as the input.  Zero padding, rather than circular wrapping, avoids
    introducing artificial audio at chunk boundaries.
    """

    channels_first = _channel_first_numpy(np.asarray(audio), channel_axis)
    delays = _integer_delays(delays_samples, channels_first.shape[0])
    beam_weights = _normalized_weights(weights, channels_first.shape[0])
    result = np.zeros(channels_first.shape[1], dtype=np.result_type(channels_first.dtype, np.float32))
    for channel, (delay, weight) in enumerate(zip(delays, beam_weights)):
        result += float(weight) * _shift_numpy(channels_first[channel], int(delay))
    return result.astype(channels_first.dtype, copy=False)


def gcc_phat_delay(
    reference: np.ndarray,
    signal: np.ndarray,
    *,
    max_delay_samples: int | None = None,
) -> int:
    """Estimate the integer delay of ``signal`` relative to ``reference``.

    This is a diagnostic calibration helper, not a replacement for measured
    microphone geometry.  Positive output means the signal is estimated to be
    delayed relative to the reference.
    """

    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    sig = np.asarray(signal, dtype=np.float64).reshape(-1)
    if len(ref) == 0 or len(sig) == 0:
        raise ValueError("reference and signal must be non-empty")
    fft_size = 1 << int((len(ref) + len(sig) - 1).bit_length())
    cross = np.fft.rfft(sig, fft_size) * np.conj(np.fft.rfft(ref, fft_size))
    correlation = np.fft.irfft(cross / np.maximum(np.abs(cross), 1.0e-12), fft_size)
    limit = min(fft_size // 2 - 1, len(ref) - 1, len(sig) - 1)
    if max_delay_samples is not None:
        limit = min(limit, max(0, int(max_delay_samples)))
    wrapped = np.concatenate((correlation[-limit:], correlation[: limit + 1]))
    lags = np.arange(-limit, limit + 1)
    return int(lags[int(np.argmax(wrapped))])


def _frame_channels(audio: np.ndarray, frame_length: int, hop_length: int) -> tuple[np.ndarray, int]:
    channels, samples = audio.shape
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive")
    frame_count = max(1, int(np.ceil(max(1, samples - frame_length) / hop_length)) + 1)
    padded_samples = (frame_count - 1) * hop_length + frame_length
    padded = np.pad(audio, ((0, 0), (0, padded_samples - samples)))
    frames = np.stack(
        [padded[:, start : start + frame_length] for start in range(0, padded_samples - frame_length + 1, hop_length)],
        axis=0,
    )
    return frames, samples


def mvdr_beamform(
    audio: np.ndarray,
    delays_samples: Sequence[int | float],
    *,
    sample_rate: int = 16000,
    frame_length: int = 512,
    hop_length: int = 128,
    noise_audio: np.ndarray | None = None,
    diagonal_loading: float = 1.0e-2,
    channel_axis: int = 0,
) -> np.ndarray:
    """Apply an offline frequency-domain MVDR beamformer.

    ``delays_samples`` defines the steering direction.  If ``noise_audio`` is
    omitted, the covariance is estimated from the whole input, which is useful
    as a reproducible baseline but can be speech-contaminated.  For deployment
    studies, provide a noise-only calibration segment or a causal covariance
    tracker and report it explicitly.
    """

    channels_first = _channel_first_numpy(np.asarray(audio), channel_axis).astype(np.float64, copy=False)
    channels, samples = channels_first.shape
    delays = _integer_delays(delays_samples, channels).astype(np.float64)
    if sample_rate <= 0 or diagonal_loading < 0.0:
        raise ValueError("sample_rate must be positive and diagonal_loading non-negative")

    window = np.sqrt(np.hanning(frame_length).astype(np.float64))
    frames, original_samples = _frame_channels(channels_first, frame_length, hop_length)
    spectrum = np.fft.rfft(frames * window[None, None, :], axis=-1)
    if noise_audio is None:
        noise_spectrum = spectrum
    else:
        noise_first = _channel_first_numpy(np.asarray(noise_audio), channel_axis).astype(np.float64, copy=False)
        if noise_first.shape[0] != channels:
            raise ValueError("noise_audio must have the same channel count as audio")
        noise_frames, _ = _frame_channels(noise_first, frame_length, hop_length)
        noise_spectrum = np.fft.rfft(noise_frames * window[None, None, :], axis=-1)

    covariance = np.einsum("ncf,ndf->fcd", noise_spectrum, np.conj(noise_spectrum))
    covariance /= max(1, noise_spectrum.shape[0])
    frequencies = np.fft.rfftfreq(frame_length, d=1.0 / float(sample_rate))
    steering = np.exp(
        -2j * np.pi * frequencies[:, None] * delays[None, :] / float(sample_rate)
    )
    identity = np.eye(channels, dtype=np.complex128)[None, :, :]
    trace = np.real(np.trace(covariance, axis1=1, axis2=2)) / max(channels, 1)
    covariance = covariance + identity * (
        float(diagonal_loading) * np.maximum(trace, 1.0e-8)[:, None, None]
    )
    inverse_steering = np.linalg.solve(covariance, steering[..., None])[..., 0]
    denominator = np.sum(np.conj(steering) * inverse_steering, axis=1)
    weights = inverse_steering / np.maximum(np.abs(denominator), 1.0e-12)[:, None]
    output_spectrum = np.einsum("fc,ncf->nf", np.conj(weights), spectrum)

    output_length = (spectrum.shape[0] - 1) * hop_length + frame_length
    output = np.zeros(output_length, dtype=np.float64)
    normalization = np.zeros(output_length, dtype=np.float64)
    for index in range(output_spectrum.shape[0]):
        start = index * hop_length
        output[start : start + frame_length] += np.fft.irfft(
            output_spectrum[index], n=frame_length
        ) * window
        normalization[start : start + frame_length] += np.square(window)
    output /= np.maximum(normalization, 1.0e-8)
    return output[:original_samples].astype(np.asarray(audio).dtype, copy=False)

