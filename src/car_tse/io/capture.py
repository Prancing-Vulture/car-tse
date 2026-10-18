"""Helpers for mapping production capture channels to model inputs.

The learned cabin front-end consumes four microphone channels and one aligned
mono playback reference.  A production recorder may store those as six
physical channels (four microphones plus two line-in playback channels).  The
mapping must be explicit in the session metadata; silently treating the
playback channels as microphones makes an evaluation look valid while
feeding the model the wrong signal.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _indices(value: Any, *, name: str) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list of channel indices")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError(f"{name} must contain integer channel indices")
        try:
            index = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain integer channel indices") from exc
        if index < 0 or index in result:
            raise ValueError(f"{name} must contain unique non-negative indices")
        result.append(index)
    return result


def resolve_capture_channels(
    metadata: dict[str, Any],
    *,
    channel_count: int,
    model_mics: int = 4,
) -> tuple[list[int], list[int]]:
    """Return ``(microphone_indices, playback_reference_indices)``.

    A model-compatible four-channel pilot uses all channels as microphones and
    has no playback reference.  Any wider capture requires an explicit
    ``capture_contract`` mapping, so that the evaluator cannot accidentally
    feed line-in channels to the microphone front-end.
    """

    channels = int(channel_count)
    microphones = int(model_mics)
    if channels <= 0 or microphones <= 0:
        raise ValueError("channel_count and model_mics must be positive")
    contract = metadata.get("capture_contract")
    if channels == microphones and not isinstance(contract, dict):
        return list(range(microphones)), []
    if not isinstance(contract, dict):
        raise ValueError(
            f"{channels}-channel capture requires capture_contract channel mapping"
        )

    mic_value = contract.get("microphone_channel_indices")
    playback_value = contract.get("playback_reference_channel_indices", [])
    if mic_value is None:
        raise ValueError("capture_contract missing microphone_channel_indices")
    mic_indices = _indices(mic_value, name="microphone_channel_indices")
    playback_indices = _indices(
        playback_value,
        name="playback_reference_channel_indices",
    )
    if len(mic_indices) != microphones:
        raise ValueError(
            f"capture contract maps {len(mic_indices)} microphone channels, "
            f"but the model expects {microphones}"
        )
    all_indices = mic_indices + playback_indices
    if any(index >= channels for index in all_indices):
        raise ValueError("capture contract refers to a channel outside the recording")
    if len(set(all_indices)) != len(all_indices):
        raise ValueError("microphone and playback channel mappings overlap")
    if bool(contract.get("playback_reference_required", False)) and not playback_indices:
        raise ValueError("capture contract requires a playback reference but maps none")
    return mic_indices, playback_indices


def split_capture_audio(
    audio: np.ndarray,
    metadata: dict[str, Any],
    *,
    model_mics: int = 4,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Split ``[samples, channels]`` audio into model mics and mono playback.

    Multiple playback-reference channels are averaged after the explicit
    mapping.  This keeps the front-end API one-dimensional while retaining the
    stereo line-in energy needed by the learned echo-cancellation branch.
    """

    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"audio must have shape [samples, channels], got {array.shape}")
    mic_indices, playback_indices = resolve_capture_channels(
        metadata,
        channel_count=array.shape[1],
        model_mics=model_mics,
    )
    microphones = np.asarray(array[:, mic_indices], dtype=np.float32)
    playback = None
    if playback_indices:
        playback = np.asarray(array[:, playback_indices].mean(axis=1), dtype=np.float32)
    return microphones, playback

