"""Shared window-start rules for offline and streaming evaluations."""

from __future__ import annotations


def full_context_window_starts(
    num_samples: int,
    window_samples: int,
    hop_samples: int,
) -> range:
    """Return starts that have a complete context window.

    A clip shorter than one model context still gets one start so callers can
    deliberately pad that short input. For longer clips, the incomplete tail
    is excluded. Counting a zero-padded tail as an ordinary negative window
    can create artificial wake detections and corrupt false-accept rates.
    """

    num = int(num_samples)
    window = int(window_samples)
    hop = int(hop_samples)
    if num < 0:
        raise ValueError("num_samples must be non-negative")
    if window <= 0:
        raise ValueError("window_samples must be positive")
    if hop <= 0:
        raise ValueError("hop_samples must be positive")
    return range(0, max(1, num - window + 1), hop)
