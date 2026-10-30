"""Real-time streaming runner with a capture ring buffer.

The offline evaluators slice pre-recorded audio directly, so their window
cadence is perfect and nothing is ever skipped.  A live microphone is
different: capture delivers small frames on a fixed cadence while inference
takes an unpredictable amount of time.  If inference cannot keep up with the
hop, a naive synchronous loop either grows latency without bound or drops
audio silently.

``StreamingRunner`` decouples capture from inference:

* ``push`` accepts arbitrary-size mono/multichannel frames (for example from a
  ``sounddevice`` callback) into a fixed-capacity ring buffer; an optional
  sample-aligned playback-reference ring follows the same windows;
* ``process_available`` runs the existing windowed pipeline whenever at least
  one new hop is buffered, retaining the configured non-causal context;
* lag is bounded by ``max_lag_seconds``.  ``drop_oldest`` skips whole hops and
  accounts for every skipped sample; ``catch_up`` first spends a bounded
  number of extra windows per call and then drops the remaining excess.

Skipping audio is never free: the wake target in a skipped span is missed, so
``skipped_seconds`` is a first-class metric rather than silent behavior.
"""

from __future__ import annotations

import threading
import time
import inspect
from typing import Any, Callable

import numpy as np
import torch


class RingBuffer:
    """Fixed-capacity ring buffer with absolute sample indexing."""

    def __init__(self, capacity_samples: int, channels: int = 1) -> None:
        self.capacity = int(capacity_samples)
        self.channels = int(channels)
        shape = (self.capacity,) if self.channels == 1 else (self.capacity, self.channels)
        self._data = np.zeros(shape, dtype=np.float32)
        self._write = 0
        self._lock = threading.Lock()
        self.overwritten_samples = 0

    @property
    def total_written(self) -> int:
        return self._write

    def available(self) -> int:
        return min(self._write, self.capacity)

    def first_retained(self) -> int:
        return max(0, self._write - self.capacity)

    def push(self, frame: np.ndarray | torch.Tensor) -> int:
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        arr = np.asarray(frame, dtype=np.float32)
        if arr.ndim == 0:
            raise ValueError("frame must contain at least one sample")
        if arr.ndim == 1 and self.channels > 1:
            arr = arr[:, None]
        elif arr.ndim == 2 and self.channels == 1:
            arr = arr.mean(axis=1)
        if arr.ndim == 1 and self.channels != 1:
            raise ValueError("mono frame pushed to multichannel ring")
        if arr.ndim == 2 and arr.shape[1] != self.channels:
            raise ValueError(
                f"frame has {arr.shape[1]} channels, ring expects {self.channels}"
            )
        num = int(arr.shape[0])
        if num == 0:
            return 0
        with self._lock:
            if num >= self.capacity:
                original = num
                arr = arr[-self.capacity :]
                num = self.capacity
                self.overwritten_samples += self._write + (original - self.capacity)
                self._data[:] = arr
                self._write += original
                return self._write
            start = self._write % self.capacity
            end = start + num
            if end <= self.capacity:
                self._data[start:end] = arr
            else:
                first = self.capacity - start
                self._data[start:] = arr[:first]
                self._data[: num - first] = arr[first:]
            self._write += num
        return self._write

    def read_absolute(self, start: int, end: int) -> np.ndarray:
        start = int(start)
        end = int(end)
        if end <= start:
            raise ValueError("end must be greater than start")
        if end - start > self.capacity:
            raise ValueError("requested range exceeds ring capacity")
        with self._lock:
            if start < self.first_retained() or end > self._write:
                raise ValueError(
                    f"range [{start}, {end}) outside retained [{self.first_retained()}, {self._write})"
                )
            offset = start % self.capacity
            count = end - start
            if offset + count <= self.capacity:
                return np.array(self._data[offset : offset + count], copy=True)
            first = self.capacity - offset
            head = self._data[offset:]
            tail = self._data[: count - first]
            if self.channels == 1:
                return np.concatenate([head, tail]).astype(np.float32)
            return np.concatenate([head, tail], axis=0).astype(np.float32)


class StreamingRunner:
    """Feed arbitrary capture frames into the stateful in-car pipeline."""

    def __init__(
        self,
        forward: Callable[..., tuple[Any, Any]],
        enrollment: np.ndarray | torch.Tensor,
        sample_rate: int = 16000,
        window_seconds: float = 3.0,
        hop_seconds: float = 0.25,
        num_mics: int = 4,
        max_lag_seconds: float = 2.0,
        overload_policy: str = "drop_oldest",
        max_windows_per_call: int = 8,
        device: torch.device | str | None = None,
        state: Any | None = None,
        use_playback_ref: bool = False,
        lock: bool = True,
    ) -> None:
        if overload_policy not in ("drop_oldest", "catch_up"):
            raise ValueError("overload_policy must be 'drop_oldest' or 'catch_up'")
        self.forward = forward
        try:
            parameters = inspect.signature(forward).parameters.values()
            self._forward_accepts_audio_start_sample = any(
                parameter.name == "audio_start_sample"
                or parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            )
        except (TypeError, ValueError):
            self._forward_accepts_audio_start_sample = False
        self.sample_rate = int(sample_rate)
        self.window_samples = int(round(float(window_seconds) * self.sample_rate))
        self.hop_samples = max(1, int(round(float(hop_seconds) * self.sample_rate)))
        self.window_seconds = self.window_samples / self.sample_rate
        self.hop_seconds = self.hop_samples / self.sample_rate
        self.num_mics = int(num_mics)
        self.max_lag_samples = int(round(float(max_lag_seconds) * self.sample_rate))
        self.overload_policy = overload_policy
        self.max_windows_per_call = max(1, int(max_windows_per_call))
        self.device = torch.device(device) if device is not None else None
        self.state = state
        self.use_playback_ref = bool(use_playback_ref)

        capacity = self.window_samples + self.max_lag_samples + self.hop_samples * (
            self.max_windows_per_call + 2
        )
        self.ring = RingBuffer(capacity, 1)
        self.multichannel_ring = RingBuffer(capacity, self.num_mics) if self.num_mics > 1 else None
        self.playback_ref_ring = RingBuffer(capacity, 1) if self.use_playback_ref else None

        if isinstance(enrollment, torch.Tensor):
            self.enrollment = enrollment.detach().cpu().numpy().astype(np.float32)
        else:
            self.enrollment = np.asarray(enrollment, dtype=np.float32)
        self._enrollment_tensor: torch.Tensor | None = None

        self.processed_until = self.window_samples - self.hop_samples
        self.windows_processed = 0
        self.skipped_samples = 0
        self.overwritten_samples = 0
        self.max_lag_samples_seen = 0
        self.wall_seconds = 0.0
        self.wake_events = 0
        self.intent_events = 0
        self.last_output: Any | None = None
        self._lock = threading.Lock() if lock else None

    def _enrollment(self) -> torch.Tensor:
        if self._enrollment_tensor is None:
            tensor = torch.from_numpy(self.enrollment)
            if self.device is not None:
                tensor = tensor.to(self.device)
            self._enrollment_tensor = tensor
        return self._enrollment_tensor

    def lag_seconds(self) -> float:
        return max(0, self.ring.total_written - self.processed_until) / self.sample_rate

    def push(
        self,
        mixture: np.ndarray | torch.Tensor,
        multichannel: np.ndarray | torch.Tensor | None = None,
        playback_ref: np.ndarray | torch.Tensor | None = None,
    ) -> float:
        """Append capture audio. Never blocks on inference.

        ``push`` is safe to call from a capture callback/thread: it only takes
        the ring-buffer locks, while ``process_available`` serializes inference
        under a separate lock. When ``use_playback_ref`` was enabled at
        construction, ``playback_ref`` must have the same number of samples as
        ``mixture`` and is forwarded window-aligned to the pipeline.
        """

        mixture_array = (
            mixture.detach().cpu().numpy() if isinstance(mixture, torch.Tensor) else np.asarray(mixture)
        )
        if mixture_array.ndim == 0:
            raise ValueError("mixture must contain at least one sample")
        frame_samples = int(mixture_array.shape[0])

        # Validate every aligned input before mutating any ring.  A malformed
        # auxiliary frame must not leave the mixture ring ahead of the
        # multichannel/playback rings, otherwise a later valid push can pair
        # the wrong capture samples with a model window.
        multichannel_array = None
        if multichannel is not None:
            if self.multichannel_ring is None:
                raise ValueError("multichannel frame pushed but runner is mono")
            multichannel_array = (
                multichannel.detach().cpu().numpy()
                if isinstance(multichannel, torch.Tensor)
                else np.asarray(multichannel)
            )
            if (
                multichannel_array.ndim != 2
                or multichannel_array.shape[0] != frame_samples
                or multichannel_array.shape[1] != self.num_mics
            ):
                raise ValueError(
                    "multichannel must have shape "
                    f"({frame_samples}, {self.num_mics})"
                )

        playback_array = None
        if self.playback_ref_ring is not None:
            if playback_ref is None:
                raise ValueError("playback_ref is required when use_playback_ref=True")
            playback_array = (
                playback_ref.detach().cpu().numpy()
                if isinstance(playback_ref, torch.Tensor)
                else np.asarray(playback_ref)
            )
            if playback_array.ndim == 2:
                playback_array = playback_array.mean(axis=1)
            if playback_array.ndim != 1 or len(playback_array) != frame_samples:
                raise ValueError("playback_ref must be a mono frame aligned with mixture")
        elif playback_ref is not None:
            raise ValueError("playback_ref supplied but runner was created without use_playback_ref")

        self.ring.push(mixture_array)
        if multichannel_array is not None:
            self.multichannel_ring.push(multichannel_array)
        if self.playback_ref_ring is not None:
            self.playback_ref_ring.push(playback_array)
        return self.lag_seconds()

    def _drop_to(self, target: int) -> None:
        if target <= self.processed_until:
            return
        advance = target - self.processed_until
        advance = int(np.ceil(advance / self.hop_samples) * self.hop_samples)
        self.processed_until += advance
        self.skipped_samples += advance

    def _catch_up_lag(self) -> None:
        if self.ring.total_written - self.processed_until <= self.max_lag_samples:
            return
        if self.overload_policy == "catch_up":
            budget = self.max_windows_per_call
            while (
                budget > 0
                and self.ring.total_written - self.processed_until > self.max_lag_samples
                and self.processed_until + self.hop_samples
                <= self.ring.total_written - self.max_lag_samples + self.hop_samples
            ):
                self._process_one()
                budget -= 1
        if self.ring.total_written - self.processed_until > self.max_lag_samples:
            target = self.ring.total_written - self.max_lag_samples
            self._drop_to(target)

    def _process_one(self) -> Any | None:
        end = self.processed_until + self.hop_samples
        start = end - self.window_samples
        if start < self.ring.first_retained():
            return None
        window = self.ring.read_absolute(start, end)
        multichannel_tensor = None
        if self.multichannel_ring is not None:
            try:
                mc = self.multichannel_ring.read_absolute(start, end)
            except ValueError:
                return None
            multichannel_tensor = torch.from_numpy(np.ascontiguousarray(mc.T))
            if self.device is not None:
                multichannel_tensor = multichannel_tensor.to(self.device)
        playback_tensor = None
        if self.playback_ref_ring is not None:
            try:
                playback = self.playback_ref_ring.read_absolute(start, end)
            except ValueError:
                return None
            playback_tensor = torch.from_numpy(np.ascontiguousarray(playback)).unsqueeze(0)
            if self.device is not None:
                playback_tensor = playback_tensor.to(self.device)
        mixture_tensor = torch.from_numpy(window)
        if self.device is not None:
            mixture_tensor = mixture_tensor.to(self.device)
        started = time.perf_counter()
        forward_kwargs: dict[str, Any] = {"multichannel": multichannel_tensor}
        if self._forward_accepts_audio_start_sample:
            forward_kwargs["audio_start_sample"] = start
        if playback_tensor is not None:
            forward_kwargs["playback_ref"] = playback_tensor
        output, self.state = self.forward(
            mixture_tensor,
            self._enrollment(),
            self.state,
            **forward_kwargs,
        )
        self.wall_seconds += time.perf_counter() - started
        self.processed_until = end
        self.windows_processed += 1
        self.last_output = output
        if getattr(output, "wake_detected", False):
            self.wake_events += 1
        if getattr(output, "intent_ready", False):
            self.intent_events += 1
        return output

    def process_available(self) -> int:
        """Process every hop that is currently buffered; returns output count."""

        if self._lock is not None:
            self._lock.acquire()
        try:
            opened = self.ring.first_retained()
            if self.processed_until < opened:
                self._drop_to(opened)
            self._catch_up_lag()
            outputs = 0
            while self.processed_until + self.hop_samples <= self.ring.total_written:
                result = self._process_one()
                if result is None:
                    break
                outputs += 1
            lag = self.ring.total_written - self.processed_until
            self.max_lag_samples_seen = max(self.max_lag_samples_seen, lag)
            self.overwritten_samples = max(
                self.overwritten_samples, self.ring.overwritten_samples
            )
            return outputs
        finally:
            if self._lock is not None:
                self._lock.release()

    def stats(self) -> dict[str, float]:
        processed_audio = self.windows_processed * self.hop_seconds
        return {
            "pushed_seconds": self.ring.total_written / self.sample_rate,
            "processed_windows": float(self.windows_processed),
            "processed_audio_seconds": processed_audio,
            "skipped_seconds": self.skipped_samples / self.sample_rate,
            "lag_seconds": self.lag_seconds(),
            "max_lag_seconds_seen": self.max_lag_samples_seen / self.sample_rate,
            "inference_wall_seconds": self.wall_seconds,
            "real_time_factor": (
                self.wall_seconds / processed_audio if processed_audio > 0 else 0.0
            ),
            "wake_events": float(self.wake_events),
            "intent_events": float(self.intent_events),
        }

    def reset(self) -> None:
        self.ring = RingBuffer(self.ring.capacity, 1)
        if self.multichannel_ring is not None:
            self.multichannel_ring = RingBuffer(self.ring.capacity, self.num_mics)
        if self.playback_ref_ring is not None:
            self.playback_ref_ring = RingBuffer(self.ring.capacity, 1)
        self.processed_until = self.window_samples - self.hop_samples
        self.windows_processed = 0
        self.skipped_samples = 0
        self.overwritten_samples = 0
        self.max_lag_samples_seen = 0
        self.wall_seconds = 0.0
        self.wake_events = 0
        self.intent_events = 0
        self.last_output = None
        self.state = None
