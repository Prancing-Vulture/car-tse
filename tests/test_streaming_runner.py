from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from car_tse.pipeline.streaming_runner import RingBuffer, StreamingRunner


class FakePipeline:
    def __init__(self) -> None:
        self.calls = 0
        self.window_lengths: list[int] = []

    def forward(self, mixture, enrollment, state=None, multichannel=None):
        self.calls += 1
        self.window_lengths.append(int(mixture.shape[-1]))
        if state is None:
            state = {"windows": 0}
        state["windows"] += 1
        output = SimpleNamespace(wake_detected=False, intent_ready=False)
        return output, state


def make_runner(
    policy: str = "drop_oldest",
    max_windows: int = 8,
    max_lag: float = 0.5,
    sample_rate: int = 1000,
) -> tuple[FakePipeline, StreamingRunner]:
    pipeline = FakePipeline()
    runner = StreamingRunner(
        forward=pipeline.forward,
        enrollment=np.zeros(160, dtype=np.float32),
        sample_rate=sample_rate,
        window_seconds=0.3,
        hop_seconds=0.1,
        num_mics=1,
        max_lag_seconds=max_lag,
        overload_policy=policy,
        max_windows_per_call=max_windows,
    )
    return pipeline, runner


def test_ring_buffer_wraparound_read() -> None:
    ring = RingBuffer(capacity_samples=10, channels=1)
    for value in range(25):
        ring.push(np.array([value], dtype=np.float32))
    data = ring.read_absolute(15, 25)
    assert np.allclose(data, np.arange(15, 25, dtype=np.float32))
    assert ring.available() == 10


def test_ring_buffer_overwrite_accounting() -> None:
    ring = RingBuffer(capacity_samples=10, channels=1)
    ring.push(np.arange(4, dtype=np.float32))
    ring.push(np.arange(20, dtype=np.float32))
    assert ring.total_written == 24
    assert ring.overwritten_samples == 14


def test_does_not_skip_when_consumer_keeps_up() -> None:
    pipeline, runner = make_runner()
    for _ in range(100):
        runner.push(np.zeros(50, dtype=np.float32))
        runner.process_available()
    stats = runner.stats()
    assert stats["skipped_seconds"] == 0.0
    assert stats["processed_windows"] == 48
    assert stats["lag_seconds"] <= 0.1 + 1e-9
    assert all(length == 300 for length in pipeline.window_lengths)


def test_drop_oldest_bounds_lag_and_accounts_skips() -> None:
    _, runner = make_runner(policy="drop_oldest", max_lag=0.5)
    runner.push(np.zeros(5000, dtype=np.float32))
    runner.process_available()
    stats = runner.stats()
    assert stats["skipped_seconds"] >= 4.0
    assert stats["lag_seconds"] <= 0.1 + 1e-9
    assert stats["processed_windows"] == 5


def test_catch_up_recovers_more_windows_than_drop_oldest() -> None:
    _, drop_runner = make_runner(policy="drop_oldest", max_windows=2, max_lag=0.5)
    drop_runner.push(np.zeros(1000, dtype=np.float32))
    drop_runner.process_available()
    _, catch_runner = make_runner(policy="catch_up", max_windows=2, max_lag=0.5)
    catch_runner.push(np.zeros(1000, dtype=np.float32))
    catch_runner.process_available()
    drop_stats = drop_runner.stats()
    catch_stats = catch_runner.stats()
    assert catch_stats["skipped_seconds"] < drop_stats["skipped_seconds"]
    assert catch_stats["processed_windows"] > drop_stats["processed_windows"]


def test_state_continuity_and_window_context() -> None:
    pipeline, runner = make_runner(max_lag=5.0)
    runner.push(np.zeros(1000, dtype=np.float32))
    processed = runner.process_available()
    assert processed == 8
    assert pipeline.calls == 8
    assert runner.state is not None
    assert runner.state["windows"] == 8
    assert all(length == 300 for length in pipeline.window_lengths)


def test_second_process_call_after_more_audio() -> None:
    _, runner = make_runner(max_lag=5.0)
    runner.push(np.zeros(500, dtype=np.float32))
    runner.process_available()
    first = runner.stats()["processed_windows"]
    runner.push(np.zeros(500, dtype=np.float32))
    runner.process_available()
    second = runner.stats()["processed_windows"]
    assert second > first
    assert runner.stats()["skipped_seconds"] == 0.0
