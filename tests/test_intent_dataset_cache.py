from __future__ import annotations

import json

import numpy as np

from car_tse.data.datasets import IntentDataset


def test_intent_dataset_reuses_longer_waveform_cache_for_short_window(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"wav": str(tmp_path / "missing.wav"), "label": 0}) + "\n")
    cache = tmp_path / "cache.npy"
    np.save(cache, np.arange(8, dtype=np.float32)[None, :])

    dataset = IntentDataset(
        manifest,
        sample_rate=8,
        segment_seconds=0.5,
        training=False,
        waveform_cache_path=cache,
    )

    item = dataset[0]
    # Validation crop is centered: the 4-sample request selects indices 2:6.
    assert item["wav"].shape == (4,)
    assert torch_equal_numpy(item["wav"].numpy(), np.arange(2, 6, dtype=np.float32))


def torch_equal_numpy(actual: np.ndarray, expected: np.ndarray) -> bool:
    return bool(np.array_equal(actual, expected))
