from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "data"))

from build_stage1_gsc_marvin_mixtures import (  # noqa: E402
    partition_noise_paths,
    target_onset_for_copy,
    vehicle_key,
)
from build_gsc_marvin_tse_kws_cache import kws_label_for_source  # noqa: E402


def test_noise_partitions_have_distinct_files_and_held_out_vehicle():
    paths = [
        Path("E:/cavemove/AlfaRomeo_146/noise/a0.wav"),
        Path("E:/cavemove/AlfaRomeo_146/noise/a1.wav"),
        Path("E:/cavemove/AlfaRomeo_146/noise/a2.wav"),
        Path("E:/cavemove/Smart_forfour/noise/s0.wav"),
        Path("E:/cavemove/Smart_forfour/noise/s1.wav"),
        Path("E:/cavemove/Volkswagen_Golf/noise/v0.wav"),
        Path("E:/cavemove/Volkswagen_Golf/noise/v1.wav"),
    ]
    pools = partition_noise_paths(paths)

    assert set(pools["train"]).isdisjoint(pools["val"])
    assert set(pools["train"]).isdisjoint(pools["test"])
    assert set(pools["val"]).isdisjoint(pools["test"])
    assert {vehicle_key(path) for path in pools["test"]} == {"Volkswagen_Golf"}
    assert {vehicle_key(path) for path in pools["train"]} == {"AlfaRomeo_146", "Smart_forfour"}
    assert {vehicle_key(path) for path in pools["val"]} == {"AlfaRomeo_146", "Smart_forfour"}


def test_training_wake_onsets_are_inside_runtime_first_second_crop():
    assert target_onset_for_copy(0) == 0
    assert target_onset_for_copy(1) == 3200
    assert target_onset_for_copy(2) == 0


def test_kws_label_is_not_confused_with_target_presence_label():
    assert kws_label_for_source({"target_present": 1, "kws_label": 0}) == 0
    assert kws_label_for_source({"target_present": 1, "kws_label": 1}) == 1
    assert kws_label_for_source({"target_present": 0, "kws_label": 0}) == 0
    assert kws_label_for_source({"target_present": 0}) == 0
