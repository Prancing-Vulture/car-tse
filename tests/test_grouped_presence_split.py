from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train.train_stage2_frame_presence import split_indices


def test_grouped_presence_split_keeps_enrollment_groups_disjoint():
    rows = [
        {"group_id": "driver-a", "target_present": 1},
        {"group_id": "driver-a", "target_present": 1},
        {"group_id": "driver-b", "target_present": 0},
        {"group_id": "driver-b", "target_present": 0},
        {"group_id": "driver-c", "target_present": 1},
        {"group_id": "driver-d", "target_present": 0},
    ]
    train, validation = split_indices(rows, val_fraction=0.33, seed=7, group_by_speaker=True)
    train_groups = {rows[index]["group_id"] for index in train}
    validation_groups = {rows[index]["group_id"] for index in validation}

    assert train
    assert validation
    assert train_groups.isdisjoint(validation_groups)
