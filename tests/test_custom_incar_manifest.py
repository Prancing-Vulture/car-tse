from __future__ import annotations

import json
from pathlib import Path

from scripts.data.build_custom_incar_manifest import main


def test_custom_incar_manifest_is_speaker_disjoint(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "custom_incar"
    for speaker in ("driver_a", "driver_b", "driver_c", "driver_d"):
        for category in ("wake", "negative"):
            folder = root / category / speaker
            folder.mkdir(parents=True)
            (folder / f"{speaker}_{category}.wav").write_bytes(b"RIFF")
        enrollment = root / "enrollment" / speaker
        enrollment.mkdir(parents=True)
        (enrollment / "reference.wav").write_bytes(b"RIFF")

    output = tmp_path / "manifests"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_custom_incar_manifest.py",
            "--root",
            str(root),
            "--output-root",
            str(output),
            "--seed",
            "7",
        ],
    )
    assert main() == 0

    rows_by_split: dict[str, list[dict]] = {}
    for split in ("train", "val", "test"):
        path = output / split / f"stage3_kws_custom_incar_{split}.jsonl"
        rows_by_split[split] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    speakers_by_split = {
        split: {row["speaker_id"] for row in rows} for split, rows in rows_by_split.items()
    }
    assert speakers_by_split["train"].isdisjoint(speakers_by_split["val"])
    assert speakers_by_split["train"].isdisjoint(speakers_by_split["test"])
    assert speakers_by_split["val"].isdisjoint(speakers_by_split["test"])
    assert all(row["enroll_wav"] for rows in rows_by_split.values() for row in rows)
    assert sorted(row["label"] for row in rows_by_split["train"]) == [0, 0, 1, 1]
