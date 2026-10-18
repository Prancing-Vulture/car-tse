from __future__ import annotations

import json
from pathlib import Path

from scripts.data.build_clean_presence_manifest import build_rows, write_rows


def _write_source(path: Path, root: Path) -> None:
    rows = []
    for speaker in ("a", "b", "c"):
        for index in range(2):
            wav = root / speaker / f"{index}.wav"
            wav.parent.mkdir(parents=True, exist_ok=True)
            wav.write_bytes(b"wav")
            rows.append({"wav": str(wav), "speaker_id": speaker})
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_clean_presence_manifest_is_balanced_and_speaker_conditioned(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    _write_source(source, tmp_path / "audio")

    rows = build_rows(source, max_rows=6, seed=7)

    assert len(rows) == 6
    assert sum(int(row["target_present"]) for row in rows) == 3
    assert sum(int(not row["target_present"]) for row in rows) == 3
    for row in rows:
        if row["target_present"]:
            assert row["speaker_id"] == row["source_speaker_id"]
        else:
            assert row["speaker_id"] != row["source_speaker_id"]
        assert len(row["mixture_multich_wavs"]) == 4


def test_clean_presence_manifest_writes_jsonl(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    _write_source(source, tmp_path / "audio")
    output = tmp_path / "out.jsonl"
    write_rows(output, build_rows(source, max_rows=4, seed=1))

    assert len(output.read_text(encoding="utf-8").splitlines()) == 4
