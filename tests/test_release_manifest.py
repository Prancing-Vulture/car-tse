from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.reporting.build_release_manifest import file_record


def test_file_record_hashes_content(tmp_path: Path):
    path = tmp_path / "artifact.bin"
    payload = b"release-manifest-test"
    path.write_bytes(payload)

    record = file_record(path)

    assert record["exists"] is True
    assert record["bytes"] == len(payload)
    assert record["sha256"] == hashlib.sha256(payload).hexdigest()


def test_file_record_marks_missing_artifact(tmp_path: Path):
    record = file_record(tmp_path / "missing.bin")

    assert record == {
        "path": str(tmp_path / "missing.bin"),
        "exists": False,
        "bytes": 0,
        "sha256": None,
    }
