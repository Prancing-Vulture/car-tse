#!/usr/bin/env python
"""Convert user vehicle recordings into test manifest JSONL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import resolve_paths
from car_tse.io.manifest import write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest vehicle test recordings")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--enroll", type=Path, required=True)
    parser.add_argument("--paths", type=Path, default=None)
    args = parser.parse_args()
    paths = resolve_paths(args.paths)
    rows = []
    for i, wav in enumerate(sorted(args.input_dir.rglob("*.wav"))):
        rows.append(
            {
                "id": f"vehicle_{i}",
                "mixture_wav": str(wav),
                "enroll_wav": str(args.enroll),
                "target_wav": str(wav),
                "sample_rate": paths.get("sample_rate", 8000),
                "speaker_id": "enrolled",
                "seat": "driver",
                "condition": "vehicle",
                "target_present": 1,
                "transcript": "",
                "intent": "",
                "playback_ref_wav": "",
                "meta": {"source": "vehicle"},
            }
        )
    out = Path(paths["manifest_root"]) / "test" / "vehicle_test.jsonl"
    write_jsonl(out, rows)
    print(f"Wrote {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
