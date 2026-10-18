#!/usr/bin/env python
"""Cache fixed-length enrollment waves for compact verifier experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from car_tse.data.segment import segment_waveform  # noqa: E402
from car_tse.io.audio import read_mono  # noqa: E402


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def source_index(rows: list[dict]) -> dict[str, dict]:
    manifests: dict[str, dict[str, dict]] = {}
    index: dict[str, dict] = {}
    for row in rows:
        manifest = str(row["source_manifest"])
        if manifest not in manifests:
            path = Path(manifest)
            if not path.is_absolute():
                path = REPO_ROOT / path
            manifests[manifest] = {
                str(item.get("id", "")): item
                for item in (
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            }
        source_id = str(row["source_id"])
        if source_id not in manifests[manifest]:
            raise KeyError(f"Missing source row {source_id} in {manifest}")
        index[source_id] = manifests[manifest][source_id]
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-root", type=Path, default=REPO_ROOT / "manifests"
    )
    parser.add_argument(
        "--manifest-template",
        default="stage3_target_kws_gsc_marvin_ft2_stage1_{split}.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data/cache/compact_speaker_verifier_gsc_marvin_2026-10-11",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--seconds", type=float, default=1.0)
    args = parser.parse_args()

    manifest_root = resolve(args.manifest_root)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    num_samples = max(1, int(round(float(args.sample_rate) * float(args.seconds))))
    for split in ("train", "val", "test"):
        manifest_path = manifest_root / split / str(args.manifest_template).format(split=split)
        rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        sources = source_index(rows)
        waves = np.zeros((len(rows), num_samples), dtype=np.float32)
        for index, row in enumerate(rows):
            source = sources[str(row["source_id"])]
            wav, _ = read_mono(Path(source["enroll_wav"]), int(args.sample_rate))
            waves[index] = segment_waveform(wav, num_samples, training=False)
        np.save(output_dir / f"{split}_enrollment_wav.npy", waves)
        print(f"{split}: cached {len(rows)} enrollment waves -> {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
