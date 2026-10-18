#!/usr/bin/env python
"""Add matched ANIR cabin-noise copies to a presence-training manifest.

The base manifest remains unchanged.  Only rows with a clean target source are
duplicated, so the generated manifest can be compared against the existing
CAVEMOVE/AISHELL presence candidate with the same labels and enrollments.
"""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--noise-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copies-per-positive", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument(
        "--snr-db",
        nargs="+",
        type=float,
        default=[-5.0, 0.0, 5.0, 10.0, 15.0, 20.0],
    )
    args = parser.parse_args()
    base_path = args.base_manifest.resolve()
    noise_path = args.noise_manifest.resolve()
    output_path = args.output.resolve()
    base_rows = read_jsonl(base_path)
    noise_manifest = json.loads(noise_path.read_text(encoding="utf-8"))
    noise_rows = list(noise_manifest.get("files", []))
    if not base_rows or not noise_rows:
        raise RuntimeError("Base manifest and ANIR manifest must both contain rows")
    rng = random.Random(int(args.seed))
    rng.shuffle(noise_rows)
    positives = [row for row in base_rows if int(row.get("target_present", 0)) == 1]
    augmented: list[dict] = []
    copies = max(0, int(args.copies_per_positive))
    snrs = [float(value) for value in args.snr_db]
    for positive_index, row in enumerate(positives):
        for copy_index in range(copies):
            noise = noise_rows[(positive_index * copies + copy_index) % len(noise_rows)]
            clone = deepcopy(row)
            clone["id"] = f"{row.get('id', positive_index)}_anir_{copy_index}"
            clone["condition"] = "gsc_marvin_plus_anir_noise"
            clone["cabin_noise_wav"] = str(noise["path"])
            clone["cabin_noise_channels"] = [0, 1, 2, 3]
            clone["cabin_noise_snr_db"] = snrs[(positive_index + copy_index) % len(snrs)]
            meta = dict(clone.get("meta", {}))
            meta.update(
                {
                    "source": "speech_commands_v2_plus_anir_v_class",
                    "anir_condition": noise.get("condition"),
                    "anir_microphones": noise.get("microphones"),
                    "augmentation_only": True,
                }
            )
            clone["meta"] = meta
            augmented.append(clone)
    output_rows = base_rows + augmented
    rng.shuffle(output_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "base_rows": len(base_rows),
                "base_positive_rows": len(positives),
                "augmented_rows": len(augmented),
                "output_rows": len(output_rows),
                "output": str(output_path),
                "not_evaluation_evidence": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
