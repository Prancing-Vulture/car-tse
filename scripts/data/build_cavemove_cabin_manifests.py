#!/usr/bin/env python
"""Attach real CAVEMOVE cabin-noise recordings to LibriMix triplets.

The generated manifests keep the clean LibriMix mixture and target paths, but
add a ``cabin_noise_wav`` field.  The dataset class mixes that real vehicle
recording into the multichannel input on the fly at a selected SNR.  Only
CAVEMOVE ``noise`` and ``ventilation`` recordings are admitted; AISHELL-5,
VISC, Gaussian noise, synthetic delays, and synthetic channel gains are not
used by this recipe.
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import resolve_paths
from car_tse.io.manifest import read_jsonl, write_jsonl


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATHS = REPO_ROOT / "configs" / "paths.yaml"
DEFAULT_SNRS = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)


def vehicle_noise_files(root: Path) -> list[Path]:
    """Return only CAVEMOVE driving/ventilation recordings."""

    files: list[Path] = []
    if not root.exists():
        return files
    for path in root.rglob("*.wav"):
        # Parent names are the dataset's semantic categories.  This excludes
        # IRs and radio IRs even though they are also WAV files.
        if path.parent.name in {"noise", "ventilation"}:
            files.append(path)
    return sorted(files)


def car_name(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    return relative.parts[0] if relative.parts else "unknown_car"


def split_noise_files(
    files: list[Path],
    root: Path,
    test_car: str,
    val_ratio: float,
    seed: int,
) -> dict[str, list[Path]]:
    """Hold out one vehicle for test and disjoint files for validation."""

    test = [p for p in files if car_name(p, root).lower() == test_car.lower()]
    train_val = [p for p in files if p not in test]
    rng = random.Random(seed)
    rng.shuffle(train_val)
    val_count = max(1, int(round(len(train_val) * val_ratio))) if train_val else 0
    val = train_val[:val_count]
    train = train_val[val_count:]
    if not train:
        train = val[:]
    if not val:
        val = train[:]
    if not test:
        # Keep the command usable if a future CAVEMOVE release changes car
        # names; the fallback still keeps train/val file-disjoint.
        test = files[-max(1, min(10, len(files))) :]
    return {"train": train, "val": val, "test": test}


def build_rows(
    clean_rows: list[dict],
    noise_files: list[Path],
    noise_root: Path,
    split: str,
    channels: list[int],
    snrs: tuple[float, ...],
    seed: int,
    max_rows: int,
) -> list[dict]:
    if not noise_files:
        raise RuntimeError(f"No CAVEMOVE cabin noise files available for {split}")
    rng = random.Random(seed + {"train": 0, "val": 1, "test": 2}[split])
    rows: list[dict] = []
    selected = clean_rows[:max_rows] if max_rows else clean_rows
    for index, original in enumerate(selected):
        row = copy.deepcopy(original)
        row.pop("mixture_multich_wav", None)
        row.pop("mixture_multich_wavs", None)
        noise_path = noise_files[rng.randrange(len(noise_files))]
        noise_type = noise_path.parent.name
        row["condition"] = "cavemove_cabin_noise"
        row["cabin_noise_wav"] = str(noise_path)
        row["cabin_noise_channels"] = [int(x) for x in channels]
        row["cabin_noise_snr_db"] = float(snrs[index % len(snrs)])
        meta = dict(row.get("meta") or {})
        meta.pop("aishell5_noise", None)
        meta.pop("multich_augmentation", None)
        meta.update(
            {
                "source": "librimix_plus_cavemove",
                "cabin_noise_dataset": "CAVEMOVE",
                "cabin_noise_type": noise_type,
                "cabin_vehicle": car_name(noise_path, noise_root),
                "cabin_mic_setup": noise_path.parent.parent.name,
                "cabin_noise_snr_db": float(row["cabin_noise_snr_db"]),
                "cabin_noise_channels": [int(x) for x in channels],
            }
        )
        row["meta"] = meta
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=DEFAULT_PATHS)
    parser.add_argument("--input-manifest", default="stage1_triplets.jsonl")
    parser.add_argument("--output-manifest", default="stage1_cavemove_cabin.jsonl")
    parser.add_argument("--test-car", default="Volkswagen_Golf")
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--channels", nargs="+", type=int, default=[0, 2, 4, 6])
    parser.add_argument("--snr-db", nargs="+", type=float, default=list(DEFAULT_SNRS))
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val", "test"])
    args = parser.parse_args()

    paths = resolve_paths(args.paths)
    manifest_root = Path(paths["manifest_root"])
    noise_root = Path(paths["corpus_paths"]["cavemove"])
    noise_files = vehicle_noise_files(noise_root)
    if not noise_files:
        print(f"No CAVEMOVE noise/ventilation WAVs under {noise_root}")
        return 1
    pools = split_noise_files(
        noise_files,
        noise_root,
        args.test_car,
        args.val_ratio,
        args.seed,
    )
    snrs = tuple(float(x) for x in args.snr_db)
    if not snrs:
        raise ValueError("At least one SNR value is required")

    print(f"CAVEMOVE cabin files: {len(noise_files)}")
    for split in args.splits:
        input_path = manifest_root / split / args.input_manifest
        clean_rows = read_jsonl(input_path)
        if not clean_rows:
            print(f"No clean rows at {input_path}")
            return 1
        rows = build_rows(
            clean_rows,
            pools[split],
            noise_root,
            split,
            args.channels,
            snrs,
            args.seed,
            args.max_rows,
        )
        output_path = manifest_root / split / args.output_manifest
        write_jsonl(output_path, rows)
        vehicle_counts = sorted({row["meta"]["cabin_vehicle"] for row in rows})
        noise_types = sorted({row["meta"]["cabin_noise_type"] for row in rows})
        print(
            f"{split}: {len(rows)} rows -> {output_path} | "
            f"noise files={len(pools[split])} vehicles={vehicle_counts} types={noise_types}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
