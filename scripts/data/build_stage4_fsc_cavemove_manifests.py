#!/usr/bin/env python
"""Attach disjoint CAVEMOVE cabin recordings to official FSC train/valid rows.

Only the official Fluent Speech Commands train and valid manifests are read.
The official test manifest is intentionally never opened. Validation uses all
noise recordings from one held-out CAVEMOVE vehicle; training uses recordings
from the other vehicles. Original FSC row fields and labels are preserved.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import REPO_ROOT, resolve_paths
from car_tse.io.manifest import read_jsonl, write_jsonl


DEFAULT_PATHS = REPO_ROOT / "configs" / "paths.e_research.yaml"
DEFAULT_SNRS = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
CHANNELS = (0, 2, 4, 6)
OUTPUT_NAMES = {
    "train": "stage4_fsc_cavemove_train.jsonl",
    "val": "stage4_fsc_cavemove_valid.jsonl",
}
SPLIT_SEEDS = {"train": 0, "val": 1}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vehicle_name(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    if not relative.parts:
        raise ValueError(f"Noise recording is outside CAVEMOVE root: {path}")
    return relative.parts[0]


def vehicle_noise_files(root: Path) -> list[Path]:
    """Use only CAVEMOVE recorded noise/ventilation WAVs, never IRs."""
    if not root.is_dir():
        raise FileNotFoundError(f"CAVEMOVE data root does not exist: {root}")
    return sorted(
        path
        for path in root.rglob("*.wav")
        if path.parent.name in {"noise", "ventilation"}
    )


def assign_rows(
    rows: list[dict[str, Any]],
    pool: list[Path],
    noise_root: Path,
    split: str,
    source_manifest: Path,
    snrs: tuple[float, ...],
    seed: int,
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError(f"Source manifest is empty: {source_manifest}")
    if not pool:
        raise ValueError(f"No CAVEMOVE noise recordings assigned to {split}")
    rng = random.Random(seed + SPLIT_SEEDS[split])
    result: list[dict[str, Any]] = []
    for index, original in enumerate(rows):
        row = copy.deepcopy(original)
        if "wav" not in row or "label" not in row:
            raise ValueError(f"Missing wav/label in {source_manifest}, row {index}")
        if not Path(row["wav"]).is_file():
            raise FileNotFoundError(f"Missing FSC WAV in {source_manifest}, row {index}: {row['wav']}")
        noise_path = pool[rng.randrange(len(pool))]
        snr = float(snrs[index % len(snrs)])
        row.update(
            {
                "condition": "fsc_plus_cavemove_cabin_noise",
                "source_manifest": str(source_manifest),
                "source_split": split,
                "cabin_noise_wav": str(noise_path),
                "cabin_noise_channels": list(CHANNELS),
                "cabin_noise_snr_db": snr,
            }
        )
        meta = dict(row.get("meta") or {})
        meta.update(
            {
                "intent_data_source": "Fluent Speech Commands official split",
                "intent_source_split": str(row.get("official_split", split)),
                "cabin_noise_dataset": "CAVEMOVE",
                "cabin_noise_vehicle": vehicle_name(noise_path, noise_root),
                "cabin_noise_type": noise_path.parent.name,
                "cabin_noise_snr_db": snr,
                "cabin_noise_channels": list(CHANNELS),
            }
        )
        row["meta"] = meta
        result.append(row)
    return result


def speaker_ids(rows: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for row in rows:
        speaker = row.get("speaker_id")
        if speaker is None:
            raise ValueError("Official FSC manifests must include speaker_id for split validation")
        ids.add(str(speaker))
    return ids


def stats(rows: list[dict[str, Any]], noise_root: Path) -> dict[str, Any]:
    labels = Counter(int(row["label"]) for row in rows)
    vehicles = Counter(vehicle_name(Path(row["cabin_noise_wav"]), noise_root) for row in rows)
    noise_types = Counter(Path(row["cabin_noise_wav"]).parent.name for row in rows)
    return {
        "rows": len(rows),
        "classes": len(labels),
        "label_counts": {str(key): labels[key] for key in sorted(labels)},
        "speakers": len(speaker_ids(rows)),
        "noise_files": len({row["cabin_noise_wav"] for row in rows}),
        "noise_vehicle_row_counts": dict(sorted(vehicles.items())),
        "noise_type_row_counts": dict(sorted(noise_types.items())),
        "snr_db_counts": dict(sorted(Counter(float(row["cabin_noise_snr_db"]) for row in rows).items())),
        "noise_channels": list(CHANNELS),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=DEFAULT_PATHS)
    parser.add_argument("--source-train-manifest", default="stage4_intent_fsc_train.jsonl")
    parser.add_argument("--source-val-manifest", default="stage4_intent_fsc_valid.jsonl")
    parser.add_argument("--validation-vehicle", default="Smart_forfour")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--snr-db", nargs="+", type=float, default=list(DEFAULT_SNRS))
    parser.add_argument("--dry-run", action="store_true", help="Validate assignments and print stats without writing manifests")
    args = parser.parse_args()

    if not args.snr_db:
        raise ValueError("At least one SNR must be provided")
    paths = resolve_paths(args.paths)
    manifest_root = Path(paths["manifest_root"])
    noise_root = Path(paths["corpus_paths"]["cavemove"])
    all_noise = vehicle_noise_files(noise_root)
    vehicles = sorted({vehicle_name(path, noise_root) for path in all_noise})
    val_vehicle = next((name for name in vehicles if name.lower() == args.validation_vehicle.lower()), None)
    if val_vehicle is None:
        raise ValueError(f"Validation vehicle {args.validation_vehicle!r} not found; available: {vehicles}")
    val_noise = [path for path in all_noise if vehicle_name(path, noise_root) == val_vehicle]
    train_noise = [path for path in all_noise if vehicle_name(path, noise_root) != val_vehicle]
    if not train_noise or not val_noise:
        raise ValueError("Need non-empty and vehicle-disjoint train and validation noise pools")

    source_paths = {
        "train": manifest_root / "official" / args.source_train_manifest,
        "val": manifest_root / "official" / args.source_val_manifest,
    }
    if any(not path.is_file() for path in source_paths.values()):
        raise FileNotFoundError(f"Official FSC train/valid source manifest missing: {source_paths}")
    clean_rows = {split: read_jsonl(path) for split, path in source_paths.items()}
    train_speakers = speaker_ids(clean_rows["train"])
    val_speakers = speaker_ids(clean_rows["val"])
    overlap = train_speakers & val_speakers
    if overlap:
        raise ValueError(f"FSC train/valid speaker leakage ({len(overlap)} speakers): {sorted(overlap)[:8]}")

    pools = {"train": train_noise, "val": val_noise}
    output_rows = {
        split: assign_rows(
            clean_rows[split], pools[split], noise_root, split, source_paths[split],
            tuple(float(value) for value in args.snr_db), args.seed,
        )
        for split in ("train", "val")
    }
    train_vehicles = {vehicle_name(Path(row["cabin_noise_wav"]), noise_root) for row in output_rows["train"]}
    val_vehicles = {vehicle_name(Path(row["cabin_noise_wav"]), noise_root) for row in output_rows["val"]}
    train_noise_paths = {row["cabin_noise_wav"] for row in output_rows["train"]}
    val_noise_paths = {row["cabin_noise_wav"] for row in output_rows["val"]}
    if train_vehicles & val_vehicles or train_noise_paths & val_noise_paths:
        raise RuntimeError("Generated train/validation noise groups overlap")
    for split, rows in output_rows.items():
        source_labels = Counter(int(row["label"]) for row in clean_rows[split])
        output_labels = Counter(int(row["label"]) for row in rows)
        if source_labels != output_labels:
            raise RuntimeError(f"Label counts changed for {split}")
        if any(not Path(row["cabin_noise_wav"]).is_file() for row in rows):
            raise RuntimeError(f"Missing CAVEMOVE noise WAV in generated {split} rows")
        if any(row.get("cabin_noise_channels") != list(CHANNELS) or "cabin_noise_snr_db" not in row for row in rows):
            raise RuntimeError(f"Generated {split} rows do not satisfy the cache builder input contract")

    output_paths: dict[str, Path] = {}
    for split, rows in output_rows.items():
        subdir = "train" if split == "train" else "val"
        output = manifest_root / subdir / OUTPUT_NAMES[split]
        if not args.dry_run:
            write_jsonl(output, rows)
        output_paths[split] = output

    provenance = {
        "recipe": "official_fsc_plus_cavemove_stage4_source_manifests",
        "seed": args.seed,
        "official_fsc_test_used": False,
        "source_manifests": {
            split: {"path": str(path), "sha256": sha256_file(path)}
            for split, path in source_paths.items()
        },
        "output_manifests": {split: str(path) for split, path in output_paths.items()},
        "cavemove_root": str(noise_root),
        "cavemove_noise_files_total": len(all_noise),
        "validation_vehicle_held_out_from_train": val_vehicle,
        "train_noise_vehicle_pool": sorted({vehicle_name(path, noise_root) for path in train_noise}),
        "validation_noise_vehicle_pool": sorted({vehicle_name(path, noise_root) for path in val_noise}),
        "channels": list(CHANNELS),
        "snr_db_schedule": [float(value) for value in args.snr_db],
        "speaker_overlap_count": len(train_speakers & val_speakers),
        "noise_vehicle_overlap_count": len(train_vehicles & val_vehicles),
        "noise_file_overlap_count": len(train_noise_paths & val_noise_paths),
        "splits": {split: stats(rows, noise_root) for split, rows in output_rows.items()},
    }
    provenance_path = manifest_root / "meta" / "stage4_fsc_cavemove_manifest_provenance.json"
    if not args.dry_run:
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2))
    print(f"provenance={provenance_path}" + (" (not written; dry run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
