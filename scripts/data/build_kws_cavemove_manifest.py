#!/usr/bin/env python
"""Build leakage-free binary KWS manifests with CAVEMOVE cabin noise.

The local workspace does not yet contain custom in-car wake-phrase recordings,
so the Google Speech Commands ``marvin`` class is used as a temporary wake
word proxy.  All other Speech Commands words are non-wake speech.  Generic
Speech Commands background-noise recordings are intentionally excluded; real
CAVEMOVE driving/ventilation recordings provide the only environmental noise.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import resolve_paths
from car_tse.io.manifest import write_jsonl


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATHS = REPO_ROOT / "configs" / "paths.yaml"
DEFAULT_SNRS = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)


def cave_noise_files(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*.wav") if p.parent.name in {"noise", "ventilation"}
    ) if root.exists() else []


def car_name(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    return relative.parts[0] if relative.parts else "unknown_car"


def split_noise_files(files: list[Path], root: Path, test_car: str, seed: int) -> dict[str, list[Path]]:
    test = [p for p in files if car_name(p, root).lower() == test_car.lower()]
    train_val = [p for p in files if p not in test]
    rng = random.Random(seed)
    rng.shuffle(train_val)
    val_count = max(1, round(len(train_val) * 0.20)) if train_val else 0
    val = train_val[:val_count]
    train = train_val[val_count:] or val[:]
    val = val or train[:]
    test = test or files[-max(1, min(10, len(files))) :]
    return {"train": train, "val": val, "test": test}


def read_split_list(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip().replace("\\", "/") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def choose_noise(pool: list[Path], rng: random.Random) -> Path:
    if not pool:
        raise RuntimeError("CAVEMOVE noise pool is empty")
    return pool[rng.randrange(len(pool))]


def add_noise_fields(row: dict, noise: Path, noise_root: Path, snr: float) -> dict:
    row["cabin_noise_wav"] = str(noise)
    row["cabin_noise_snr_db"] = float(snr)
    row["meta"] = {
        "cabin_noise_dataset": "CAVEMOVE",
        "cabin_noise_type": noise.parent.name,
        "cabin_vehicle": car_name(noise, noise_root),
        "cabin_noise_snr_db": float(snr),
    }
    return row


def build_split(
    gsc_root: Path,
    noise_root: Path,
    split: str,
    wake_word: str,
    negative_ratio: int,
    noise_pool: list[Path],
    snrs: tuple[float, ...],
    seed: int,
    noise_only_repeats: int,
) -> list[dict]:
    validation = read_split_list(gsc_root / "validation_list.txt")
    testing = read_split_list(gsc_root / "testing_list.txt")
    candidates: list[tuple[Path, int]] = []
    for wav in sorted(gsc_root.rglob("*.wav")):
        relative = wav.relative_to(gsc_root).as_posix()
        if wav.parent.name == "_background_noise_":
            continue
        if split == "val" and relative not in validation:
            continue
        if split == "test" and relative not in testing:
            continue
        if split == "train" and (relative in validation or relative in testing):
            continue
        candidates.append((wav, int(wav.parent.name.lower() == wake_word.lower())))

    positives = [p for p, label in candidates if label == 1]
    negatives = [p for p, label in candidates if label == 0]
    rng = random.Random(seed + {"train": 0, "val": 1, "test": 2}[split])
    rng.shuffle(negatives)
    if negative_ratio > 0:
        negatives = negatives[: min(len(negatives), max(1, len(positives) * negative_ratio))]
    selected = [(p, 1) for p in positives] + [(p, 0) for p in negatives]
    rng.shuffle(selected)

    rows: list[dict] = []
    for index, (wav, label) in enumerate(selected):
        noise = choose_noise(noise_pool, rng)
        row = {
            "wav": str(wav),
            "label": int(label),
            "split": split,
            "wake_word": wake_word,
            "noise_only": False,
        }
        rows.append(add_noise_fields(row, noise, noise_root, snrs[index % len(snrs)]))

    # Explicit cabin-only negatives help control false accepts when the
    # vehicle contains no speech.  They use the CAVEMOVE recording directly;
    # no generic background-noise corpus is introduced.
    repeats = noise_only_repeats if split == "train" else 1
    for index, noise in enumerate(noise_pool * max(1, repeats)):
        rows.append(
            {
                "wav": str(noise),
                "label": 0,
                "split": split,
                "wake_word": wake_word,
                "noise_only": True,
                "meta": {
                    "cabin_noise_dataset": "CAVEMOVE",
                    "cabin_noise_type": noise.parent.name,
                    "cabin_vehicle": car_name(noise, noise_root),
                },
            }
        )
    rng.shuffle(rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=DEFAULT_PATHS)
    parser.add_argument("--wake-word", default="marvin")
    parser.add_argument("--test-car", default="Volkswagen_Golf")
    parser.add_argument("--negative-ratio", type=int, default=4)
    parser.add_argument("--noise-only-repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--snr-db", nargs="+", type=float, default=list(DEFAULT_SNRS))
    args = parser.parse_args()

    paths = resolve_paths(args.paths)
    manifest_root = Path(paths["manifest_root"])
    gsc_root = Path(paths["corpus_paths"]["speech_commands_v2"])
    noise_root = Path(paths["corpus_paths"]["cavemove"])
    noise_files = cave_noise_files(noise_root)
    if not gsc_root.exists() or not noise_files:
        print(f"Missing GSC ({gsc_root}) or CAVEMOVE cabin noise ({noise_root})")
        return 1
    pools = split_noise_files(noise_files, noise_root, args.test_car, args.seed)
    snrs = tuple(float(x) for x in args.snr_db)
    if not snrs:
        raise ValueError("At least one SNR is required")

    output_names = {
        "train": "stage3_kws_cabin_train.jsonl",
        "val": "stage3_kws_cabin_val.jsonl",
        "test": "stage3_kws_cabin_test.jsonl",
    }
    for split in ("train", "val", "test"):
        rows = build_split(
            gsc_root,
            noise_root,
            split,
            args.wake_word,
            args.negative_ratio,
            pools[split],
            snrs,
            args.seed,
            args.noise_only_repeats,
        )
        output = manifest_root / ("train" if split == "train" else "val" if split == "val" else "test") / output_names[split]
        write_jsonl(output, rows)
        positives = sum(int(row["label"] == 1) for row in rows)
        noise_only = sum(int(row.get("noise_only", False)) for row in rows)
        print(f"{split}: rows={len(rows)} positives={positives} noise_only={noise_only} -> {output}")
    print(f"wake_word_proxy={args.wake_word}; CAVEMOVE files={len(noise_files)}; test_vehicle={args.test_car}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
