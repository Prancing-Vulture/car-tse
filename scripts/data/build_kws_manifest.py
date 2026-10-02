#!/usr/bin/env python
"""Google Speech Commands v2 + MSWC negatives + Hey Snips eval lists."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from car_tse.config import REPO_ROOT, resolve_paths
from car_tse.data.prep_common import manifest_dir, prep_argparser
from car_tse.io.manifest import write_jsonl

GSC_LABELS = [
    "_silence_",
    "_unknown_",
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
]


def scan_gsc(root: Path, label_map: dict[str, int]) -> list[dict]:
    rows = []
    if not root.exists():
        return rows
    for wav in root.rglob("*.wav"):
        label_name = wav.parent.name
        if label_name not in label_map:
            continue
        rows.append({"wav": str(wav), "label": label_map[label_name], "split": "train"})
    return rows


def scan_folder(root: Path, label: int) -> list[dict]:
    if not root.exists():
        return []
    return [{"wav": str(p), "label": label, "split": "eval"} for p in root.rglob("*.wav")]


def main() -> int:
    parser = prep_argparser("Build KWS manifests")
    args = parser.parse_args()
    paths = resolve_paths(args.paths)
    gsc_root = paths["corpus_paths"]["speech_commands_v2"]
    mswc = paths["corpus_paths"]["mswc_en"]
    hey = paths["corpus_paths"]["hey_snips"]
    label_map = {name: i for i, name in enumerate(GSC_LABELS)}
    label_map_path = Path(paths["manifest_root"]) / "meta" / "kws_label_map.json"
    label_map_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)
    train_rows = scan_gsc(gsc_root, label_map)
    train_rows += [{"wav": str(p), "label": 1, "split": "train"} for p in (paths["corpus_paths"]["custom_incar"] / "wake").rglob("*.wav") if (paths["corpus_paths"]["custom_incar"] / "wake").exists()]
    if mswc.exists():
        train_rows += [{"wav": str(p), "label": 1, "split": "train"} for p in list(mswc.rglob("*.wav"))[:5000]]
    eval_rows = scan_folder(hey, label=0)
    write_jsonl(manifest_dir(paths, "train") / "stage3_kws_train.jsonl", train_rows)
    write_jsonl(manifest_dir(paths, "test") / "stage3_kws_eval.jsonl", eval_rows)
    print(f"KWS train={len(train_rows)} eval={len(eval_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
