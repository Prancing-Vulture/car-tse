#!/usr/bin/env python
"""Build a deterministic mixed-domain Stage-1 manifest.

The clean/dual-domain rows and the target-plus-passenger rows already point
to materialized audio. This utility only combines and shuffles their JSONL
metadata, preserving the source domains for later audit.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def read(path: Path, domain: str, limit: int | None = None) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row = dict(row)
        row["domain_mix_source"] = domain
        row["id"] = f"{domain}__{str(row.get('id', len(rows)))}"
        rows.append(row)
        if limit is not None and len(rows) >= int(limit):
            break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--passenger-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--clean-limit",
        type=int,
        default=None,
        help="Optional deterministic prefix limit for the clean source.",
    )
    parser.add_argument(
        "--passenger-limit",
        type=int,
        default=None,
        help="Optional deterministic prefix limit for the passenger source.",
    )
    parser.add_argument("--seed", type=int, default=20261018)
    args = parser.parse_args()
    rows = read(args.clean_manifest, "clean_dual_domain", args.clean_limit) + read(
        args.passenger_manifest, "fsc_passenger", args.passenger_limit
    )
    random.Random(args.seed).shuffle(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
