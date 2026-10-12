#!/usr/bin/env python
"""Export the embedded SpEx+ weights from a complete Stage-1 checkpoint.

This lets transfer experiments initialize from the exact Stage-1 model without
depending on a machine-specific upstream checkpoint path that may no longer
exist. The original checkpoint is read-only; the small derived payload records
its provenance and is written to the caller-selected location.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload.get("model"))
    if not isinstance(state, dict):
        raise ValueError("Input checkpoint does not contain a model state dictionary")
    prefix = "spex.model."
    extracted = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if not extracted:
        raise ValueError(f"No SpEx+ tensors found under the expected {prefix!r} prefix")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": extracted,
            "provenance": {
                "source_checkpoint": str(args.input.resolve()),
                "source_epoch": payload.get("epoch"),
                "selection_rule": f"Copied tensors with prefix {prefix!r}; no reinitialization or training",
            },
        },
        args.output,
    )
    print(f"Exported {len(extracted)} tensors to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
