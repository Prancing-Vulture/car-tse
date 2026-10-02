"""Manifest dataset roundtrip."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from car_tse.data.datasets import TripletExtractionDataset
from car_tse.io.manifest import write_jsonl


def test_triplet_dataset(tmp_path=None):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for name in ("mix", "enroll", "target"):
            sf.write(str(td / f"{name}.wav"), np.random.randn(8000).astype(np.float32), 8000)
        row = {
            "id": "t0",
            "mixture_wav": str(td / "mix.wav"),
            "enroll_wav": str(td / "enroll.wav"),
            "target_wav": str(td / "target.wav"),
            "sample_rate": 8000,
            "speaker_id": "spk1",
            "target_present": 1,
        }
        man = td / "m.jsonl"
        write_jsonl(man, [row])
        ds = TripletExtractionDataset(man, 8000, 1.0, training=False)
        item = ds[0]
        assert item["mixture"].shape[0] == 8000
