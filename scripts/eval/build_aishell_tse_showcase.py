#!/usr/bin/env python
"""Export one held-out AISHELL-5 Stage 1 input/output showcase."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.metrics.si_sdr import si_sdr
from car_tse.models.stage1_system import Stage1System
from car_tse.train.utils import get_device, load_checkpoint
from scripts.data.build_presence_embedding_cache import read_segment, render_row, resolve_path
from car_tse.io.manifest import read_jsonl


def listen_normalize(wave: np.ndarray, peak: float = 0.95) -> np.ndarray:
    wave = np.asarray(wave, dtype=np.float32)
    maximum = float(np.max(np.abs(wave))) if wave.size else 0.0
    return wave * (peak / maximum) if maximum > 1.0e-8 else wave


def write_float(path: Path, wave: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(wave, dtype=np.float32), sample_rate, subtype="FLOAT")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export an AISHELL-5 TSE example")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "manifests/presence_aishell_target/test/stage2_presence_aishell_target.jsonl",
    )
    parser.add_argument("--row-id", default=None)
    parser.add_argument(
        "--stage1-config", type=Path, default=REPO_ROOT / "configs/stage1_spex_transfer_mc_cavemove.yaml"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints/stage1_spex_transfer_mc_cavemove_cabin/best.pt",
    )
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "reports/artifacts/aishell5_example_baseline"
    )
    parser.add_argument(
        "--normalize-model-input",
        action="store_true",
        help="Peak-normalize mixture, enrollment, and multichannel input before Stage 1 inference",
    )
    args = parser.parse_args()
    rows = [row for row in read_jsonl(resolve_path(args.manifest)) if int(row.get("target_present", 0)) == 1]
    if args.row_id:
        rows = [row for row in rows if row.get("id") == args.row_id]
    if not rows:
        raise SystemExit("No matching target-present row found")
    row = rows[0]

    cfg = load_stage_config(args.stage1_config, REPO_ROOT / "configs/paths.yaml")
    device = get_device()
    sample_rate = int(cfg.get("sample_rate", 16000))
    window_samples = int(round(float(cfg.get("segment_seconds", 3.0)) * sample_rate))
    enrollment_samples = window_samples // 2
    pretrained = resolve_path(cfg["pretrained_checkpoint"])
    model = Stage1System(
        spex_cfg=cfg["spex"],
        sample_rate=sample_rate,
        use_cabin_frontend=True,
        num_mics=int(cfg.get("num_mics", 4)),
        frontend_cfg=cfg.get("frontend"),
        pretrained_dir=str(pretrained),
        backend=cfg.get("spex_backend", "transfer_spexplus_16k"),
    ).to(device)
    load_checkpoint(resolve_path(args.checkpoint), model)
    model.eval()

    mixture, enrollment, multichannel = render_row(
        row, 0, sample_rate, window_samples, enrollment_samples, {}
    )
    model_mixture = mixture
    model_enrollment = enrollment
    model_multichannel = multichannel
    if args.normalize_model_input:
        model_mixture = listen_normalize(mixture)
        model_enrollment = listen_normalize(enrollment)
        model_multichannel = listen_normalize(multichannel.T).T
    target = read_segment(
        resolve_path(row["target_wav"]),
        sample_rate,
        window_samples,
        int(round(float(row["segment_start_seconds"]) * sample_rate)),
    )[:, 0]
    with torch.inference_mode():
        extracted, _, _, _ = model(
            torch.from_numpy(model_mixture[None]).to(device),
            torch.from_numpy(model_enrollment[None]).to(device),
            multichannel=torch.from_numpy(model_multichannel[None]).to(device),
        )
    extracted_np = extracted[0].detach().cpu().numpy().astype(np.float32)
    input_score = float(si_sdr(torch.from_numpy(mixture[None]), torch.from_numpy(target[None])))
    output_score = float(si_sdr(torch.from_numpy(extracted_np[None]), torch.from_numpy(target[None])))

    output = resolve_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_float(output / "input_mean4ch_raw.wav", mixture, sample_rate)
    write_float(output / "input_mean4ch_listen.wav", listen_normalize(mixture), sample_rate)
    write_float(output / "input_4ch_raw.wav", multichannel.T, sample_rate)
    write_float(output / "input_4ch_listen.wav", listen_normalize(multichannel.T), sample_rate)
    write_float(output / "enrollment_raw.wav", enrollment, sample_rate)
    write_float(output / "enrollment_listen.wav", listen_normalize(enrollment), sample_rate)
    write_float(output / "target_reference_raw.wav", target, sample_rate)
    write_float(output / "target_reference_listen.wav", listen_normalize(target), sample_rate)
    write_float(output / "extracted_raw.wav", extracted_np, sample_rate)
    write_float(output / "extracted_listen.wav", listen_normalize(extracted_np), sample_rate)
    metadata = {
        "row_id": row.get("id"),
        "speaker_id": row.get("speaker_id"),
        "source_split": "AISHELL-5 Eval1 held-out target-present",
        "mixture_wav": row.get("mixture_wav"),
        "mixture_multich_wavs": row.get("mixture_multich_wavs"),
        "enrollment_wav": row.get("enroll_wav"),
        "target_reference_wav": row.get("target_wav"),
        "segment_start_seconds": row.get("segment_start_seconds"),
        "enrollment_start_seconds": row.get("enroll_start_seconds"),
        "transcript": row.get("meta", {}).get("text", ""),
        "checkpoint": str(resolve_path(args.checkpoint)),
        "device": str(device),
        "model_input_peak_normalized": bool(args.normalize_model_input),
        "sample_rate": sample_rate,
        "input_si_sdr_db": input_score,
        "extracted_si_sdr_db": output_score,
        "si_sdri_db": output_score - input_score,
        "note": "*_listen.wav files are independently peak-normalized for listening; metrics use raw float waveforms.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    # Keep the CLI usable on Windows consoles that still default to cp1252.
    print(json.dumps(metadata, ensure_ascii=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
