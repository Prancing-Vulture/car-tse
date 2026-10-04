#!/usr/bin/env python
"""Evaluate Stage 1 under bounded waveform-scale conditioning variants.

This is an inference-only diagnostic for the AISHELL-5 amplitude/frontend
mismatch.  It changes the multichannel waveform (and, for joint modes, the
enrollment waveform) immediately before the Stage 1 model.  It does not
modify checkpoints or train anything.  The same sweep can be run on AISHELL
and CAVEMOVE to reject a scale fix that only helps one domain.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from torch.utils.data import DataLoader, Subset

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.datasets import TripletExtractionDataset
from car_tse.metrics.si_sdr import si_sdr_per_example
from car_tse.models.stage1_system import Stage1System
from car_tse.train.utils import get_device, load_checkpoint


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            out[key] = torch.stack(values)
        else:
            out[key] = values
    return out


def rms(wave: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    return torch.sqrt(wave.float().square().mean(dim=dims).clamp_min(1.0e-8))


def scale_batch(
    mode: str,
    multichannel: torch.Tensor,
    enrollment: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one bounded scale policy to a batch."""

    mc_rms = rms(multichannel, (1, 2))
    enroll_rms = rms(enrollment, (1,))

    if mode == "raw":
        return multichannel, enrollment
    if mode == "mix_rms_to_enrollment":
        scale = (enroll_rms / mc_rms).clamp(0.05, 20.0)
        return multichannel * scale[:, None, None], enrollment
    if mode.startswith("mix_rms_"):
        target = float(mode.removeprefix("mix_rms_"))
        scale = (target / mc_rms).clamp(0.05, 20.0)
        return multichannel * scale[:, None, None], enrollment
    if mode.startswith("joint_rms_"):
        target = float(mode.removeprefix("joint_rms_"))
        mix_scale = (target / mc_rms).clamp(0.05, 20.0)
        enroll_scale = (target / enroll_rms).clamp(0.05, 20.0)
        return (
            multichannel * mix_scale[:, None, None],
            enrollment * enroll_scale[:, None],
        )
    raise ValueError(f"Unsupported scale mode: {mode}")


def summarize(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "n": int(tensor.numel()),
        "mean": float(tensor.mean()),
        "median": float(tensor.median()),
        "std": float(tensor.std(unbiased=False)),
        "p10": float(torch.quantile(tensor, 0.10)),
        "p90": float(torch.quantile(tensor, 0.90)),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "negative": int((tensor < 0).sum()),
    }


def resolve_path(value: str | Path, base: Path = REPO_ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "stage1_spex_transfer_mc_cavemove.yaml",
    )
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--modes",
        default="raw,mix_rms_to_enrollment,mix_rms_0.02,mix_rms_0.05,mix_rms_0.1,mix_rms_0.2,joint_rms_0.05,joint_rms_0.1",
    )
    args = parser.parse_args()

    cfg = load_stage_config(args.config, args.paths)
    paths = cfg["paths"]
    device = get_device()
    manifest = resolve_path(args.manifest)
    checkpoint = resolve_path(args.checkpoint)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]

    dataset = TripletExtractionDataset(
        manifest,
        sample_rate=int(cfg["sample_rate"]),
        segment_seconds=float(cfg["segment_seconds"]),
        training=False,
        cabin_noise=bool(cfg.get("cabin_noise", False)),
        num_mics=int(cfg.get("num_mics", 4)),
        cabin_noise_channels=cfg.get("cabin_noise_channels"),
        cabin_noise_snr_db=cfg.get("cabin_noise_snr_db"),
        cabin_noise_cache_size=int(cfg.get("cabin_noise_cache_size", 32)),
    )
    if args.limit > 0:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    pretrained = cfg.get("pretrained_checkpoint", paths["spex_pretrained_dir"])
    pretrained = resolve_path(pretrained)
    model = Stage1System(
        spex_cfg=cfg["spex"],
        sample_rate=int(cfg["sample_rate"]),
        use_cabin_frontend=bool(cfg.get("use_cabin_frontend", False)),
        num_mics=int(cfg.get("num_mics", 4)),
        frontend_cfg=cfg.get("frontend"),
        pretrained_dir=str(pretrained),
        backend=cfg.get("spex_backend", "vendor_8k"),
    ).to(device)
    load_checkpoint(checkpoint, model)
    model.eval()

    per_mode: dict[str, dict[str, Any]] = {
        mode: {"output_si_sdr_db": [], "input_si_sdr_db": [], "si_sdri_db": [], "mix_rms": [], "enrollment_rms": [], "frontend_rms": []}
        for mode in modes
    }
    examples = 0
    with torch.no_grad():
        for batch in loader:
            mix = batch["mixture"].to(device)
            enrollment = batch["enrollment"].to(device)
            target = batch["target"].to(device)
            multichannel = batch.get("multichannel")
            playback = batch.get("playback_ref")
            if not isinstance(multichannel, torch.Tensor):
                raise RuntimeError("The scale sweep requires multichannel input in every manifest row")
            multichannel = multichannel.to(device)
            if isinstance(playback, torch.Tensor):
                playback = playback.to(device)
            present = batch.get("target_present")
            if isinstance(present, torch.Tensor):
                present_mask = present.to(device).bool()
            else:
                present_mask = torch.tensor(present, device=device, dtype=torch.bool)

            for mode in modes:
                scaled_mc, scaled_enroll = scale_batch(mode, multichannel, enrollment)
                if model.frontend is not None:
                    model_input = model.frontend(scaled_mc, playback)
                else:
                    model_input = mix
                output, _, _, _ = model(
                    mix,
                    scaled_enroll,
                    multichannel=scaled_mc,
                    playback_ref=playback,
                )
                if not present_mask.any():
                    continue
                output_scores = si_sdr_per_example(output[present_mask], target[present_mask]).float()
                input_scores = si_sdr_per_example(model_input[present_mask], target[present_mask]).float()
                improvement = output_scores - input_scores
                bucket = per_mode[mode]
                bucket["output_si_sdr_db"].extend(float(x) for x in output_scores.cpu())
                bucket["input_si_sdr_db"].extend(float(x) for x in input_scores.cpu())
                bucket["si_sdri_db"].extend(float(x) for x in improvement.cpu())
                bucket["mix_rms"].extend(float(x) for x in rms(scaled_mc[present_mask], (1, 2)).cpu())
                bucket["enrollment_rms"].extend(float(x) for x in rms(scaled_enroll[present_mask], (1,)).cpu())
                bucket["frontend_rms"].extend(float(x) for x in rms(model_input[present_mask], (1,)).cpu())
            examples += int(present_mask.sum())

    results: dict[str, Any] = {
        "benchmark": "stage1_input_scale_sweep",
        "config": str(args.config),
        "manifest": str(manifest),
        "checkpoint": str(checkpoint),
        "device": str(device),
        "examples": examples,
        "modes": {},
    }
    for mode, bucket in per_mode.items():
        output_scores = bucket.pop("output_si_sdr_db")
        input_scores = bucket.pop("input_si_sdr_db")
        improvement = bucket.pop("si_sdri_db")
        results["modes"][mode] = {
            "output_si_sdr_db": summarize(output_scores),
            "input_si_sdr_db": summarize(input_scores),
            "si_sdri_db": summarize(improvement),
            "mean_mix_rms": float(torch.tensor(bucket["mix_rms"]).mean()),
            "mean_enrollment_rms": float(torch.tensor(bucket["enrollment_rms"]).mean()),
            "mean_frontend_rms": float(torch.tensor(bucket["frontend_rms"]).mean()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    ranked = sorted(
        results["modes"].items(),
        key=lambda item: item[1]["si_sdri_db"]["mean"],
        reverse=True,
    )
    print(f"Evaluated {examples} target-present examples on {device}")
    for mode, result in ranked:
        print(
            f"{mode:24s} output={result['output_si_sdr_db']['mean']:+7.2f} dB "
            f"input={result['input_si_sdr_db']['mean']:+7.2f} dB "
            f"SI-SDRi={result['si_sdri_db']['mean']:+7.2f} dB "
            f"frontend_rms={result['mean_frontend_rms']:.5f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
