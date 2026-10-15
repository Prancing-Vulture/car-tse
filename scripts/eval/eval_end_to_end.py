#!/usr/bin/env python
"""Evaluate the current 16 kHz cabin-aware end-to-end pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from torch.utils.data import DataLoader

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.datasets import TripletExtractionDataset
from car_tse.metrics.si_sdr import si_sdr
from car_tse.models.stage1_system import Stage1System
from car_tse.pipeline.in_car_pipeline import InCarPipeline, PipelineState
from car_tse.train.utils import get_device, load_checkpoint, write_metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "eval_end_to_end_cavemove.yaml",
    )
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    cfg = load_stage_config(args.config, args.paths)
    paths = cfg["paths"]
    device = get_device()

    stage1_config_path = Path(
        cfg.get("stage1_config", REPO_ROOT / "configs" / "stage1_spex_transfer_mc_cavemove.yaml")
    )
    if not stage1_config_path.is_absolute():
        stage1_config_path = REPO_ROOT / stage1_config_path
    stage1_cfg = load_stage_config(stage1_config_path, args.paths)
    pretrained = Path(stage1_cfg["pretrained_checkpoint"])
    if not pretrained.is_absolute():
        pretrained = REPO_ROOT / pretrained
    stage1 = Stage1System(
        spex_cfg=stage1_cfg["spex"],
        sample_rate=stage1_cfg["sample_rate"],
        use_cabin_frontend=stage1_cfg.get("use_cabin_frontend", True),
        num_mics=stage1_cfg.get("num_mics", cfg.get("num_mics", 4)),
        frontend_cfg=stage1_cfg.get("frontend"),
        pretrained_dir=str(pretrained),
        backend=stage1_cfg.get("spex_backend", "transfer_spexplus_16k"),
    ).to(device)
    load_checkpoint(Path(paths["checkpoint_root"]) / cfg["checkpoints"]["spex"], stage1)

    label_map_path = Path(paths["manifest_root"]) / "meta" / "intent_label_map.json"
    label_map = {}
    if label_map_path.exists():
        raw_map = json.loads(label_map_path.read_text(encoding="utf-8"))
        label_map = {str(value): key for key, value in raw_map.items()}
    stage23_fusion_checkpoint = cfg.get("checkpoints", {}).get("stage23_fusion")
    if stage23_fusion_checkpoint:
        stage23_fusion_checkpoint = Path(paths["checkpoint_root"]) / stage23_fusion_checkpoint
    pipeline = InCarPipeline.from_checkpoints(
        stage1,
        Path(paths["checkpoint_root"]) / cfg["checkpoints"]["presence"],
        Path(paths["checkpoint_root"]) / cfg["checkpoints"]["kws"],
        Path(paths["checkpoint_root"]) / cfg["checkpoints"]["intent"],
        kws_sample_rate=cfg["sample_rate"],
        intent_sample_rate=cfg["sample_rate"],
        kws_num_classes=int(cfg.get("kws_num_classes", 2)),
        intent_type=str(cfg.get("intent_type", "crnn")),
        intent_encoder=str(cfg.get("intent_encoder", "facebook/wav2vec2-base-960h")),
        intent_encoder_checkpoint=(
            Path(paths["checkpoint_root"]) / cfg["intent_encoder_checkpoint"]
            if cfg.get("intent_encoder_checkpoint")
            else None
        ),
        intent_num_classes=int(cfg.get("intent_num_classes", 31)),
        presence_threshold=float(cfg["presence_threshold"]),
        kws_threshold=float(cfg["kws_threshold"]),
        stage23_fusion_checkpoint=stage23_fusion_checkpoint,
        stage23_fusion_threshold=float(cfg.get("stage23_fusion_threshold", 0.5)),
        wake_class_id=int(cfg.get("wake_class_id", 1)),
        kws_input_seconds=float(cfg.get("kws_input_seconds", 1.0)),
        kws_smoothing_window=int(cfg.get("kws_smoothing_window", 1)),
        kws_required_hits=int(cfg.get("kws_required_hits", 1)),
        kws_refractory_calls=int(cfg.get("kws_refractory_calls", 0)),
        intent_input_seconds=float(cfg.get("intent_input_seconds", 2.0)),
        command_timeout_calls=int(cfg.get("command_timeout_calls", 4)),
        command_presence_threshold=(
            float(cfg["command_presence_threshold"])
            if cfg.get("command_presence_threshold") is not None
            else None
        ),
        presence_type=str(cfg.get("presence_type", "scalar")),
        presence_sample_rate=int(cfg.get("presence_sample_rate", cfg["sample_rate"])),
        presence_n_mels=int(cfg.get("presence_n_mels", 40)),
        presence_hidden=int(cfg.get("presence_hidden", 64)),
        presence_n_fft=int(cfg.get("presence_n_fft", 512)),
        presence_hop=int(cfg.get("presence_hop", 160)),
        presence_top_fraction=float(cfg.get("presence_top_fraction", 0.20)),
        label_map=label_map,
        device=device,
    )

    noise_cache_dir = Path(cfg.get("cabin_noise_cache_dir", "")) if cfg.get("cabin_noise_cache_dir") else None
    if noise_cache_dir is not None and not noise_cache_dir.is_absolute():
        noise_cache_dir = REPO_ROOT / noise_cache_dir
    manifest = Path(paths["manifest_root"]) / cfg["split"] / cfg["manifest"]
    dataset = TripletExtractionDataset(
        manifest,
        cfg["sample_rate"],
        cfg.get("segment_seconds", stage1_cfg.get("segment_seconds", 3.0)),
        training=False,
        cabin_noise=bool(cfg.get("cabin_noise", False)),
        num_mics=int(cfg.get("num_mics", 4)),
        cabin_noise_channels=cfg.get("cabin_noise_channels"),
        cabin_noise_snr_db=cfg.get("cabin_noise_snr_db"),
        cabin_noise_cache_size=int(cfg.get("cabin_noise_cache_size", 32)),
        cabin_noise_cache_dir=noise_cache_dir,
    )
    if args.limit:
        from torch.utils.data import Subset

        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    accepted = wake = present = 0
    si_scores: list[float] = []
    state = PipelineState()
    with torch.inference_mode():
        for batch in loader:
            mix = batch["mixture"][0]
            enroll = batch["enrollment"][0]
            target = batch["target"][0].to(device).unsqueeze(0)
            multichannel = batch.get("multichannel")
            if isinstance(multichannel, torch.Tensor):
                multichannel = multichannel[0]
            out, state = pipeline.forward(mix, enroll, state, multichannel=multichannel)
            if out.s1 is not None:
                si_scores.append(float(si_sdr(out.s1, target)))
            if out.accepted:
                accepted += 1
            if out.wake_detected:
                wake += 1
            if out.accepted and out.s1 is not None:
                present += 1
    n = len(dataset)
    metrics = {
        "si_sdr_mean": sum(si_scores) / max(len(si_scores), 1),
        "accept_rate": accepted / max(n, 1),
        "presence_rate": present / max(n, 1),
        "wake_rate": wake / max(n, 1),
        "n": n,
        "device": str(device),
        "kws_threshold": float(cfg["kws_threshold"]),
    }
    output = Path(cfg.get("output", Path(paths["log_root"]) / "eval_end_to_end_cavemove.json"))
    if not output.is_absolute():
        output = REPO_ROOT / output
    write_metrics(output, metrics)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
