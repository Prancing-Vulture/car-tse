#!/usr/bin/env python
"""Create a small audio/JSON showcase from the current cabin-aware models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import soundfile as sf
import torch

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.datasets import IntentDataset, TripletExtractionDataset
from car_tse.data.segment import segment_waveform
from car_tse.metrics.si_sdr import si_sdr
from car_tse.models.stage1_system import Stage1System
from car_tse.pipeline.in_car_pipeline import InCarPipeline
from car_tse.train.utils import get_device, load_checkpoint


def save_audio(path: Path, audio: np.ndarray, sample_rate: int) -> float:
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    scale = 1.0
    if peak > 0.95:
        scale = 0.95 / peak
        audio = audio * scale
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate)
    return scale


def load_pipeline(cfg: dict, paths: dict, device: torch.device) -> InCarPipeline:
    stage1_cfg_path = Path(cfg["stage1_config"])
    if not stage1_cfg_path.is_absolute():
        stage1_cfg_path = REPO_ROOT / stage1_cfg_path
    stage1_cfg = load_stage_config(stage1_cfg_path, REPO_ROOT / "configs" / "paths.yaml")
    pretrained = Path(stage1_cfg["pretrained_checkpoint"])
    if not pretrained.is_absolute():
        pretrained = REPO_ROOT / pretrained
    stage1 = Stage1System(
        spex_cfg=stage1_cfg["spex"],
        sample_rate=stage1_cfg["sample_rate"],
        use_cabin_frontend=True,
        num_mics=cfg["num_mics"],
        frontend_cfg=stage1_cfg.get("frontend"),
        pretrained_dir=str(pretrained),
        backend=stage1_cfg["spex_backend"],
    ).to(device)
    load_checkpoint(Path(paths["checkpoint_root"]) / cfg["checkpoints"]["spex"], stage1)
    label_map_path = Path(paths["manifest_root"]) / "meta" / "intent_label_map.json"
    raw_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    label_map = {str(value): key for key, value in raw_map.items()}
    pipeline = InCarPipeline.from_checkpoints(
        stage1,
        Path(paths["checkpoint_root"]) / cfg["checkpoints"]["presence"],
        Path(paths["checkpoint_root"]) / cfg["checkpoints"]["kws"],
        Path(paths["checkpoint_root"]) / cfg["checkpoints"]["intent"],
        kws_sample_rate=cfg["sample_rate"],
        kws_num_classes=cfg.get("kws_num_classes", 2),
        intent_sample_rate=cfg["sample_rate"],
        presence_threshold=cfg["presence_threshold"],
        kws_threshold=cfg["kws_threshold"],
        wake_class_id=cfg.get("wake_class_id", 1),
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
    return pipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "logs" / "showcase_cavemove")
    parser.add_argument("--tse-indices", default="0,100,200,300")
    parser.add_argument("--intent-indices", default="0,1,2,3")
    args = parser.parse_args()
    paths_cfg = load_stage_config(REPO_ROOT / "configs" / "eval_end_to_end_cavemove.yaml", args.paths)
    paths = paths_cfg["paths"]
    device = get_device()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    noise_cache_dir = Path(paths_cfg["cabin_noise_cache_dir"])
    if not noise_cache_dir.is_absolute():
        noise_cache_dir = REPO_ROOT / noise_cache_dir
    tse_manifest = Path(paths["manifest_root"]) / paths_cfg["split"] / paths_cfg["manifest"]
    tse_ds = TripletExtractionDataset(
        tse_manifest,
        paths_cfg["sample_rate"],
        paths_cfg["segment_seconds"],
        training=False,
        cabin_noise=True,
        num_mics=paths_cfg["num_mics"],
        cabin_noise_channels=paths_cfg["cabin_noise_channels"],
        cabin_noise_snr_db=paths_cfg["cabin_noise_snr_db"],
        cabin_noise_cache_size=paths_cfg["cabin_noise_cache_size"],
        cabin_noise_cache_dir=noise_cache_dir,
    )
    pipeline = load_pipeline(paths_cfg, paths, device)
    tse_results = []
    with torch.inference_mode():
        for index in [int(x) for x in args.tse_indices.split(",") if x.strip()]:
            item = tse_ds[index]
            mix = item["mixture"]
            enroll = item["enrollment"]
            target = item["target"].to(device).unsqueeze(0)
            multichannel = item["multichannel"]
            output, _ = pipeline.forward(mix, enroll, multichannel=multichannel)
            extracted = output.s1.detach().cpu().squeeze(0)
            input_mono = multichannel.mean(dim=0)
            input_score = float(si_sdr(input_mono.unsqueeze(0).to(device), target).cpu())
            output_score = float(si_sdr(extracted.unsqueeze(0).to(device), target).cpu())
            presence_probability = float(torch.sigmoid(pipeline.presence(output.s1)).mean().cpu())
            wake_probability = float(torch.softmax(pipeline.kws(output.s1), dim=-1)[..., pipeline.wake_class_id].mean().cpu())
            intent_logits = pipeline.intent(output.s1)
            intent_id = str(int(intent_logits.argmax(dim=-1).item()))
            label = pipeline.label_map.get(intent_id, intent_id) if pipeline.label_map else intent_id
            prefix = output_dir / "tse" / f"sample_{index:03d}"
            input_path = prefix.with_name(prefix.name + "_input_mean.wav")
            extracted_path = prefix.with_name(prefix.name + "_extracted.wav")
            target_path = prefix.with_name(prefix.name + "_clean_target.wav")
            save_audio(input_path, input_mono.numpy(), paths_cfg["sample_rate"])
            save_audio(extracted_path, extracted.numpy(), paths_cfg["sample_rate"])
            save_audio(target_path, target.cpu().numpy().squeeze(0), paths_cfg["sample_rate"])
            row = tse_ds.rows[index]
            tse_results.append(
                {
                    "index": index,
                    "vehicle": (row.get("meta") or {}).get("cabin_vehicle"),
                    "cabin_noise_type": (row.get("meta") or {}).get("cabin_noise_type"),
                    "snr_db": row.get("cabin_noise_snr_db"),
                    "input_si_sdr_db": input_score,
                    "extracted_si_sdr_db": output_score,
                    "si_sdri_db": output_score - input_score,
                    "presence_probability": presence_probability,
                    "wake_probability": wake_probability,
                    "accepted": output.accepted,
                    "wake_detected": output.wake_detected,
                    "direct_intent_prediction": label,
                    "input_audio": str(input_path),
                    "extracted_audio": str(extracted_path),
                    "clean_target_audio": str(target_path),
                }
            )

    intent_cfg = load_stage_config(REPO_ROOT / "configs" / "stage4_intent_cavemove.yaml", args.paths)
    intent_cache_dir = Path(intent_cfg["waveform_cache_dir"])
    if not intent_cache_dir.is_absolute():
        intent_cache_dir = REPO_ROOT / intent_cache_dir
    intent_manifest = Path(paths["manifest_root"]) / "val" / intent_cfg["manifest_eval"]
    intent_ds = IntentDataset(
        intent_manifest,
        intent_cfg["sample_rate"],
        intent_cfg["segment_seconds"],
        training=False,
        cabin_noise=True,
        cabin_noise_snr_db=intent_cfg["cabin_noise_snr_db"],
        cabin_noise_cache_size=intent_cfg["cabin_noise_cache_size"],
        audio_cache_size=0,
        waveform_cache_path=intent_cache_dir / f"val_cabin_wav_{intent_cfg['sample_rate']}hz.npy",
        cabin_noise_cache_dir=REPO_ROOT / intent_cfg["cabin_noise_cache_dir"],
        waveform_cache_is_noisy=True,
    )
    raw_labels = json.loads((Path(paths["manifest_root"]) / "meta" / intent_cfg["label_map"]).read_text(encoding="utf-8"))
    labels_by_id = {int(value): key for key, value in raw_labels.items()}
    intent_results = []
    with torch.inference_mode():
        for index in [int(x) for x in args.intent_indices.split(",") if x.strip()]:
            item = intent_ds[index]
            noisy = item["wav"]
            row = intent_ds.rows[index]
            logits = pipeline.intent(noisy.unsqueeze(0).to(device))
            probabilities = torch.softmax(logits, dim=-1)[0]
            prediction = int(probabilities.argmax().item())
            expected = int(row["label"])
            clean, _ = __import__("car_tse.io.audio", fromlist=["read_mono"]).read_mono(row["wav"], intent_cfg["sample_rate"])
            clean = segment_waveform(clean, intent_ds.num_samples, training=False)
            prefix = output_dir / "intent" / f"sample_{index:03d}"
            noisy_path = prefix.with_name(prefix.name + "_cabin_input.wav")
            clean_path = prefix.with_name(prefix.name + "_clean_input.wav")
            save_audio(noisy_path, noisy.numpy(), intent_cfg["sample_rate"])
            save_audio(clean_path, clean, intent_cfg["sample_rate"])
            intent_results.append(
                {
                    "index": index,
                    "expected_intent": labels_by_id.get(expected, str(expected)),
                    "predicted_intent": labels_by_id.get(prediction, str(prediction)),
                    "confidence": float(probabilities[prediction].cpu()),
                    "correct": prediction == expected,
                    "snr_db": row.get("cabin_noise_snr_db"),
                    "cabin_input_audio": str(noisy_path),
                    "clean_input_audio": str(clean_path),
                }
            )

    report = {
        "description": "Held-out Volkswagen CAVEMOVE TSE samples and held-out Fluent Speech Commands intent samples.",
        "sample_rate": paths_cfg["sample_rate"],
        "tse": tse_results,
        "intent": intent_results,
    }
    (output_dir / "showcase.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
