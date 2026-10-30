#!/usr/bin/env python
"""Export one complete CAVEMOVE cabin pipeline input/output showcase.

The scene is a held-out Libri2Mix target+distractor mixture rendered through a
four-microphone CAVEMOVE cabin-noise condition, then passed through the actual
Stage 1 -> presence -> wake-word -> intent path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from car_tse.config import load_stage_config
from car_tse.data.datasets import TripletExtractionDataset
from car_tse.data.segment import segment_waveform
from car_tse.io.audio import read_mono
from car_tse.metrics.si_sdr import si_sdr
from car_tse.pipeline.in_car_pipeline import normalize_extracted_waveform
from car_tse.train.utils import get_device
from scripts.eval.build_showcase import load_pipeline


def write_float(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sample_rate, subtype="FLOAT")


def listen_normalize(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    maximum = float(np.max(np.abs(audio))) if audio.size else 0.0
    if maximum <= 1.0e-8:
        return audio
    return audio * (peak / maximum)


def write_pair(output: Path, stem: str, audio: np.ndarray, sample_rate: int) -> dict[str, str]:
    raw = output / f"{stem}_raw.wav"
    listen = output / f"{stem}_listen.wav"
    write_float(raw, audio, sample_rate)
    write_float(listen, listen_normalize(audio), sample_rate)
    return {"raw": str(raw.resolve()), "listen": str(listen.resolve())}


def main() -> int:
    parser = argparse.ArgumentParser(description="Export one full CAVEMOVE pipeline example")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "reports/artifacts/full_pipeline_cavemove_sample_000",
    )
    args = parser.parse_args()

    paths_cfg = load_stage_config(REPO_ROOT / "configs/eval_end_to_end_cavemove.yaml", REPO_ROOT / "configs/paths.yaml")
    paths = paths_cfg["paths"]
    device = get_device()
    sample_rate = int(paths_cfg["sample_rate"])
    num_samples = int(round(float(paths_cfg["segment_seconds"]) * sample_rate))

    noise_cache_dir = Path(paths_cfg["cabin_noise_cache_dir"])
    if not noise_cache_dir.is_absolute():
        noise_cache_dir = REPO_ROOT / noise_cache_dir
    manifest = Path(paths["manifest_root"]) / paths_cfg["split"] / paths_cfg["manifest"]
    dataset = TripletExtractionDataset(
        manifest,
        sample_rate,
        paths_cfg["segment_seconds"],
        training=False,
        cabin_noise=True,
        num_mics=int(paths_cfg["num_mics"]),
        cabin_noise_channels=paths_cfg["cabin_noise_channels"],
        cabin_noise_snr_db=paths_cfg["cabin_noise_snr_db"],
        cabin_noise_cache_size=paths_cfg["cabin_noise_cache_size"],
        cabin_noise_cache_dir=noise_cache_dir,
    )
    if args.index < 0 or args.index >= len(dataset):
        raise SystemExit(f"index {args.index} is outside the test set of {len(dataset)} rows")

    row = dataset.rows[args.index]
    item = dataset[args.index]
    multichannel = item["multichannel"].numpy().astype(np.float32)  # [4, T]
    target = item["target"].numpy().astype(np.float32)
    enrollment = item["enrollment"].numpy().astype(np.float32)
    cabin_input = multichannel.mean(axis=0)

    # The CAVEMOVE renderer adds the same Libri2Mix speech mixture to each
    # microphone and adds the selected real cabin recording per channel.
    speech_mix_full, _ = read_mono(row["mixture_wav"], sample_rate)
    speech_mix = segment_waveform(speech_mix_full, num_samples, training=False)
    cabin_noise = multichannel - speech_mix[None, :]
    estimated_distractor = speech_mix - target

    pipeline = load_pipeline(paths_cfg, paths, device)
    with torch.inference_mode():
        output, _ = pipeline.forward(
            item["mixture"],
            item["enrollment"],
            multichannel=item["multichannel"],
        )
        extracted = output.s1.detach().cpu().squeeze(0).numpy().astype(np.float32)
        downstream = normalize_extracted_waveform(output.s1).detach().cpu().squeeze(0).numpy().astype(np.float32)
        presence_probability = float(torch.sigmoid(pipeline.presence(output.s1)).mean().cpu())

        kws_input = normalize_extracted_waveform(output.s1)
        kws_samples = max(1, int(round(pipeline.kws_sample_rate * pipeline.kws_input_seconds)))
        if kws_input.shape[-1] >= kws_samples:
            kws_input = kws_input[..., :kws_samples]
        else:
            kws_input = torch.nn.functional.pad(kws_input, (0, kws_samples - kws_input.shape[-1]))
        kws_probabilities = torch.softmax(pipeline.kws(kws_input), dim=-1)[0]
        wake_probability = float(kws_probabilities[pipeline.wake_class_id].cpu())

        intent_probabilities = None
        predicted_intent = None
        intent_confidence = None
        if output.intent_logits is not None:
            intent_probabilities = torch.softmax(output.intent_logits, dim=-1)[0]
            predicted_intent_id = str(int(intent_probabilities.argmax().item()))
            predicted_intent = pipeline.label_map.get(predicted_intent_id, predicted_intent_id)
            intent_confidence = float(intent_probabilities.max().cpu())

    input_score = float(si_sdr(torch.from_numpy(cabin_input[None]), torch.from_numpy(target[None])))
    output_score = float(si_sdr(torch.from_numpy(extracted[None]), torch.from_numpy(target[None])))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    audio = {
        "cabin_input_4ch": write_pair(output_dir, "01_cabin_input_4ch", multichannel.T, sample_rate),
        "cabin_input_mean": write_pair(output_dir, "02_cabin_input_mean", cabin_input, sample_rate),
        "speech_mixture_without_cabin_noise": write_pair(
            output_dir, "03_speech_mixture_without_cabin_noise", speech_mix, sample_rate
        ),
        "estimated_passenger_interference": write_pair(
            output_dir, "04_estimated_passenger_interference", estimated_distractor, sample_rate
        ),
        "cabin_noise_4ch": write_pair(output_dir, "05_cabin_noise_4ch", cabin_noise.T, sample_rate),
        "target_clean_reference": write_pair(output_dir, "06_target_clean_reference", target, sample_rate),
        "target_enrollment": write_pair(output_dir, "07_target_enrollment", enrollment, sample_rate),
        "stage1_extracted_output": write_pair(output_dir, "08_stage1_extracted_output", extracted, sample_rate),
        "stage1_downstream_normalized": write_pair(
            output_dir, "09_stage1_downstream_normalized", downstream, sample_rate
        ),
        "kws_input_1s": write_pair(output_dir, "10_kws_input_1s", kws_input[0].detach().cpu().numpy(), sample_rate),
    }

    action = output.action.action if output.action is not None else None
    intent_id = output.action.intent_id if output.action is not None else None
    report = {
        "description": "One held-out CAVEMOVE/Libri2Mix cabin scene through Stage 1 TSE, presence, KWS, and intent.",
        "dataset": "Libri2Mix target+distractor rendered with CAVEMOVE cabin noise",
        "sample_rate": sample_rate,
        "row_id": row.get("id"),
        "vehicle": (row.get("meta") or {}).get("cabin_vehicle"),
        "mic_setup": (row.get("meta") or {}).get("cabin_mic_setup"),
        "cabin_noise_type": (row.get("meta") or {}).get("cabin_noise_type"),
        "cabin_noise_wav": row.get("cabin_noise_wav"),
        "cabin_noise_channels": row.get("cabin_noise_channels"),
        "cabin_noise_snr_db": row.get("cabin_noise_snr_db"),
        "target_speaker_id": row.get("speaker_id"),
        "pipeline": {
            "presence_probability": presence_probability,
            "presence_threshold": paths_cfg["presence_threshold"],
            "wake_probability": wake_probability,
            "wake_threshold": paths_cfg["kws_threshold"],
            "accepted": bool(output.accepted),
            "wake_detected": bool(output.wake_detected),
            "predicted_intent_id": intent_id,
            "predicted_intent": predicted_intent,
            "intent_confidence": intent_confidence,
            "vehicle_action": action,
        },
        "tse_metrics": {
            "input_si_sdr_db": input_score,
            "extracted_si_sdr_db": output_score,
            "si_sdri_db": output_score - input_score,
        },
        "audio": audio,
        "note": (
            "This is a valid full computational-path showcase, but this CAVEMOVE TSE row has no ground-truth "
            "wake-word or intent label. The final KWS/intent values are model decisions, not an accuracy claim. "
            "*_listen.wav files are independently peak-normalized; raw files preserve the generated waveforms."
        ),
    }
    metadata_path = output_dir / "full_pipeline.json"
    metadata_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
