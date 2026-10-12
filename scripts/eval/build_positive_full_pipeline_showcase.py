#!/usr/bin/env python
"""Build and validate a positive full-pipeline cabin example.

The public corpora do not contain the same speaker saying both the wake word
and an FSC command.  This script therefore creates an explicitly labeled
cross-clip positive: a real ``marvin`` clip followed by a real FSC command,
then adds Libri2Mix passenger speech and CAVEMOVE multichannel noise.
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
from car_tse.io.audio import read_mono, read_multichannel
from car_tse.metrics.si_sdr import si_sdr
from car_tse.pipeline.in_car_pipeline import normalize_extracted_waveform
from car_tse.train.utils import get_device
from scripts.eval.build_showcase import load_pipeline


def fit_start(wav: np.ndarray, samples: int) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    out = np.zeros(samples, dtype=np.float32)
    out[: min(samples, len(wav))] = wav[:samples]
    return out


def fit_center(wav: np.ndarray, samples: int) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if len(wav) <= samples:
        return fit_start(wav, samples)
    start = (len(wav) - samples) // 2
    return wav[start : start + samples].astype(np.float32)


def peak_listen(wav: np.ndarray, peak: float = 0.95) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    maximum = float(np.max(np.abs(wav))) if wav.size else 0.0
    return wav * (peak / maximum) if maximum > 1.0e-8 else wav


def write_float(path: Path, wav: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(wav, dtype=np.float32), sample_rate, subtype="FLOAT")


def save_pair(output: Path, stem: str, wav: np.ndarray, sample_rate: int) -> dict[str, str]:
    raw = output / f"{stem}_raw.wav"
    listen = output / f"{stem}_listen.wav"
    write_float(raw, wav, sample_rate)
    write_float(listen, peak_listen(wav), sample_rate)
    return {"raw": str(raw.resolve()), "listen": str(listen.resolve())}


def rms(wav: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(wav, dtype=np.float32) ** 2) + 1.0e-8))


def main() -> int:
    parser = argparse.ArgumentParser(description="Find and export a fully positive CAVEMOVE pipeline example")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "reports/artifacts/full_pipeline_positive")
    parser.add_argument("--max-candidates", type=int, default=48)
    parser.add_argument("--noise-snrs", default="0,5,10,15", help="Cabin-noise SNR candidates, lowest tried first")
    parser.add_argument("--passenger-snrs", default="0,5,10,15", help="Passenger-speech SNR candidates, lowest tried first")
    args = parser.parse_args()

    cfg = load_stage_config(REPO_ROOT / "configs/eval_end_to_end_cavemove.yaml", REPO_ROOT / "configs/paths.yaml")
    paths = cfg["paths"]
    sample_rate = int(cfg["sample_rate"])
    total_samples = 3 * sample_rate
    wake_samples = 1 * sample_rate
    command_samples = 2 * sample_rate
    device = get_device()
    pipeline = load_pipeline(cfg, paths, device)

    wake_manifest = REPO_ROOT / "manifests/test/stage3_kws_cabin_test.jsonl"
    wake_rows = [
        json.loads(line)
        for line in wake_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    wake_rows = [row for row in wake_rows if int(row.get("label", 0)) == 1 and not row.get("noise_only", False)]

    intent_manifest = REPO_ROOT / "manifests/val/stage4_cavemove_intent.jsonl"
    intent_rows = [
        json.loads(line)
        for line in intent_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Keep the search compact while covering several intent classes.
    selected_intents: list[dict] = []
    seen_intents: set[str] = set()
    for row in intent_rows:
        intent = str(row.get("intent", row.get("label", "")))
        if intent in seen_intents:
            continue
        selected_intents.append(row)
        seen_intents.add(intent)
        if len(selected_intents) >= 8:
            break

    # A held-out Libri2Mix speech mixture supplies a competing/passenger voice.
    passenger_row = json.loads(
        (REPO_ROOT / "manifests/test/stage1_cavemove_cabin.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    passenger_mix, _ = read_mono(passenger_row["mixture_wav"], sample_rate)
    passenger_target, _ = read_mono(passenger_row["target_wav"], sample_rate)
    passenger = fit_center(passenger_mix, total_samples) - fit_center(passenger_target, total_samples)

    noise_path = Path(passenger_row["cabin_noise_wav"])
    cabin_noise_all, _ = read_multichannel(noise_path, sample_rate)
    if cabin_noise_all.shape[1] < 4:
        repeats = int(np.ceil(4 / cabin_noise_all.shape[1]))
        cabin_noise_all = np.tile(cabin_noise_all, (1, repeats))
    cabin_noise_all = cabin_noise_all[:, [0, 2, 4, 6] if cabin_noise_all.shape[1] >= 7 else list(range(4))]
    cabin_noise_all = np.asarray(cabin_noise_all, dtype=np.float32)
    if cabin_noise_all.shape[0] >= total_samples:
        noise_start = (cabin_noise_all.shape[0] - total_samples) // 2
        cabin_noise_base = cabin_noise_all[noise_start : noise_start + total_samples]
    else:
        repeats = int(np.ceil(total_samples / max(1, cabin_noise_all.shape[0])))
        cabin_noise_base = np.tile(cabin_noise_all, (repeats, 1))[:total_samples]

    candidates: list[dict] = []
    max_candidates = max(1, int(args.max_candidates))
    noise_snrs = tuple(float(value.strip()) for value in args.noise_snrs.split(",") if value.strip())
    passenger_snrs = tuple(float(value.strip()) for value in args.passenger_snrs.split(",") if value.strip())
    if not noise_snrs or not passenger_snrs:
        raise SystemExit("--noise-snrs and --passenger-snrs must each contain at least one numeric value")

    for wake_index, wake_row in enumerate(wake_rows[:8]):
        wake, _ = read_mono(wake_row["wav"], sample_rate)
        wake = fit_start(wake, wake_samples)
        for intent_index, intent_row in enumerate(selected_intents):
            command, _ = read_mono(intent_row["wav"], sample_rate)
            command = fit_start(command, command_samples)
            target = np.concatenate([wake, command], axis=0).astype(np.float32)
            # The best available cross-corpus enrollment is the target prefix;
            # its construction is recorded explicitly in the final metadata.
            enrollment = fit_start(target, int(1.5 * sample_rate))
            for noise_snr in noise_snrs:
                for passenger_snr in passenger_snrs:
                    target_rms = rms(target)
                    passenger_component = passenger * (
                        target_rms * (10.0 ** (-passenger_snr / 20.0)) / max(rms(passenger), 1.0e-6)
                    )
                    speech_mix = target + passenger_component
                    noise_scale = rms(speech_mix) * (10.0 ** (-noise_snr / 20.0)) / max(
                        rms(cabin_noise_base), 1.0e-6
                    )
                    multichannel = speech_mix[:, None] + cabin_noise_base * noise_scale
                    mixture = multichannel.mean(axis=1)
                    mix_tensor = torch.from_numpy(mixture).to(device)
                    enrollment_tensor = torch.from_numpy(enrollment).to(device)
                    multichannel_tensor = torch.from_numpy(multichannel.T).to(device)

                    with torch.inference_mode():
                        wake_output, state = pipeline.forward(
                            mix_tensor,
                            enrollment_tensor,
                            multichannel=multichannel_tensor,
                        )
                        # The command is deliberately evaluated on a second
                        # post-wake window.  The assembled positive places the
                        # command immediately after the one-second wake clip.
                        output = wake_output
                        if wake_output.command_pending:
                            post_wake_mix = np.zeros_like(mixture)
                            post_wake_multichannel = np.zeros_like(multichannel)
                            post_start = wake_samples
                            post_count = min(len(mixture) - post_start, len(post_wake_mix))
                            if post_count > 0:
                                post_wake_mix[:post_count] = mixture[post_start : post_start + post_count]
                                post_wake_multichannel[:post_count] = multichannel[post_start : post_start + post_count]
                            output, state = pipeline.forward(
                                torch.from_numpy(post_wake_mix).to(device),
                                enrollment_tensor,
                                state,
                                multichannel=torch.from_numpy(post_wake_multichannel.T).to(device),
                            )

                        presence_probability = float(torch.sigmoid(pipeline.presence(wake_output.s1)).mean().cpu())
                        extracted = wake_output.s1.detach().cpu().squeeze(0).numpy().astype(np.float32)
                        extracted_downstream = normalize_extracted_waveform(wake_output.s1)
                        if output.intent_logits is not None:
                            probs = torch.softmax(output.intent_logits, dim=-1)[0]
                            predicted_id = str(int(probs.argmax().item()))
                            predicted_intent = pipeline.label_map.get(predicted_id, predicted_id)
                            intent_confidence = float(probs.max().cpu())
                        else:
                            predicted_id = None
                            predicted_intent = None
                            intent_confidence = None
                        kws_input = extracted_downstream
                        kws_samples = max(1, int(round(pipeline.kws_sample_rate * pipeline.kws_input_seconds)))
                        kws_input = kws_input[..., :kws_samples]
                        if kws_input.shape[-1] < kws_samples:
                            kws_input = torch.nn.functional.pad(kws_input, (0, kws_samples - kws_input.shape[-1]))
                        kws_probs = torch.softmax(pipeline.kws(kws_input), dim=-1)[0]
                        wake_probability = float(kws_probs[pipeline.wake_class_id].cpu())

                    expected_intent = str(intent_row.get("intent", ""))
                    passed = bool(
                        wake_output.accepted
                        and wake_output.wake_detected
                        and output.intent_ready
                        and predicted_intent == expected_intent
                    )
                    candidate = {
                        "wake_row": wake_row,
                        "intent_row": intent_row,
                        "target": target,
                        "enrollment": enrollment,
                        "passenger_component": passenger_component,
                        "cabin_noise_component": cabin_noise_base * noise_scale,
                        "multichannel": multichannel.astype(np.float32),
                        "mixture": mixture.astype(np.float32),
                        "extracted": extracted,
                        "extracted_downstream": extracted_downstream.detach().cpu().squeeze(0).numpy().astype(np.float32),
                        "kws_input": kws_input.detach().cpu().squeeze(0).numpy().astype(np.float32),
                        "presence_probability": presence_probability,
                        "wake_probability": wake_probability,
                        "accepted": bool(wake_output.accepted),
                        "wake_detected": bool(wake_output.wake_detected),
                        "intent_ready": bool(output.intent_ready),
                        "predicted_intent_id": predicted_id,
                        "predicted_intent": predicted_intent,
                        "intent_confidence": intent_confidence,
                        "expected_intent": expected_intent,
                        "noise_snr_db": noise_snr,
                        "passenger_snr_db": passenger_snr,
                        "passed": passed,
                        "wake_index": wake_index,
                        "intent_index": intent_index,
                    }
                    candidates.append(candidate)
                if len(candidates) >= max_candidates:
                    break
            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break

    winners = [candidate for candidate in candidates if candidate["passed"]]
    if not winners:
        best = max(candidates, key=lambda c: (c["wake_detected"], c["predicted_intent"] == c["expected_intent"], c["wake_probability"]))
        raise SystemExit(
            "No fully positive composite found; best candidate was "
            f"wake={best['wake_detected']} intent={best['predicted_intent']} expected={best['expected_intent']} "
            f"wake_probability={best['wake_probability']:.3f}"
        )

    # Prefer the hardest passing scene; use model margin as a tie-breaker.
    winner = min(
        winners,
        key=lambda candidate: (
            candidate["noise_snr_db"] + candidate["passenger_snr_db"],
            -candidate["wake_probability"],
            -float(candidate["intent_confidence"] or 0.0),
        ),
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = winner["target"]
    output_score = float(si_sdr(torch.from_numpy(winner["extracted"][None]), torch.from_numpy(target[None])))
    input_score = float(si_sdr(torch.from_numpy(winner["mixture"][None]), torch.from_numpy(target[None])))
    audio = {
        "cabin_input_4ch": save_pair(output_dir, "01_cabin_input_4ch", winner["multichannel"], sample_rate),
        "cabin_input_mean": save_pair(output_dir, "02_cabin_input_mean", winner["mixture"], sample_rate),
        "target_wake_plus_command": save_pair(output_dir, "03_target_wake_plus_command", target, sample_rate),
        "passenger_interference": save_pair(output_dir, "04_passenger_interference", winner["passenger_component"], sample_rate),
        "cabin_noise_4ch": save_pair(output_dir, "05_cabin_noise_4ch", winner["cabin_noise_component"], sample_rate),
        "target_enrollment": save_pair(output_dir, "06_target_enrollment", winner["enrollment"], sample_rate),
        "stage1_extracted": save_pair(output_dir, "07_stage1_extracted", winner["extracted"], sample_rate),
        "stage1_downstream_normalized": save_pair(output_dir, "08_stage1_downstream_normalized", winner["extracted_downstream"], sample_rate),
        "kws_input_1s": save_pair(output_dir, "09_kws_input_1s", winner["kws_input"], sample_rate),
    }
    report = {
        "description": "Validated positive full pipeline using real Marvin and FSC clips plus synthetic cabin rendering.",
        "positive_definition": {
            "wake_word": "marvin",
            "wake_ground_truth": True,
            "intent_ground_truth": winner["expected_intent"],
            "intent_prediction_matches": True,
        },
        "construction": {
            "wake_clip": winner["wake_row"]["wav"],
            "command_clip": winner["intent_row"]["wav"],
            "passenger_source": passenger_row["mixture_wav"],
            "cabin_noise_source": str(noise_path),
            "cabin_noise_vehicle": (passenger_row.get("meta") or {}).get("cabin_vehicle"),
            "cabin_noise_snr_db": winner["noise_snr_db"],
            "passenger_snr_db": winner["passenger_snr_db"],
            "enrollment": "first 1.5 seconds of the assembled Marvin+command target",
            "sample_rate": sample_rate,
        },
        "pipeline": {
            "presence_probability": winner["presence_probability"],
            "presence_passed": winner["accepted"],
            "wake_probability": winner["wake_probability"],
            "wake_detected": winner["wake_detected"],
            "predicted_intent": winner["predicted_intent"],
            "expected_intent": winner["expected_intent"],
            "intent_confidence": winner["intent_confidence"],
            "all_positive_checks_passed": True,
        },
        "tse_metrics": {
            "input_si_sdr_db": input_score,
            "extracted_si_sdr_db": output_score,
            "si_sdri_db": output_score - input_score,
        },
        "audio": audio,
        "note": (
            "This is an end-to-end positive engineering demo. The wake-word and command come from separate public "
            "corpora because no same-speaker Marvin+command recording is present. The assembled target/enrollment "
            "is therefore a cross-clip validation, not a natural-recording benchmark. *_listen.wav files are peak-normalized."
        ),
    }
    metadata_path = output_dir / "positive_full_pipeline.json"
    metadata_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
