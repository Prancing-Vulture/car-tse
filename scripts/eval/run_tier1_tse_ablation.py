#!/usr/bin/env python
"""Run the Tier-1 Stage-1 TSE ablation matrix on the fixed CAVEMOVE test set.

The local SpEx+ models are loaded through the repository implementation. The
external TD-SpeakerBeam checkpoint is loaded as a restricted tensor-only state
dict and evaluated through the pinned ESPnet architecture.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio.functional as AF
import yaml
from pesq import pesq
from pystoi import stoi
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "third_party" / "speaker_extraction_SpEx"))

from car_tse.data.datasets import TripletExtractionDataset
from car_tse.io.audio import read_multichannel
from car_tse.metrics.si_sdr import si_sdr_per_example
from car_tse.models.stage1_system import Stage1System
from car_tse.vendor.spex_plus_adapter import SpExPlusAdapter, SpEx_Plus


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_gender_map(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.lstrip().startswith(";") or "|" not in line:
            continue
        fields = [x.strip() for x in line.split("|")]
        if len(fields) >= 2 and fields[0].isdigit() and fields[1] in {"M", "F"}:
            result[fields[0]] = fields[1]
    return result


def gender_pair(row: dict[str, Any], genders: dict[str, str]) -> tuple[str, str, str]:
    target_id = str(row.get("speaker_id", "")).split("-", 1)[0]
    mix_stem = Path(row["mixture_wav"]).stem
    parts = mix_stem.split("_", 1)
    interferer_id = parts[1].split("-", 1)[0] if len(parts) == 2 else ""
    tg, ig = genders.get(target_id, "unknown"), genders.get(interferer_id, "unknown")
    if tg == "unknown" or ig == "unknown":
        group = "unknown"
    else:
        group = "same_gender" if tg == ig else "different_gender"
    return tg, ig, group


def noise_crop_for_row(
    row: dict[str, Any], sample_rate: int, length: int, cache: dict[str, np.ndarray]
) -> np.ndarray:
    path = str(row["cabin_noise_wav"])
    if path not in cache:
        cache[path] = read_multichannel(path, sample_rate)[0].astype(np.float32, copy=False)
    noise = cache[path]
    channels = [int(x) for x in row.get("cabin_noise_channels", [0, 2, 4, 6])]
    channels = [i for i in channels if 0 <= i < noise.shape[1]]
    if not channels:
        channels = list(range(min(4, noise.shape[1])))
    noise = noise[:, channels]
    if noise.shape[0] < length:
        reps = int(math.ceil(length / max(1, noise.shape[0])))
        noise = np.tile(noise, (reps, 1))
    start = max(0, (noise.shape[0] - length) // 2)
    return noise[start : start + length].T.copy()


def mvdr_from_noise_covariance(
    mixture: torch.Tensor, noise: torch.Tensor, n_fft: int = 512, hop: int = 128
) -> torch.Tensor:
    """Oracle-covariance MVDR with the known identical-channel speech steering vector.

    This is an upper-bound classical comparator: it estimates noise covariance
    from the paired CAVEMOVE noise recording, which is not available in a live
    microphone pipeline.
    """
    batch, channels, length = mixture.shape
    window = torch.hann_window(n_fft, device=mixture.device, dtype=mixture.dtype)

    def stft(x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(batch * channels, length)
        return torch.stft(
            flat,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            center=True,
            return_complex=True,
        ).reshape(batch, channels, n_fft // 2 + 1, -1)

    x_spec = stft(mixture)
    n_spec = stft(noise)
    n_frames = n_spec.shape[-1]
    covariance = torch.einsum("bcft,bdft->bfcd", n_spec, n_spec.conj()) / max(1, n_frames)
    trace = covariance.diagonal(dim1=-2, dim2=-1).real.sum(dim=-1).clamp_min(1.0e-9)
    eye = torch.eye(channels, device=mixture.device, dtype=covariance.dtype)
    covariance = covariance + (1.0e-3 * trace / channels)[..., None, None] * eye
    steering = torch.ones(
        batch, covariance.shape[1], channels, device=mixture.device, dtype=covariance.dtype
    )
    solved = torch.linalg.solve(covariance, steering.unsqueeze(-1)).squeeze(-1)
    denom = (steering.conj() * solved).sum(dim=-1, keepdim=True) + 1.0e-9
    weights = solved / denom
    output_spec = torch.einsum("bfc,bcft->bft", weights.conj(), x_spec)
    return torch.istft(
        output_spec,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        length=length,
    )


def make_current_models(
    args: argparse.Namespace, cfg: dict[str, Any], device: torch.device, safe_base_spex: Path
):
    base_spex = Path(safe_base_spex)
    checkpoint = torch.load(args.current_checkpoint, map_location="cpu", weights_only=True)
    current_state = checkpoint["model_state_dict"]
    common = dict(
        spex_cfg=cfg["spex"],
        sample_rate=cfg["sample_rate"],
        num_mics=4,
        frontend_cfg=cfg["frontend"],
        pretrained_dir=str(base_spex),
        backend="transfer_spexplus_16k",
    )
    full = Stage1System(use_cabin_frontend=True, **common).to(device)
    full.load_state_dict(current_state, strict=True)
    full.eval()

    mono = Stage1System(use_cabin_frontend=False, **common).to(device)
    spex_state = {k: v for k, v in current_state.items() if k.startswith("spex.")}
    mono.load_state_dict(spex_state, strict=True)
    mono.eval()
    return full, mono


def make_external_spex(args: argparse.Namespace, device: torch.device) -> SpExPlusAdapter:
    payload = torch.load(args.external_spex_checkpoint, map_location="cpu", weights_only=True)
    raw_state = payload["model"]
    # The public WSJ0 checkpoint was trained with equivalent module names from
    # another SpEx+ implementation. Only deterministic key renames are allowed;
    # require an exact strict load so no learned tensors are silently dropped.
    state = {}
    for key, value in raw_state.items():
        key = key.removeprefix("module.sep_network.")
        key = key.replace(".lnorm1.", ".norm1.").replace(".lnorm2.", ".norm2.")
        key = key.replace("aux_enc3.", "spk_encoder.")
        key = key.replace("decoder_1d_1.", "decoder_1d_short.")
        key = key.replace("decoder_1d_2.", "decoder_1d_middle.")
        key = key.replace("decoder_1d_3.", "decoder_1d_long.")
        if key in state:
            raise ValueError(f"duplicate external SpEx+ tensor after key mapping: {key}")
        state[key] = value
    nnet = SpEx_Plus(
        L1=20,
        L2=80,
        L3=160,
        N=256,
        B=8,
        O=256,
        P=512,
        Q=3,
        num_spks=101,
        spk_embed_dim=256,
        causal=False,
    )
    nnet.load_state_dict(state, strict=True)
    model = SpExPlusAdapter(nnet).to(device).eval()
    return model


@torch.inference_mode()
def external_spex_forward(model, mixture_16k, enrollment_16k):
    mixture_8k = AF.resample(mixture_16k, 16000, 8000)
    enrollment_8k = AF.resample(enrollment_16k, 16000, 8000)
    extracted_8k = model(mixture_8k, enrollment_8k)[0]
    return AF.resample(extracted_8k, 8000, 16000)


def make_speakerbeam(args: argparse.Namespace, device: torch.device):
    # The wheel and dependencies are isolated under the experiment directory;
    # the research environment's installed packages are left untouched.
    sys.path[:0] = [str(args.espnet_runtime), str(args.espnet_dependencies)]
    from espnet2.enh.decoder.conv_decoder import ConvDecoder
    from espnet2.enh.encoder.conv_encoder import ConvEncoder
    from espnet2.enh.espnet_model_tse import ESPnetExtractionModel
    from espnet2.enh.extractor.td_speakerbeam_extractor import TDSpeakerBeamExtractor

    model_cfg = yaml.safe_load(args.speakerbeam_config.read_text(encoding="utf-8"))
    encoder = ConvEncoder(**model_cfg["encoder_conf"])
    extractor = TDSpeakerBeamExtractor(encoder.output_dim, **model_cfg["extractor_conf"])
    decoder = ConvDecoder(**model_cfg["decoder_conf"])
    model = ESPnetExtractionModel(
        encoder,
        extractor,
        decoder,
        [],
        num_spk=model_cfg["model_conf"]["num_spk"],
        share_encoder=model_cfg["model_conf"]["share_encoder"],
    )
    state = torch.load(args.speakerbeam_checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


@torch.inference_mode()
def speakerbeam_forward(model, mixture_16k, enrollment_16k, device):
    mixture_8k = AF.resample(mixture_16k, 16000, 8000)
    enrollment_8k = AF.resample(enrollment_16k, 16000, 8000)
    lengths = torch.full(
        (mixture_8k.shape[0],), mixture_8k.shape[-1], dtype=torch.long, device=device
    )
    aux_lengths = torch.full(
        (enrollment_8k.shape[0],), enrollment_8k.shape[-1], dtype=torch.long, device=device
    )
    mix_features, mix_feature_lengths = model.encoder(mixture_8k, lengths)
    aux_features, aux_feature_lengths = model.encoder(enrollment_8k, aux_lengths)
    extracted_features, _, _ = model.extractor(
        mix_features, mix_feature_lengths, aux_features, aux_feature_lengths
    )
    extracted_8k, _ = model.decoder(extracted_features, lengths)
    return AF.resample(extracted_8k, 8000, 16000)


def safe_metric(function, target: np.ndarray, estimate: np.ndarray) -> float | None:
    try:
        value = float(function(target, estimate))
        return value if np.isfinite(value) else None
    except Exception:
        return None


def summarize(rows: list[dict[str, Any]], output_path: Path) -> None:
    keys = ("variant", "cabin_noise_snr_db", "gender_pair", "passenger_sir_bin")
    metrics = ("si_sdr_db", "si_sdri_vs_delay_sum_db", "pesq_wb", "stoi")
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for variant, record in row["variants"].items():
            key = (
                variant,
                row["cabin_noise_snr_db"],
                row["gender_pair"],
                row["passenger_sir_bin"],
            )
            groups[key].append(record)
    columns = list(keys) + ["n"] + [f"{m}_{s}" for m in metrics for s in ("mean", "std")]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for key, records in sorted(groups.items(), key=lambda x: tuple(str(y) for y in x[0])):
            out = dict(zip(keys, key))
            out["n"] = len(records)
            for metric in metrics:
                values = [float(r[metric]) for r in records if r.get(metric) is not None]
                out[f"{metric}_mean"] = float(np.mean(values)) if values else ""
                out[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else ""
            # Keep the sample count in every summary row, including metrics.
            out["n"] = len(records)
            writer.writerow(out)


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=REPO / "manifests" / "test" / "stage1_cavemove_cabin.jsonl")
    parser.add_argument("--stage-config", type=Path, default=REPO / "configs" / "stage1_spex_fsc_passenger_cochannel_ft1_2026-10-16.yaml")
    parser.add_argument("--current-checkpoint", type=Path, default=REPO / "checkpoints" / "stage1_spex_fsc_passenger_cochannel_ft1_2026-10-16" / "best.pt")
    parser.add_argument("--base-spex", type=Path, default=Path(r"E:\car-tse-data\models\car_tse_spexplus_from_stage1.pt"))
    parser.add_argument("--external-spex-checkpoint", type=Path, default=Path(r"E:\car-tse-data\models\tier1_external\spexplus_wsj0_2mix_8k\last_best_checkpoint.pt"))
    parser.add_argument("--speakerbeam-checkpoint", type=Path, default=Path(r"E:\car-tse-data\models\tier1_external\espnet_speakerbeam\99epoch.pth"))
    parser.add_argument("--speakerbeam-config", type=Path, default=Path(r"E:\car-tse-data\models\tier1_external\espnet_speakerbeam\config.yaml"))
    parser.add_argument("--espnet-runtime", type=Path, default=Path(r"E:\car-tse-data\experiments\tier1_ablation_2026-10-19\espnet_202511_runtime"))
    parser.add_argument("--espnet-dependencies", type=Path, default=Path(r"E:\car-tse-data\experiments\tier1_ablation_2026-10-19\espnet_202301_runtime"))
    parser.add_argument("--speakers-file", type=Path, default=Path(r"E:\bss-audio-data\speech\LibriSpeech\LibriSpeech\SPEAKERS.TXT"))
    parser.add_argument("--output-dir", type=Path, default=REPO / "reports" / "evaluation" / "tier1_ablation_2026-10-19" / "tse")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    cfg = yaml.safe_load(args.stage_config.read_text(encoding="utf-8"))
    sample_rate = int(cfg["sample_rate"])
    dataset = TripletExtractionDataset(
        args.manifest,
        sample_rate=sample_rate,
        segment_seconds=float(cfg["segment_seconds"]),
        training=False,
        cabin_noise=True,
        num_mics=4,
        cabin_noise_channels=[0, 2, 4, 6],
        cabin_noise_cache_size=32,
    )
    if args.limit:
        dataset.rows = dataset.rows[: args.limit]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    genders = read_gender_map(args.speakers_file)
    base_payload = torch.load(args.base_spex, map_location="cpu", weights_only=True)
    base_state = base_payload.get("model", base_payload.get("model_state_dict", base_payload))
    safe_base_spex = args.output_dir / "base_spex_safe_state_dict.pt"
    torch.save({"model_state_dict": base_state}, safe_base_spex)
    current_fe, current_mono = make_current_models(args, cfg, device, safe_base_spex)
    external_spex = make_external_spex(args, device)
    speakerbeam = make_speakerbeam(args, device)
    noise_cache: dict[str, np.ndarray] = {}

    row_results: list[dict[str, Any]] = []
    next_index = 0
    for batch_index, batch in enumerate(loader):
        mix = batch["mixture"].to(device).float()
        enrollment = batch["enrollment"].to(device).float()
        target = batch["target"].to(device).float()
        multichannel = batch["multichannel"].to(device).float()
        batch_count, channel_count, length = multichannel.shape
        rows = dataset.rows[next_index : next_index + batch_count]
        next_index += batch_count

        noise_np = np.stack(
            [noise_crop_for_row(r, sample_rate, length, noise_cache) for r in rows]
        )
        noise_mc = torch.from_numpy(noise_np).to(device=device, dtype=multichannel.dtype)
        delay_sum = multichannel.mean(dim=1)
        mvdr = mvdr_from_noise_covariance(multichannel, noise_mc)
        base_sisdr = si_sdr_per_example(delay_sum, target)

        outputs: dict[str, torch.Tensor] = {
            "beamformer_delay_sum_only": delay_sum,
            "beamformer_mvdr_oracle_noise_cov_only": mvdr,
        }
        outputs["spx_pretrained_no_cabin_ft_wsj0_2mix"] = external_spex_forward(
            external_spex, delay_sum, enrollment
        )
        outputs["spx_ft_single_ch0"] = current_mono.spex(multichannel[:, 0], enrollment)[0]
        outputs["spx_ft_mean_delay_sum"] = current_mono.spex(delay_sum, enrollment)[0]
        outputs["spx_ft_mvdr_oracle_noise_cov"] = current_mono.spex(mvdr, enrollment)[0]

        for active_channels in (1, 2, 4):
            masked_mc = torch.zeros_like(multichannel)
            masked_mc[:, :active_channels] = multichannel[:, :active_channels]
            outputs[f"learned_frontend_{active_channels}ch_test_time_ablation"] = current_fe(
                mix, enrollment, multichannel=masked_mc
            )[0]

        outputs["espnet_td_speakerbeam_librimix"] = speakerbeam_forward(
            speakerbeam, delay_sum, enrollment, device
        )

        scores = {
            key: si_sdr_per_example(value, target)
            for key, value in outputs.items()
        }
        arrays = {
            key: value.detach().float().cpu().numpy()
            for key, value in outputs.items()
        }
        target_np = target.detach().float().cpu().numpy()
        base_np = base_sisdr.detach().float().cpu().numpy()
        record_rows = []
        for bi, row in enumerate(rows):
            tg, ig, pair = gender_pair(row, genders)
            speech_sir = 10.0 * math.log10(
                float(target[bi].square().mean().cpu())
                / max(float((mix[bi] - target[bi]).square().mean().cpu()), 1.0e-12)
            )
            sir_bin = "low_le_minus3_db" if speech_sir <= -3 else ("mid_minus3_to3_db" if speech_sir <= 3 else "high_gt3_db")
            variant_metrics = {}
            for name, estimate_array in arrays.items():
                estimate = estimate_array[bi]
                ref = target_np[bi]
                variant_metrics[name] = {
                    "si_sdr_db": float(scores[name][bi].detach().cpu()),
                    "si_sdri_vs_delay_sum_db": float(scores[name][bi].detach().cpu()) - float(base_np[bi]),
                    "pesq_wb": safe_metric(lambda a, b: pesq(sample_rate, a, b, "wb"), ref, estimate),
                    "stoi": safe_metric(lambda a, b: stoi(a, b, sample_rate, extended=False), ref, estimate),
                }
            record_rows.append(
                {
                    "row_id": row.get("id", str(next_index - batch_count + bi)),
                    "target_speaker_id": str(row.get("speaker_id", "")),
                    "interferer_speaker_id": (
                        Path(row["mixture_wav"]).stem.split("_", 1)[1].split("-", 1)[0]
                        if "_" in Path(row["mixture_wav"]).stem
                        else "unknown"
                    ),
                    "target_gender": tg,
                    "interferer_gender": ig,
                    "gender_pair": pair,
                    "cabin_noise_snr_db": float(row.get("cabin_noise_snr_db", row.get("meta", {}).get("cabin_noise_snr_db", float("nan")))),
                    "passenger_sir_db_estimated": speech_sir,
                    "passenger_sir_bin": sir_bin,
                    "delay_sum_input_si_sdr_db": float(base_np[bi]),
                    "variants": variant_metrics,
                }
            )
        row_results.extend(record_rows)
        print(f"evaluated {len(row_results)}/{len(dataset)} rows", flush=True)
        del outputs, scores, arrays, target_np, noise_mc, noise_np, mvdr
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows_path = args.output_dir / "per_example_metrics.jsonl"
    with rows_path.open("w", encoding="utf-8") as f:
        for row in row_results:
            f.write(json.dumps(row, allow_nan=False) + "\n")
    summarize(row_results, args.output_dir / "summary_by_snr_gender_sir.csv")

    metadata = {
        "experiment": "Tier-1 TSE ablation matrix",
        "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "repository": str(REPO),
        "stage1_config": str(args.stage_config),
        "stage1_checkpoint": str(args.current_checkpoint),
        "stage1_checkpoint_sha256": sha256_file(args.current_checkpoint),
        "base_spex_checkpoint": str(args.base_spex),
        "base_spex_checkpoint_sha256": sha256_file(args.base_spex),
        "base_spex_sanitized_tensor_state_sha256": sha256_file(safe_base_spex),
        "external_spex_checkpoint": str(args.external_spex_checkpoint),
        "external_spex_checkpoint_sha256": sha256_file(args.external_spex_checkpoint),
        "external_spex_source": "Hugging Face niobures/SpEx-Plus@2cab2f77d560734f83a9cfead45d7f128b804f0f, WSJ0-2mix SpEx+ last_best_checkpoint.pt (epoch 96, best_val_loss about -18.84, 8 kHz, 101 training speakers); SHA256 verified against published file metadata; restricted weights_only load; one-to-one module-key renames followed by strict load; resampled to 8 kHz for inference and back to 16 kHz for metrics",
        "speakerbeam_checkpoint": str(args.speakerbeam_checkpoint),
        "speakerbeam_checkpoint_sha256": sha256_file(args.speakerbeam_checkpoint),
        "speakerbeam_source": "Hugging Face espnet/Wangyou_Zhang_librimix_train_enh_tse_td_speakerbeam_raw@db03c34c0b8c26a87c0d6cac41229891c34b4a02; CC-BY-4.0; restricted weights_only load; pinned ESPnet architecture",
        "test_manifest": str(args.manifest),
        "test_manifest_sha256": sha256_file(args.manifest),
        "speakers_file": str(args.speakers_file),
        "sample_rate_hz": sample_rate,
        "examples": len(row_results),
        "cabin_noise_snr_levels_db": sorted({r["cabin_noise_snr_db"] for r in row_results}),
        "device": str(device),
        "pytorch": torch.__version__,
        "interpretation_limit": "LibriMix target and passenger speech are duplicated identically across channels; only CAVEMOVE noise differs by microphone. Channel count tests are test-time channel-drop ablations of the current four-input frontend, not separately trained 1/2/4-mic models. MVDR uses the paired clean CAVEMOVE noise recording for covariance and is an oracle upper bound, not an on-vehicle deployable estimator.",
        "si_sdri_reference": "All SI-SDRi values use delay-and-sum/mean of the four cabin channels as the common input reference.",
        "seeds": "One current best checkpoint per trained variant; seed-to-seed training variance is not estimated in this run.",
        "files": [rows_path.name, "summary_by_snr_gender_sir.csv", safe_base_spex.name],
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
