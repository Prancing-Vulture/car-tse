"""Check that an intent waveform cache and feature cache use identical rows.

This is a downstream-data diagnostic. It never updates TSE or any model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))

from car_tse.data.datasets import IntentDataset  # noqa: E402
from train_stage4_intent_wav2vec import IntentHead, normalize_waveform  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wave-cache", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = IntentDataset(
        args.manifest,
        16000,
        2.0,
        training=False,
        cabin_noise=False,
        cabin_noise_cache_size=0,
        audio_cache_size=0,
        waveform_cache_path=args.wave_cache,
        waveform_cache_is_noisy=True,
    )
    feature_cache = np.load(args.feature_cache, mmap_mode="r", allow_pickle=False)
    if len(dataset) != feature_cache.shape[0]:
        raise ValueError(f"row mismatch: dataset={len(dataset)} features={feature_cache.shape[0]}")

    encoder = AutoModel.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    encoder_state = torch.load(args.encoder_checkpoint, map_location="cpu", weights_only=False)
    encoder.load_state_dict(encoder_state["model_state_dict"], strict=True)
    encoder.eval()
    head = IntentHead(768, 512, 31, 0.2).to(device)
    head_state = torch.load(args.head_checkpoint, map_location="cpu", weights_only=False)
    head.load_state_dict(head_state["model_state_dict"], strict=True)
    head.eval()

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    checked = 0
    feature_mse = []
    feature_cosine = []
    raw_correct = cached_correct = 0
    with torch.inference_mode():
        for batch in loader:
            if checked >= args.limit:
                break
            take = min(len(batch["wav"]), args.limit - checked)
            wav = normalize_waveform(batch["wav"][:take].to(device))
            raw_features = encoder(wav).last_hidden_state.mean(dim=1).float()
            cached_features = torch.from_numpy(
                np.asarray(feature_cache[checked : checked + take], dtype=np.float32)
            ).to(device)
            raw_logits = head(raw_features)
            cached_logits = head(cached_features)
            labels = batch["label"][:take].to(device)
            raw_correct += int((raw_logits.argmax(-1) == labels).sum())
            cached_correct += int((cached_logits.argmax(-1) == labels).sum())
            feature_mse.append(float(torch.mean((raw_features - cached_features) ** 2).cpu()))
            feature_cosine.append(
                float(torch.nn.functional.cosine_similarity(raw_features, cached_features, dim=-1).mean().cpu())
            )
            checked += take

    print({
        "device": str(device),
        "checked": checked,
        "raw_accuracy": raw_correct / max(checked, 1),
        "cached_feature_accuracy": cached_correct / max(checked, 1),
        "feature_mse": float(np.mean(feature_mse)),
        "feature_cosine": float(np.mean(feature_cosine)),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
