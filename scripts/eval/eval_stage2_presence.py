#!/usr/bin/env python
"""Evaluate presence FAR/FRR."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from torch.utils.data import DataLoader, Subset, default_collate

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.datasets import PresenceDataset
from car_tse.models.presence_head import PresenceHead
from car_tse.models.stage1_system import Stage1System
from car_tse.train.utils import get_device, load_checkpoint, write_metrics


def collate_presence(batch: list[dict], expected_mics: int) -> dict:
    normalized = []
    for item in batch:
        row = dict(item)
        multichannel = row.get("multichannel")
        if multichannel is None:
            multichannel = row["mixture"].unsqueeze(0)
        elif multichannel.dim() == 1:
            multichannel = multichannel.unsqueeze(0)
        if multichannel.shape[0] < expected_mics:
            repeats = (expected_mics + multichannel.shape[0] - 1) // multichannel.shape[0]
            multichannel = multichannel.repeat(repeats, 1)[:expected_mics]
        elif multichannel.shape[0] > expected_mics:
            multichannel = multichannel[:expected_mics]
        row["multichannel"] = multichannel
        normalized.append(row)
    return default_collate(normalized)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "stage2_presence.yaml")
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs" / "paths.yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional explicit manifest override")
    parser.add_argument(
        "--allow-generated-val",
        action="store_true",
        help="Allow the legacy deterministic validation split from the training manifest when no val manifest exists",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional presence checkpoint override")
    parser.add_argument("--output", type=Path, default=None, help="Optional metrics output override")
    args = parser.parse_args()
    cfg = load_stage_config(args.config, args.paths)
    paths = cfg["paths"]
    device = get_device()
    stage1_cfg_path = Path(cfg.get("stage1_config", REPO_ROOT / "configs" / "stage1_spex_mc.yaml"))
    if not stage1_cfg_path.is_absolute():
        stage1_cfg_path = REPO_ROOT / stage1_cfg_path
    stage1_cfg = load_stage_config(stage1_cfg_path, args.paths)
    stage1 = Stage1System(
        spex_cfg=stage1_cfg["spex"],
        sample_rate=stage1_cfg["sample_rate"],
        use_cabin_frontend=stage1_cfg.get("use_cabin_frontend", False),
        num_mics=stage1_cfg.get("num_mics", 4),
        frontend_cfg=stage1_cfg.get("frontend"),
        pretrained_dir=str(stage1_cfg.get("pretrained_checkpoint", paths["spex_pretrained_dir"])),
        backend=stage1_cfg.get("spex_backend", "vendor_8k"),
    ).to(device)
    load_checkpoint(paths["checkpoint_root"] / cfg["spex_checkpoint"], stage1)
    head = PresenceHead(cfg["model"]["hidden"]).to(device)
    head_checkpoint = args.checkpoint or (paths["checkpoint_root"] / cfg["name"] / "best.pt")
    if not head_checkpoint.is_absolute():
        head_checkpoint = REPO_ROOT / head_checkpoint
    load_checkpoint(head_checkpoint, head)
    stage1.eval()
    head.eval()
    manifest_root = Path(paths["manifest_root"])
    manifest = args.manifest or (manifest_root / args.split / cfg["manifest"])
    if not manifest.is_absolute():
        manifest = REPO_ROOT / manifest
    dataset_kwargs = {
        "cabin_noise": bool(cfg.get("cabin_noise", False)),
        "num_mics": int(cfg.get("num_mics", 4)),
        "cabin_noise_channels": cfg.get("cabin_noise_channels"),
        "cabin_noise_snr_db": cfg.get("cabin_noise_snr_db"),
        "cabin_noise_cache_size": int(cfg.get("cabin_noise_cache_size", 32)),
    }
    manifest_mode = "explicit_or_configured"
    if manifest.exists():
        ds = PresenceDataset(
            manifest, cfg["sample_rate"], cfg["segment_seconds"], training=False, **dataset_kwargs
        )
    elif args.split == "val" and args.manifest is None and args.allow_generated_val:
        # The CAVEMOVE presence trainer creates its validation split from the
        # training manifest. Recreate that exact deterministic split when no
        # standalone validation manifest was exported.
        source_manifest = manifest_root / "train" / cfg["manifest"]
        if not source_manifest.exists():
            raise FileNotFoundError(f"Training manifest for generated validation split not found: {source_manifest}")
        ds_base = PresenceDataset(
            source_manifest,
            cfg["sample_rate"],
            cfg["segment_seconds"],
            training=False,
            **dataset_kwargs,
        )
        rng = random.Random(int(cfg["train"].get("seed", 2026)))
        val_indices = []
        for label in (0, 1):
            indices = [
                i for i, row in enumerate(ds_base.rows)
                if int(row.get("target_present", 1)) == label
            ]
            rng.shuffle(indices)
            cut = max(1, int(round(len(indices) * float(cfg["train"].get("val_fraction", 0.2))))) if indices else 0
            val_indices.extend(indices[:cut])
        rng.shuffle(val_indices)
        ds = Subset(ds_base, val_indices)
        manifest = source_manifest
        manifest_mode = "generated_from_training_manifest"
    else:
        hint = "; pass --allow-generated-val only for the legacy deterministic split" if args.split == "val" else ""
        raise FileNotFoundError(f"Presence manifest not found: {manifest}{hint}")
    expected_mics = int(getattr(stage1.frontend, "num_mics", cfg.get("num_mics", 4)))
    loader = DataLoader(ds, batch_size=8, collate_fn=lambda batch: collate_presence(batch, expected_mics))
    # Older stage-2 training configs did not persist the inference threshold.
    # Keep the benchmark reproducible by using the pipeline default in that
    # case rather than failing during evaluation.
    thr = float(cfg.get("presence_threshold", 0.5))
    tp = fp = tn = fn = 0
    probabilities: list[float] = []
    truths: list[bool] = []
    with torch.no_grad():
        for batch in loader:
            mix = batch["mixture"].to(device)
            enroll = batch["enrollment"].to(device)
            labels = batch["target_present"]
            multichannel = batch.get("multichannel")
            playback_ref = batch.get("playback_ref")
            if multichannel is not None:
                multichannel = multichannel.to(device)
            if playback_ref is not None:
                playback_ref = playback_ref.to(device)
            s1, _, _, _ = stage1(mix, enroll, multichannel=multichannel, playback_ref=playback_ref)
            prob = torch.sigmoid(head(s1)).cpu()
            for p, y in zip(prob, labels):
                score = p.item()
                pred = score >= thr
                truth = float(y) >= 0.5
                probabilities.append(score)
                truths.append(truth)
                if truth and pred:
                    tp += 1
                elif truth and not pred:
                    fn += 1
                elif not truth and pred:
                    fp += 1
                else:
                    tn += 1
    far = fp / max(fp + tn, 1)
    frr = fn / max(fn + tp, 1)

    def at_threshold(candidate: float) -> dict[str, float | int]:
        ctp = cfp = ctn = cfn = 0
        for score, truth in zip(probabilities, truths):
            pred = score >= candidate
            if truth and pred:
                ctp += 1
            elif truth and not pred:
                cfn += 1
            elif not truth and pred:
                cfp += 1
            else:
                ctn += 1
        precision = ctp / max(ctp + cfp, 1)
        recall = ctp / max(ctp + cfn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        specificity = ctn / max(ctn + cfp, 1)
        return {
            "threshold": candidate,
            "accuracy": (ctp + ctn) / max(len(probabilities), 1),
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "far": cfp / max(cfp + ctn, 1),
            "frr": cfn / max(cfn + ctp, 1),
            "true_positive": ctp,
            "false_positive": cfp,
            "true_negative": ctn,
            "false_negative": cfn,
        }

    sweep = [at_threshold(i / 100.0) for i in range(1, 100)]
    best_f1 = max(sweep, key=lambda item: float(item["f1"]))
    best_balanced = max(
        sweep,
        key=lambda item: (float(item["recall"]) + float(item["specificity"])) / 2.0,
    )
    write_metrics(
        args.output or (paths["log_root"] / cfg["name"] / f"metrics_{args.split}.json"),
        {
            "far": far,
            "frr": frr,
            "threshold": thr,
            "examples": tp + fp + tn + fn,
            "manifest": str(manifest),
            "manifest_mode": manifest_mode,
            "requested_split": args.split,
            "best_f1": best_f1,
            "best_balanced_accuracy": best_balanced,
        },
    )
    print(f"FAR={far:.3f} FRR={frr:.3f} best_f1={best_f1['f1']:.3f}@{best_f1['threshold']:.2f} balanced={best_balanced['accuracy']:.3f}@{best_balanced['threshold']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
