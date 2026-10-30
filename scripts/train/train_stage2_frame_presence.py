#!/usr/bin/env python
"""Train a lightweight framewise target-activity head on frozen Stage 1 output."""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.data.datasets import PresenceDataset
from car_tse.losses.presence import (
    frame_activity_loss,
    frame_presence_loss,
    target_activity_class_labels,
    target_activity_labels,
)
from car_tse.models.presence_head import (
    FramePresenceHead,
    PersonalActivityHead,
    PersonalFramePresenceHead,
)
from car_tse.models.stage1_system import Stage1System
from car_tse.io.manifest import read_jsonl
from car_tse.train.utils import append_jsonl, get_device, load_checkpoint, save_checkpoint, write_metrics


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collate(batch: list[dict]) -> dict:
    output: dict = {}
    keys = set().union(*(item.keys() for item in batch))
    for key in keys:
        values = [item.get(key) for item in batch]
        if all(isinstance(value, torch.Tensor) for value in values):
            output[key] = torch.stack(values)
        elif all(isinstance(value, (int, float)) for value in values):
            output[key] = torch.tensor(values)
        else:
            output[key] = values
    return output


class CachedPresenceDataset(Dataset):
    """CPU feature cache for repeated frozen-Stage-1 head experiments."""

    def __init__(
        self,
        stage1_output: torch.Tensor,
        frame_labels: torch.Tensor,
        target_present: torch.Tensor,
        enrollment_embedding: torch.Tensor | None = None,
        target_activity_class: torch.Tensor | None = None,
    ):
        self.stage1_output = stage1_output.float().contiguous()
        self.frame_labels = frame_labels.float().contiguous()
        self.target_present = target_present.float().contiguous()
        self.enrollment_embedding = (
            enrollment_embedding.float().contiguous() if enrollment_embedding is not None else None
        )
        self.target_activity_class = (
            target_activity_class.long().contiguous() if target_activity_class is not None else None
        )

    def __len__(self) -> int:
        return int(self.target_present.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | float]:
        item = {
            "s1": self.stage1_output[index],
            "frame_labels": self.frame_labels[index],
            "target_present": self.target_present[index],
        }
        if self.enrollment_embedding is not None:
            item["enrollment_embedding"] = self.enrollment_embedding[index]
        if self.target_activity_class is not None:
            item["target_activity_class"] = self.target_activity_class[index]
        return item


def split_indices(
    rows: list[dict],
    val_fraction: float,
    seed: int,
    max_rows: int = 0,
    group_by_speaker: bool = False,
) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    selected_indices = list(range(len(rows)))
    if max_rows > 0:
        grouped: dict[int, list[int]] = {0: [], 1: []}
        for index, row in enumerate(rows):
            grouped[int(row.get("target_present", 0) >= 0.5)].append(index)
        per_class = max(1, max_rows // 2)
        for label in grouped:
            rng.shuffle(grouped[label])
            grouped[label] = grouped[label][:per_class]
        selected_indices = grouped[0] + grouped[1]

    if group_by_speaker:
        grouped_by_speaker: dict[str, list[int]] = {}
        for index in selected_indices:
            row = rows[index]
            group = str(
                row.get("group_id")
                or row.get("speaker_id")
                or row.get("source_speaker_id")
                or f"row-{index}"
            )
            grouped_by_speaker.setdefault(group, []).append(index)
        groups = list(grouped_by_speaker.values())
        rng.shuffle(groups)
        target_validation_rows = min(
            max(1, int(round(len(selected_indices) * val_fraction))),
            max(1, len(selected_indices) - 1),
        )
        validation: list[int] = []
        train: list[int] = []
        for group in groups:
            if len(validation) < target_validation_rows and len(validation) + len(group) < len(selected_indices):
                validation.extend(group)
            else:
                train.extend(group)
        if not train and validation:
            train.extend(validation.pop())
        rng.shuffle(train)
        rng.shuffle(validation)
        return train, validation

    grouped = {0: [], 1: []}
    for index in selected_indices:
        grouped[int(rows[index].get("target_present", 0) >= 0.5)].append(index)
    train: list[int] = []
    validation: list[int] = []
    for indices in grouped.values():
        rng.shuffle(indices)
        cut = max(1, int(round(len(indices) * val_fraction))) if indices else 0
        validation.extend(indices[:cut])
        train.extend(indices[cut:])
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def align_frames(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    frames = min(logits.shape[-1], labels.shape[-1])
    return logits[..., :frames], labels[..., :frames]


def binary_metrics(scores: np.ndarray, truth: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = scores >= float(threshold)
    truth = truth.astype(bool)
    tp = int(np.logical_and(predicted, truth).sum())
    fp = int(np.logical_and(predicted, ~truth).sum())
    tn = int(np.logical_and(~predicted, ~truth).sum())
    fn = int(np.logical_and(~predicted, truth).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-8)
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "false_accept_rate": fp / max(fp + tn, 1),
        "false_reject_rate": fn / max(fn + tp, 1),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
    }


@torch.no_grad()
def build_stage1_cache(
    stage1: Stage1System,
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    n_fft: int,
    hop: int,
    target_threshold_db: float,
    personal_conditioned: bool = False,
    activity_mode: bool = False,
) -> CachedPresenceDataset:
    """Run frozen Stage 1 once so later head epochs avoid re-running SpEx+."""

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    stage1_outputs: list[torch.Tensor] = []
    frame_labels: list[torch.Tensor] = []
    target_present_values: list[torch.Tensor] = []
    target_activity_classes: list[torch.Tensor] = []
    enrollment_embeddings: list[torch.Tensor] = []
    for batch in loader:
        mix = batch["mixture"].to(device, non_blocking=True)
        enroll = batch["enrollment"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        target_present = batch["target_present"].to(device, non_blocking=True).float()
        activity_class = batch.get("target_activity_class")
        if isinstance(activity_class, torch.Tensor):
            activity_class = activity_class.to(device, non_blocking=True).long()
        else:
            activity_class = target_present.long()
        mc = batch.get("multichannel")
        if isinstance(mc, torch.Tensor):
            mc = mc.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            s1, _, _, _ = stage1(mix, enroll, multichannel=mc)
        if personal_conditioned:
            enrollment_embedding = stage1.speaker_embedding(enroll)
            if enrollment_embedding is None:
                raise RuntimeError("Personal presence training requires Stage 1 speaker embeddings")
            enrollment_embeddings.append(enrollment_embedding.detach().float().cpu())
        labels = (
            target_activity_class_labels(
                target,
                activity_class,
                n_fft=n_fft,
                hop=hop,
                threshold_db_below_peak=target_threshold_db,
            )
            if activity_mode
            else target_activity_labels(
                target,
                target_present,
                n_fft=n_fft,
                hop=hop,
                threshold_db_below_peak=target_threshold_db,
            )
        )
        stage1_outputs.append(s1.detach().float().cpu())
        frame_labels.append(labels.detach().float().cpu())
        target_present_values.append(target_present.detach().float().cpu())
        if activity_mode:
            target_activity_classes.append(activity_class.detach().long().cpu())
    if not stage1_outputs:
        return CachedPresenceDataset(
            torch.zeros((0, 1), dtype=torch.float32),
            torch.zeros((0, 1), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.float32),
            torch.zeros((0, 256), dtype=torch.float32) if personal_conditioned else None,
            torch.zeros((0,), dtype=torch.long) if activity_mode else None,
        )
    return CachedPresenceDataset(
        torch.cat(stage1_outputs, dim=0),
        torch.cat(frame_labels, dim=0),
        torch.cat(target_present_values, dim=0),
        torch.cat(enrollment_embeddings, dim=0) if personal_conditioned else None,
        torch.cat(target_activity_classes, dim=0) if activity_mode else None,
    )


@torch.no_grad()
def evaluate(
    head: torch.nn.Module,
    stage1: Stage1System,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    n_fft: int,
    hop: int,
    target_threshold_db: float,
    top_fraction: float,
    personal_conditioned: bool = False,
    activity_mode: bool = False,
) -> dict:
    head.eval()
    frame_scores: list[np.ndarray] = []
    frame_labels: list[np.ndarray] = []
    chunk_scores: list[np.ndarray] = []
    chunk_labels: list[np.ndarray] = []
    for batch in loader:
        target_present = batch["target_present"].to(device, non_blocking=True).float()
        activity_class = batch.get("target_activity_class")
        if isinstance(activity_class, torch.Tensor):
            activity_class = activity_class.to(device, non_blocking=True).long()
        else:
            activity_class = target_present.long()
        cached_s1 = batch.get("s1")
        cached_labels = batch.get("frame_labels")
        if isinstance(cached_s1, torch.Tensor) and isinstance(cached_labels, torch.Tensor):
            s1 = cached_s1.to(device, non_blocking=True)
            labels = cached_labels.to(device, non_blocking=True)
            enrollment_embedding = batch.get("enrollment_embedding")
            if isinstance(enrollment_embedding, torch.Tensor):
                enrollment_embedding = enrollment_embedding.to(device, non_blocking=True)
        else:
            mix = batch["mixture"].to(device, non_blocking=True)
            enroll = batch["enrollment"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            mc = batch.get("multichannel")
            if isinstance(mc, torch.Tensor):
                mc = mc.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                s1, _, _, _ = stage1(mix, enroll, multichannel=mc)
            enrollment_embedding = stage1.speaker_embedding(enroll) if personal_conditioned else None
            labels = (
                target_activity_class_labels(
                    target,
                    activity_class,
                    n_fft=n_fft,
                    hop=hop,
                    threshold_db_below_peak=target_threshold_db,
                )
                if activity_mode
                else target_activity_labels(
                    target,
                    target_present,
                    n_fft=n_fft,
                    hop=hop,
                    threshold_db_below_peak=target_threshold_db,
                )
            )
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = (
                head(s1.float(), enrollment_embedding)
                if personal_conditioned
                else head(s1.float())
            )
        logits = logits.float()
        if activity_mode:
            frames = min(logits.shape[-1], labels.shape[-1])
            logits, labels = logits[..., :frames], labels[..., :frames]
            scores = torch.softmax(logits, dim=1)[:, 1, :]
            binary_labels = (labels == 1).float()
        else:
            logits, labels = align_frames(logits, labels)
            scores = torch.sigmoid(logits)
            binary_labels = labels
        chunk = FramePresenceHead.aggregate_probability(scores, top_fraction=top_fraction)
        frame_scores.append(scores.cpu().numpy().reshape(-1))
        frame_labels.append(binary_labels.cpu().numpy().reshape(-1))
        chunk_scores.append(chunk.cpu().numpy().reshape(-1))
        chunk_labels.append(
            ((activity_class == 1) if activity_mode else (target_present >= 0.5))
            .cpu()
            .numpy()
            .reshape(-1)
        )

    frame_score_array = np.concatenate(frame_scores) if frame_scores else np.zeros(0, dtype=np.float32)
    frame_label_array = np.concatenate(frame_labels) if frame_labels else np.zeros(0, dtype=np.float32)
    chunk_score_array = np.concatenate(chunk_scores) if chunk_scores else np.zeros(0, dtype=np.float32)
    chunk_label_array = np.concatenate(chunk_labels) if chunk_labels else np.zeros(0, dtype=np.float32)
    thresholds = [index / 100.0 for index in range(5, 100, 5)]
    frame_sweep = [binary_metrics(frame_score_array, frame_label_array, threshold) for threshold in thresholds]
    chunk_sweep = [binary_metrics(chunk_score_array, chunk_label_array, threshold) for threshold in thresholds]
    best_frame = max(frame_sweep, key=lambda item: float(item["f1"])) if frame_sweep else {}
    best_chunk = max(chunk_sweep, key=lambda item: float(item["f1"])) if chunk_sweep else {}
    balanced_chunk = max(
        chunk_sweep,
        key=lambda item: (float(item["recall"]) + float(item["specificity"])) / 2.0,
    ) if chunk_sweep else {}
    return {
        "frame_at_0.5": binary_metrics(frame_score_array, frame_label_array, 0.5),
        "chunk_at_0.5": binary_metrics(chunk_score_array, chunk_label_array, 0.5),
        "frame_threshold_sweep": frame_sweep,
        "chunk_threshold_sweep": chunk_sweep,
        "best_frame_f1": best_frame,
        "best_chunk_f1": best_chunk,
        "best_chunk_balanced": balanced_chunk,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/stage2_frame_presence_cavemove.yaml")
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs/paths.yaml")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional balanced row cap for a fast pilot")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--wandb", action="store_true", help="Log epoch metrics to Weights & Biases")
    parser.add_argument("--wandb-project", default="car-tse")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-dir", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_stage_config(args.config, args.paths)
    paths = cfg["paths"]
    device = get_device()
    seed = int(cfg["train"].get("seed", 20261004))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    stage1_cfg_path = Path(cfg["stage1_config"])
    if not stage1_cfg_path.is_absolute():
        stage1_cfg_path = REPO_ROOT / stage1_cfg_path
    stage1_cfg = load_stage_config(stage1_cfg_path, args.paths)
    pretrained = Path(stage1_cfg["pretrained_checkpoint"])
    if not pretrained.is_absolute():
        pretrained = REPO_ROOT / pretrained
    stage1 = Stage1System(
        spex_cfg=stage1_cfg["spex"],
        sample_rate=int(stage1_cfg["sample_rate"]),
        use_cabin_frontend=bool(stage1_cfg.get("use_cabin_frontend", True)),
        num_mics=int(stage1_cfg.get("num_mics", 4)),
        frontend_cfg=stage1_cfg.get("frontend"),
        pretrained_dir=str(pretrained),
        backend=stage1_cfg.get("spex_backend", "transfer_spexplus_16k"),
    ).to(device)
    checkpoint = Path(paths["checkpoint_root"]) / cfg["spex_checkpoint"]
    load_checkpoint(checkpoint, stage1)
    stage1.eval()
    for parameter in stage1.parameters():
        parameter.requires_grad = False

    manifest_names = cfg.get("manifests") or [cfg["manifest"]]
    if isinstance(manifest_names, str):
        manifest_names = [manifest_names]
    manifest = Path(paths["manifest_root"]) / "train" / str(manifest_names[0])
    cache_stage1 = bool(cfg["train"].get("cache_stage1", False))
    model_cfg = cfg.get("model", {})
    model_type = str(model_cfg.get("type", "frame")).lower()
    personal_conditioned = model_type in {"personal", "personal_frame", "personal_activity", "personal_activity_3way"}
    activity_mode = model_type in {"personal_activity", "personal_activity_3way"}
    dataset_kwargs = {
        "cabin_noise": bool(cfg.get("cabin_noise", True)),
        "num_mics": int(cfg.get("num_mics", 4)),
        "cabin_noise_channels": cfg.get("cabin_noise_channels", [0, 2, 4, 6]),
        "cabin_noise_snr_db": cfg.get("cabin_noise_snr_db"),
        "cabin_noise_cache_size": int(cfg.get("cabin_noise_cache_size", 32)),
    }
    train_base = PresenceDataset(
        manifest,
        cfg["sample_rate"],
        cfg["segment_seconds"],
        training=not cache_stage1,
        **dataset_kwargs,
    )
    # A mixed manifest lets the small frame head retain the GSC/AISHELL
    # target-positive scale while learning passenger hard negatives.  The
    # dataset class is row-backed, so extending its rows avoids duplicating
    # the audio-loading and cabin-noise logic.
    for extra_name in list(manifest_names)[1:]:
        extra_path = Path(paths["manifest_root"]) / "train" / str(extra_name)
        train_base.rows.extend(read_jsonl(extra_path))
    train_indices, val_indices = split_indices(
        train_base.rows,
        float(cfg["train"].get("val_fraction", 0.2)),
        seed,
        max_rows=int(args.max_rows or 0),
        group_by_speaker=bool(cfg["train"].get("group_by_speaker", False)),
    )
    val_base = PresenceDataset(manifest, cfg["sample_rate"], cfg["segment_seconds"], training=False, **dataset_kwargs)
    for extra_name in list(manifest_names)[1:]:
        extra_path = Path(paths["manifest_root"]) / "train" / str(extra_name)
        val_base.rows.extend(read_jsonl(extra_path))
    train_ds = Subset(train_base, train_indices)
    val_ds = Subset(val_base, val_indices)
    workers = int(cfg["train"].get("num_workers", 0))
    loader_args = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        loader_args["prefetch_factor"] = 2
    batch_size = int(args.batch_size or cfg["train"]["batch_size"])
    cache_use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    cache_model_cfg = cfg.get("model", {})
    if cache_stage1:
        print("Caching frozen Stage 1 outputs for fast frame-head epochs...", flush=True)
        train_ds = build_stage1_cache(
            stage1,
            train_ds,
            int(cfg["train"].get("cache_batch_size", batch_size)),
            device,
            cache_use_amp,
            int(cache_model_cfg.get("n_fft", 512)),
            int(cache_model_cfg.get("hop", 160)),
            float(cfg.get("model", {}).get("target_threshold_db_below_peak", 30.0)),
            personal_conditioned=personal_conditioned,
            activity_mode=activity_mode,
        )
        val_ds = build_stage1_cache(
            stage1,
            val_ds,
            int(cfg["train"].get("cache_batch_size", batch_size)),
            device,
            cache_use_amp,
            int(cfg.get("model", {}).get("n_fft", 512)),
            int(cfg.get("model", {}).get("hop", 160)),
            float(cfg.get("model", {}).get("target_threshold_db_below_peak", 30.0)),
            personal_conditioned=personal_conditioned,
            activity_mode=activity_mode,
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=collate,
        **loader_args,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"].get("val_batch_size", batch_size)),
        shuffle=False,
        collate_fn=collate,
        **loader_args,
    )

    n_fft = int(model_cfg.get("n_fft", 512))
    hop = int(model_cfg.get("hop", 160))
    target_threshold_db = float(model_cfg.get("target_threshold_db_below_peak", 30.0))
    top_fraction = float(model_cfg.get("top_fraction", 0.20))
    head_kwargs = {
        "sample_rate": int(cfg["sample_rate"]),
        "n_mels": int(model_cfg.get("n_mels", 40)),
        "hidden": int(model_cfg.get("hidden", 64)),
        "n_fft": n_fft,
        "hop": hop,
    }
    if activity_mode:
        head = PersonalActivityHead(
            **head_kwargs,
            speaker_embedding_dim=int(model_cfg.get("speaker_embedding_dim", 256)),
        ).to(device)
    elif personal_conditioned:
        head = PersonalFramePresenceHead(
            **head_kwargs,
            speaker_embedding_dim=int(model_cfg.get("speaker_embedding_dim", 256)),
        ).to(device)
    else:
        head = FramePresenceHead(**head_kwargs).to(device)
    init_checkpoint = args.init_checkpoint or cfg["train"].get("init_checkpoint")
    init_path = None
    if init_checkpoint:
        init_path = Path(init_checkpoint)
        if not init_path.is_absolute():
            init_path = Path(paths["checkpoint_root"]) / init_path
        load_checkpoint(init_path, head)
        print(f"Initialized frame presence head from {init_path}", flush=True)
    run_name = args.run_name or str(cfg["name"])
    wandb_run = None
    if args.wandb:
        import wandb

        if args.wandb_dir is not None:
            args.wandb_dir.mkdir(parents=True, exist_ok=True)
        config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            job_type="train-stage2-frame-presence",
            mode=args.wandb_mode,
            dir=str(args.wandb_dir) if args.wandb_dir is not None else None,
            config={
                "stage": "stage2_frame_presence",
                "config": str(args.config),
                "config_sha256": sha256_file(config_path),
                "train_manifest": str(manifest),
                "train_manifest_sha256": sha256_file(manifest),
                "validation_manifest": str(val_manifest) if explicit_val_manifest else "row_split_from_train",
                "validation_manifest_sha256": sha256_file(val_manifest) if explicit_val_manifest else None,
                "stage1_checkpoint": str(checkpoint),
                "stage1_checkpoint_sha256": sha256_file(checkpoint),
                "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
                "init_checkpoint_sha256": sha256_file(init_path) if init_path is not None else None,
                "sample_rate": int(cfg["sample_rate"]),
                "segment_seconds": float(cfg["segment_seconds"]),
                "model": {"n_mels": int(model_cfg.get("n_mels", 40)), "hidden": int(model_cfg.get("hidden", 64))},
                "train": cfg["train"],
                "device": str(device),
                "speaker_disjoint_validation": not bool(overlap) if explicit_val_manifest else False,
            },
        )
        wandb_run.define_metric("epoch")
        wandb_run.define_metric("train/*", step_metric="epoch")
        wandb_run.define_metric("validation/*", step_metric="epoch")

    optimizer = torch.optim.AdamW(head.parameters(), lr=float(cfg["train"]["lr"]))
    use_amp = cache_use_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    max_epochs = int(args.epochs or cfg["train"].get("max_epochs", 4))
    out_dir = Path(paths["checkpoint_root"]) / cfg["name"]
    run_id = uuid.uuid4().hex
    history_path = Path(paths["log_root"]) / cfg["name"] / f"epoch_metrics_{run_id}.jsonl"
    write_metrics(
        Path(paths["log_root"]) / cfg["name"] / f"run_metadata_{run_id}.json",
        {
            "run_id": run_id,
            "run_name": cfg["name"],
            "device": str(device),
            "config": str(args.config),
            "train_manifest": str(manifest),
            "additional_train_manifests": [str(value) for value in manifest_names[1:]],
            "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
        },
    )
    best_f1 = float("-inf")
    if init_checkpoint:
        initial_validation = evaluate(
            head,
            stage1,
            val_loader,
            device,
            use_amp,
            n_fft,
            hop,
            target_threshold_db,
            top_fraction,
        )
        initial_f1 = float(initial_validation["best_chunk_f1"].get("f1", 0.0))
        best_f1 = initial_f1
        save_checkpoint(
            out_dir / "best.pt",
            head,
            0,
            {"epoch": 0, "validation": initial_validation, "initialized_from": str(init_checkpoint)},
        )
        print(f"warm_start_best_chunk_f1={initial_f1:.3f}", flush=True)
        if wandb_run is not None:
            warm = initial_validation["best_chunk_f1"]
            wandb_run.log(
                {
                    "epoch": 0,
                    "validation/chunk_f1": float(warm.get("f1", 0.0)),
                    "validation/chunk_precision": float(warm.get("precision", 0.0)),
                    "validation/chunk_recall": float(warm.get("recall", 0.0)),
                    "validation/chunk_threshold": float(warm.get("threshold", 0.0)),
                    "validation/false_accept_rate": float(warm.get("false_accept_rate", 0.0)),
                },
                step=0,
                commit=True,
            )
    for epoch in range(1, max_epochs + 1):
        head.train()
        losses: list[float] = []
        for batch in train_loader:
            target_present = batch["target_present"].to(device, non_blocking=True).float()
            cached_s1 = batch.get("s1")
            cached_labels = batch.get("frame_labels")
            activity_class = batch.get("target_activity_class")
            if isinstance(activity_class, torch.Tensor):
                activity_class = activity_class.to(device, non_blocking=True).long()
            else:
                activity_class = target_present.long()
            if isinstance(cached_s1, torch.Tensor) and isinstance(cached_labels, torch.Tensor):
                s1 = cached_s1.to(device, non_blocking=True)
                labels = cached_labels.to(device, non_blocking=True)
                enrollment_embedding = batch.get("enrollment_embedding")
                if isinstance(enrollment_embedding, torch.Tensor):
                    enrollment_embedding = enrollment_embedding.to(device, non_blocking=True)
            else:
                mix = batch["mixture"].to(device, non_blocking=True)
                enroll = batch["enrollment"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                mc = batch.get("multichannel")
                if isinstance(mc, torch.Tensor):
                    mc = mc.to(device, non_blocking=True)
                with torch.inference_mode(), torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=use_amp
                ):
                    s1, _, _, _ = stage1(mix, enroll, multichannel=mc)
                enrollment_embedding = stage1.speaker_embedding(enroll) if personal_conditioned else None
                labels = (
                    target_activity_class_labels(
                        target,
                        activity_class,
                        n_fft=n_fft,
                        hop=hop,
                        threshold_db_below_peak=target_threshold_db,
                    )
                    if activity_mode
                    else target_activity_labels(
                        target,
                        target_present,
                        n_fft=n_fft,
                        hop=hop,
                        threshold_db_below_peak=target_threshold_db,
                    )
                )
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = (
                    head(s1.float(), enrollment_embedding)
                    if personal_conditioned
                    else head(s1.float())
                )
                if activity_mode:
                    frames = min(logits.shape[-1], labels.shape[-1])
                    logits, labels = logits[..., :frames], labels[..., :frames]
                    loss, _ = frame_activity_loss(
                        logits,
                        labels,
                        class_weights=tuple(
                            float(value)
                            for value in model_cfg.get("class_weights", [1.0, 2.0, 1.5])
                        ),
                        smooth_weight=float(cfg["train"].get("smooth_weight", 0.02)),
                    )
                else:
                    logits, labels = align_frames(logits, labels)
                    loss, _ = frame_presence_loss(
                        logits,
                        labels,
                        positive_weight=float(cfg["train"].get("positive_weight", 2.0)),
                        smooth_weight=float(cfg["train"].get("smooth_weight", 0.02)),
                    )
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Frame presence loss became non-finite; "
                    f"s1_range=({float(s1.nan_to_num().min()):.4g},{float(s1.nan_to_num().max()):.4g}) "
                    f"logit_range=({float(logits.nan_to_num().min()):.4g},{float(logits.nan_to_num().max()):.4g})"
                )
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(head.parameters(), float(cfg["train"].get("grad_clip", 5.0)))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), float(cfg["train"].get("grad_clip", 5.0)))
                optimizer.step()
            losses.append(float(loss.detach().cpu()))

        validation = evaluate(
            head,
            stage1,
            val_loader,
            device,
            use_amp,
            n_fft,
            hop,
            target_threshold_db,
            top_fraction,
            personal_conditioned=personal_conditioned,
            activity_mode=activity_mode,
        )
        metrics = {
            "epoch": epoch,
            "run_id": run_id,
            "train_loss": float(np.mean(losses)) if losses else None,
            "train_examples": len(train_ds),
            "val_examples": len(val_ds),
            "validation": validation,
            "model": {
                "type": model_type,
                "sample_rate": int(cfg["sample_rate"]),
                "n_mels": int(model_cfg.get("n_mels", 40)),
                "hidden": int(model_cfg.get("hidden", 64)),
                "n_fft": n_fft,
                "hop": hop,
                "top_fraction": top_fraction,
                "target_threshold_db_below_peak": target_threshold_db,
                "speaker_embedding_dim": int(model_cfg.get("speaker_embedding_dim", 256)),
                "num_classes": 3 if activity_mode else 1,
                "group_by_speaker": bool(cfg["train"].get("group_by_speaker", False)),
            },
        }
        write_metrics(Path(paths["log_root"]) / cfg["name"] / "metrics_last.json", metrics)
        append_jsonl(history_path, metrics)
        best = validation["best_chunk_f1"]
        print(
            f"epoch {epoch} loss={metrics['train_loss']:.4f} "
            f"chunk_f1={float(best['f1']):.3f}@{float(best['threshold']):.2f} "
            f"chunk_recall={float(best['recall']):.3f}",
            flush=True,
        )
        if float(best["f1"]) > best_f1:
            best_f1 = float(best["f1"])
            save_checkpoint(out_dir / "best.pt", head, epoch, metrics)

    write_metrics(
        Path(paths["log_root"]) / cfg["name"] / "train_last.json",
        {
            "best_chunk_f1": best_f1,
            "checkpoint": str(out_dir / "best.pt"),
            "run_id": run_id,
            "history": str(history_path),
        },
    )
    if wandb_run is not None:
        wandb_run.summary["best_validation_chunk_f1"] = best_f1
        wandb_run.summary["best_checkpoint"] = str(out_dir / "best.pt")
        wandb_run.finish()
    print(f"Done. Best checkpoint: {out_dir / 'best.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
