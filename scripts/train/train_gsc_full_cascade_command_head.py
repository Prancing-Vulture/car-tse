#!/usr/bin/env python
"""Train a small command-word head on frozen Stage-1 extractions, then replay the full cascade.

Speech Commands clips are paired by speaker: one ``marvin`` recording enrolls
Stage 1 and that speaker's command-word recordings are its targets. Speakers
are split before extraction. Stage 1, Stage 2, and Stage 3 are frozen; SpEx
weights are loaded for inference only and are never written by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel

from car_tse.config import REPO_ROOT, load_stage_config
from car_tse.models.kws_model import KWSModel
from car_tse.models.presence_head import FramePresenceHead
from car_tse.models.stage1_system import Stage1System
from car_tse.models.wav2vec_intent import Wav2VecIntentHead
from car_tse.pipeline.in_car_pipeline import InCarPipeline, PipelineState, normalize_extracted_waveform
from car_tse.train.utils import load_checkpoint, save_checkpoint

SR = 16000
COMMANDS = ("on", "off", "up", "down", "left", "right", "go", "stop")
CLASSES = COMMANDS + ("no_command",)
CH_TO_USE = (0, 2, 4, 6)


def speaker(path: Path) -> str:
    return path.stem.split("_nohash_")[0]


def split_for(sid: str) -> str:
    value = int(hashlib.sha1(sid.encode()).hexdigest()[:8], 16) % 100
    return "train" if value < 80 else ("val" if value < 90 else "test")


def fit(wav: np.ndarray, count: int) -> np.ndarray:
    out = np.zeros(count, np.float32)
    wav = np.asarray(wav, np.float32).reshape(-1)
    n = min(count, len(wav))
    out[:n] = wav[:n]
    return out


def load_noise(path: Path, rng: np.random.Generator) -> np.ndarray:
    audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    if sr != SR:
        # Noise recordings used in this corpus are normally 16 kHz. Linear
        # resampling keeps this prep self-contained if a different rate occurs.
        new_n = max(1, round(len(audio) * SR / sr))
        xp = np.linspace(0, 1, len(audio), endpoint=False)
        xn = np.linspace(0, 1, new_n, endpoint=False)
        audio = np.stack([np.interp(xn, xp, audio[:, c]) for c in range(audio.shape[1])], 1).astype(np.float32)
    if audio.shape[1] <= max(CH_TO_USE):
        audio = np.tile(audio, (1, int(np.ceil(7 / audio.shape[1]))))
    audio = audio[:, CH_TO_USE]
    n = 3 * SR
    start = int(rng.integers(0, max(1, len(audio) - n + 1))) if len(audio) > n else 0
    chunk = audio[start:start+n]
    if len(chunk) < n:
        chunk = np.tile(chunk, (int(np.ceil(n / max(1, len(chunk)))), 1))[:n]
    return chunk.astype(np.float32)


def mix_clip(path: Path | None, noise_path: Path, rng: np.random.Generator, *, seconds: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    n = round(seconds * SR)
    speech = np.zeros(n, np.float32)
    if path is not None:
        x, sr = sf.read(str(path), dtype="float32")
        if x.ndim == 2:
            x = x.mean(axis=1)
        if sr != SR:
            nn = max(1, round(len(x) * SR / sr))
            x = np.interp(np.linspace(0, 1, nn, endpoint=False), np.linspace(0, 1, len(x), endpoint=False), x).astype(np.float32)
        x = np.asarray(x, np.float32)
        # Put short words early in the segment, matching the leading part of
        # the runtime's fixed two-second Stage-4 input.
        start = int(0.25 * SR)
        take = min(len(x), n - start)
        speech[start:start+take] = x[:take]
    noise = load_noise(noise_path, rng)
    if path is None or float(np.sqrt(np.mean(speech**2) + 1e-12)) < 1e-6:
        # Genuine quiet/no-command examples, while still retaining cabin noise.
        multichannel = noise * 0.05
    else:
        snr = float(rng.choice([-5, 0, 5, 10, 15]))
        srms = float(np.sqrt(np.mean(speech**2) + 1e-12))
        nrms = float(np.sqrt(np.mean(noise**2) + 1e-12))
        multichannel = noise * (srms * 10 ** (-snr / 20) / max(nrms, 1e-8)) + speech[:, None]
    return multichannel.mean(axis=1).astype(np.float32), multichannel.astype(np.float32)


def discover(root: Path, noise_root: Path, max_per_class: int, seed: int) -> tuple[dict[str, list[dict]], list[Path], set[str]]:
    marvin = {speaker(p): p for p in sorted((root / "marvin").glob("*.wav"))}
    marvin_multi: dict[str, list[Path]] = {}
    for p in sorted((root / "marvin").glob("*.wav")):
        marvin_multi.setdefault(speaker(p), []).append(p)
    noises = sorted(p for p in noise_root.rglob("*.wav") if "noise" in p.parts or "ventilation" in p.parts)
    if not marvin or not noises:
        raise RuntimeError(f"Missing GSC Marvin recordings or CAVEMOVE noise under {root} / {noise_root}")
    selected: dict[str, list[dict]] = {k: [] for k in ("train", "val", "test")}
    rng = random.Random(seed)
    speakers = sorted(marvin)
    for split in selected:
        sids = [s for s in speakers if split_for(s) == split]
        for label in COMMANDS:
            paths = sorted((root / label).glob("*.wav"))
            by_spk: dict[str, list[Path]] = {}
            for p in paths:
                sid = speaker(p)
                if sid in sids:
                    by_spk.setdefault(sid, []).append(p)
            for sid in sids:
                picks = by_spk.get(sid, [])
                rng.shuffle(picks)
                for p in picks[:max_per_class]:
                    selected[split].append({"wav": p, "speaker": sid, "label": CLASSES.index(label), "label_name": label, "enroll": marvin[sid]})
        # A couple of same-speaker ordinary words teach the post-wake head to
        # abstain when the user says unrelated speech.
        other_dirs = [p for p in root.iterdir() if p.is_dir() and p.name not in set(COMMANDS) | {"marvin", "_background_noise_"}]
        by_spk_other: dict[str, list[Path]] = {}
        for folder in other_dirs:
            for p in folder.glob("*.wav"):
                sid = speaker(p)
                if sid in sids:
                    by_spk_other.setdefault(sid, []).append(p)
        for sid in sids:
            picks = by_spk_other.get(sid, [])
            rng.shuffle(picks)
            for p in picks[:max(1, min(2, max_per_class))]:
                selected[split].append({"wav": p, "speaker": sid, "label": 8, "label_name": "no_command", "enroll": marvin[sid]})
    eval_speakers = {sid for sid, rows in marvin_multi.items() if len(rows) >= 2 and all(any(speaker(p) == sid for p in (root / c).glob("*.wav")) for c in COMMANDS)}
    # Keep the genuinely independent-enrollment speakers out of all head-fit splits.
    for split in selected:
        selected[split] = [r for r in selected[split] if r["speaker"] not in eval_speakers]
    return selected, noises, eval_speakers


def make_stage1(cfg: dict, paths_cfg: Path, device: torch.device) -> tuple[Stage1System, dict]:
    stage_cfg = load_stage_config(REPO_ROOT / cfg["stage1_config"], paths_cfg)
    stage = Stage1System(
        spex_cfg=stage_cfg["spex"], sample_rate=SR, use_cabin_frontend=True,
        num_mics=4, frontend_cfg=stage_cfg.get("frontend"),
        pretrained_dir=str(stage_cfg["pretrained_checkpoint"]), backend=stage_cfg.get("spex_backend"),
    ).to(device)
    ckpt = Path(stage_cfg["paths"]["checkpoint_root"]) / cfg["checkpoints"]["spex"]
    load_checkpoint(ckpt, stage)
    stage.eval().requires_grad_(False)
    return stage, stage_cfg


@torch.inference_mode()
def extract_features(rows: list[dict], noise_paths: list[Path], stage: Stage1System, wav_encoder: torch.nn.Module,
                     device: torch.device, cache: Path, rng_seed: int, desc: str) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1((desc + str(rng_seed) + str(len(rows))).encode()).hexdigest()[:12]
    feature_file = cache / f"{desc}_{key}.npz"
    if feature_file.exists():
        saved = np.load(feature_file, allow_pickle=False)
        return torch.from_numpy(saved["x"]), torch.from_numpy(saved["y"]), rows
    rng = np.random.default_rng(rng_seed)
    vectors: list[np.ndarray] = []
    ys: list[int] = []
    stage.eval(); wav_encoder.eval()
    for i, row in enumerate(rows):
        enrollment, sr = sf.read(str(row["enroll"]), dtype="float32")
        if enrollment.ndim == 2: enrollment = enrollment.mean(axis=1)
        if sr != SR:
            nn = max(1, round(len(enrollment) * SR / sr))
            enrollment = np.interp(np.linspace(0, 1, nn, endpoint=False), np.linspace(0, 1, len(enrollment), endpoint=False), enrollment).astype(np.float32)
        enrollment = fit(enrollment, int(1.5 * SR))
        mix, mc = mix_clip(row["wav"], noise_paths[int(rng.integers(0, len(noise_paths)))], rng)
        mix_t = torch.from_numpy(mix).unsqueeze(0).to(device)
        mc_t = torch.from_numpy(mc.T).unsqueeze(0).to(device)
        enr_t = torch.from_numpy(enrollment).unsqueeze(0).to(device)
        s1 = stage(mix_t, enr_t, multichannel=mc_t)[0]
        s1 = normalize_extracted_waveform(s1)
        # Stage 4 consumes the first two seconds of its rolling command buffer.
        cmd = s1[..., : 2 * SR]
        cmd = cmd - cmd.mean(dim=-1, keepdim=True)
        cmd = cmd / cmd.std(dim=-1, keepdim=True).clamp_min(1e-5)
        states = wav_encoder(cmd).last_hidden_state
        vectors.append(states.mean(dim=1).squeeze(0).float().cpu().numpy())
        ys.append(int(row["label"]))
        if (i + 1) % 100 == 0 or i + 1 == len(rows):
            print(f"{desc}: frozen Stage-1 + encoder features {i+1}/{len(rows)}", flush=True)
    x = np.asarray(vectors, np.float32); y = np.asarray(ys, np.int64)
    np.savez_compressed(feature_file, x=x, y=y)
    return torch.from_numpy(x), torch.from_numpy(y), rows


def scores(logits: torch.Tensor, labels: torch.Tensor) -> tuple[dict, np.ndarray]:
    pred = logits.argmax(1).cpu().numpy(); truth = labels.cpu().numpy()
    cm = np.zeros((len(CLASSES), len(CLASSES)), np.int64)
    np.add.at(cm, (truth, pred), 1)
    tp = np.diag(cm).astype(float); prec = tp / np.maximum(cm.sum(0), 1); rec = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return {"accuracy": float(tp.sum() / max(1, cm.sum())), "macro_f1": float(f1.mean()),
            "weighted_f1": float((f1 * cm.sum(1) / max(1, cm.sum())).sum()),
            **{f"f1/{CLASSES[i]}": float(f1[i]) for i in range(len(CLASSES))}}, cm


def calibrate_abstention(model: Wav2VecIntentHead, xval: torch.Tensor, yval: torch.Tensor) -> tuple[float,list[dict]]:
    """Choose a confidence cutoff on held-out speakers, maximizing macro-F1."""
    model.eval()
    with torch.inference_mode():
        probabilities=torch.softmax(model(xval.cuda()),dim=-1).cpu()
    rows=[]
    best=(-1.0,1.0)
    for threshold in np.linspace(0.0,0.95,20):
        pred=probabilities.argmax(1)
        pred=torch.where(probabilities.max(1).values < float(threshold),torch.full_like(pred,8),pred)
        proxy_logits=torch.full((len(pred),len(CLASSES)),-10.0)
        proxy_logits.scatter_(1,pred[:,None],10.0)
        metrics,_=scores(proxy_logits,yval)
        no_command=(yval==8)
        false_actions=int(((pred<8)&no_command).sum())
        command_count=max(1,int((yval<8).sum()))
        exact_command_accuracy=float(((pred==yval)&(yval<8)).sum())/command_count
        false_action_rate=false_actions/max(1,int(no_command.sum()))
        row={"threshold":float(threshold),"macro_f1":metrics["macro_f1"],"accuracy":metrics["accuracy"],
             "command_exact_accuracy":exact_command_accuracy,"no_command_false_action_rate":false_action_rate}
        rows.append(row)
        candidate=(metrics["macro_f1"],-false_action_rate)
        if candidate>best:
            best=candidate; selected=float(threshold)
    return selected,rows


def train_head(xtr: torch.Tensor, ytr: torch.Tensor, xval: torch.Tensor, yval: torch.Tensor,
               output: Path, run: wandb.sdk.wandb_run.Run, epochs: int, batch: int, seed: int,
               metric_prefix: str = "isolated_head", step_offset: int = 0) -> tuple[Wav2VecIntentHead, dict, np.ndarray]:
    torch.manual_seed(seed)
    model = Wav2VecIntentHead(input_dim=xtr.shape[1], hidden=0, num_classes=len(CLASSES)).to("cuda")
    counts = torch.bincount(ytr, minlength=len(CLASSES)).float()
    weights = 1 / counts.clamp_min(1)
    opt = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=1e-3)
    best = -1.0; best_metrics = {}; best_cm = np.zeros((9,9), np.int64); bad = 0
    gen = torch.Generator().manual_seed(seed)
    ds = TensorDataset(xtr, ytr)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, generator=gen)
    for epoch in range(1, epochs + 1):
        model.train(); total_loss = 0; total_n = 0
        for xx, yy in loader:
            xx, yy = xx.cuda(non_blocking=True), yy.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xx)
            loss = F.cross_entropy(logits, yy, weight=weights.cuda(), label_smoothing=0.02)
            loss.backward(); opt.step()
            total_loss += float(loss.detach()) * len(yy); total_n += len(yy)
        model.eval()
        with torch.no_grad():
            vl = model(xval.cuda())
        metric, cm = scores(vl, yval)
        row = {"epoch": epoch, f"{metric_prefix}/train_loss": total_loss / max(1, total_n),
               f"{metric_prefix}/val_loss": float(F.cross_entropy(vl, yval.cuda()).item()),
               **{f"{metric_prefix}/val_{k}": v for k,v in metric.items()}, "learning_rate": opt.param_groups[0]["lr"]}
        run.log(row, step=step_offset + epoch)
        print(f"{metric_prefix} epoch {epoch}/{epochs}: train loss {row[f'{metric_prefix}/train_loss']:.4f}, val acc {metric['accuracy']:.3f}, val macro-F1 {metric['macro_f1']:.3f}", flush=True)
        if metric["macro_f1"] > best:
            best = metric["macro_f1"]; bad = 0; best_metrics = metric; best_cm = cm
            save_checkpoint(output, model, epoch, {"metrics": metric, "classes": CLASSES, "head_hidden": 0})
        else:
            bad += 1
            if bad >= 5: break
    return model, best_metrics, best_cm


def read_16k(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32")
    if x.ndim == 2: x = x.mean(axis=1)
    if sr != SR:
        n = max(1, round(len(x) * SR / sr))
        x = np.interp(np.linspace(0, 1, n, endpoint=False), np.linspace(0, 1, len(x), endpoint=False), x).astype(np.float32)
    return np.asarray(x, np.float32)


def select_runtime_scenes(rows: list[dict], marvin_by_speaker: dict[str, list[Path]], maximum_speakers: int,
                          seed: int) -> list[tuple[str, str, Path, Path, Path | None, Path]]:
    """Select speaker-disjoint wake+command, wake+no-command, and no-wake scenes."""
    by_speaker: dict[str, list[dict]] = {}
    for row in rows: by_speaker.setdefault(row["speaker"], []).append(row)
    candidates=[]
    for sid, entries in sorted(by_speaker.items()):
        if sid not in marvin_by_speaker: continue
        commands=[r for r in entries if int(r["label"]) < 8]
        noncommands=[r for r in entries if int(r["label"]) == 8]
        if commands: candidates.append((sid,commands,noncommands))
    random.Random(seed).shuffle(candidates)
    candidates=candidates[:max(0,maximum_speakers)]
    speaker_ids=[x[0] for x in candidates]
    scenes=[]
    for i,(sid,commands,noncommands) in enumerate(candidates):
        marv=marvin_by_speaker[sid]
        wake=marv[1] if len(marv)>1 else marv[0]
        enroll=marv[0]
        other=speaker_ids[(i+1)%len(speaker_ids)] if len(speaker_ids)>1 else sid
        passenger=marvin_by_speaker[other][0]
        commands=sorted(commands,key=lambda r:(int(r["label"]),str(r["wav"])))
        # Rotate the selected class across speakers to keep the run balanced.
        desired=int(hashlib.sha1(sid.encode()).hexdigest()[:8],16)%8
        chosen=next((r for r in commands if int(r["label"])==desired),commands[i%len(commands)])
        scenes.append((sid,str(chosen["label_name"]),wake,enroll,Path(chosen["wav"]),passenger))
        if noncommands:
            chosen_neg=sorted(noncommands,key=lambda r:str(r["wav"]))[i%len(noncommands)]
            scenes.append((sid,"no_command",wake,enroll,Path(chosen_neg["wav"]),passenger))
        scenes.append((sid,"no_wake",passenger,enroll,None,passenger))
    return scenes


@torch.inference_mode()
def collect_runtime_captures(rows: list[dict], split: str, marvin_by_speaker: dict[str,list[Path]],
                             noise_paths: list[Path], pipeline: InCarPipeline, device: torch.device,
                             cache_dir: Path, seed: int, maximum_speakers: int) -> tuple[np.ndarray,np.ndarray,list[dict]]:
    """Capture the exact tensors sent to Stage 4 after real Stage 1-3 decisions."""
    threshold_key=f"p{pipeline.presence_threshold:.3f}_c{pipeline.command_presence_threshold:.3f}_k{pipeline.kws_threshold:.3f}"
    cache_key=hashlib.sha1(f"{split}|{seed}|{maximum_speakers}|{threshold_key}|buffer-v3".encode()).hexdigest()[:12]
    cache_path=cache_dir/f"runtime_{split}_{cache_key}.npz"
    manifest_path=cache_dir/f"runtime_{split}_{cache_key}.jsonl"
    if cache_path.exists() and manifest_path.exists():
        data=np.load(cache_path,allow_pickle=False)
        return data["audio"],data["labels"], [json.loads(s) for s in manifest_path.read_text(encoding="utf-8").splitlines() if s]
    scene_rows=select_runtime_scenes(rows,marvin_by_speaker,maximum_speakers,seed)
    rng=np.random.default_rng(seed); audio_rows=[]; labels=[]; metadata=[]
    for si,(sid,label,wake_path,enroll_path,command_path,passenger_path) in enumerate(scene_rows):
        wake=fit(read_16k(wake_path),SR); passenger=fit(read_16k(passenger_path),SR)
        command=fit(read_16k(command_path),SR) if command_path is not None else np.zeros(SR,np.float32)
        enroll=fit(read_16k(enroll_path),int(1.5*SR))
        duration=9*SR; speech=np.zeros(duration,np.float32)
        if label!="no_wake":
            speech[SR:2*SR]=wake
            speech[2*SR:3*SR]=command
        speech[5*SR:6*SR]=passenger
        noise=load_noise(noise_paths[int(rng.integers(0,len(noise_paths)))],rng)
        noise=np.tile(noise,(int(np.ceil(duration/len(noise))),1))[:duration]
        snr=float(rng.choice([-5,0,5,10])) if split=="train" else -5.0
        active=speech[5*SR:6*SR] if label=="no_wake" else speech[:3*SR]
        speech_rms=float(np.sqrt(np.mean(active**2)+1e-12)); nrms=float(np.sqrt(np.mean(noise**2)+1e-12))
        noise=noise*(speech_rms*10**(-snr/20)/max(nrms,1e-8))
        multichannel=speech[:,None]+noise; mixture=multichannel.mean(axis=1).astype(np.float32)
        state=PipelineState(); captures=[]; original=pipeline._infer_command
        def capture_stage4(command_state,presence_probability,speaker_similarity,s1):
            if command_state.command_audio is not None:
                exact=pipeline._prepare_intent_input(command_state.command_audio).detach().cpu().squeeze(0).numpy().astype(np.float32)
                start=command_state.command_capture_start_sample
                result=original(command_state,presence_probability,speaker_similarity,s1)
                pred=int(result.intent_logits.argmax(-1).item()) if result.intent_logits is not None else -1
                captures.append((exact,start,pred))
                return result
            return original(command_state,presence_probability,speaker_similarity,s1)
        pipeline._infer_command=capture_stage4
        try:
            for start in range(0,duration,SR):
                end=min(duration,start+3*SR); w=np.zeros(3*SR,np.float32); mc=np.zeros((4,3*SR),np.float32)
                w[:end-start]=mixture[start:end]; mc[:,:end-start]=multichannel[start:end].T
                pipeline.forward(torch.from_numpy(w).to(device),torch.from_numpy(enroll).to(device),state,
                                 multichannel=torch.from_numpy(mc).to(device),audio_start_sample=start)
        finally:
            pipeline._infer_command=original
        for ci,(wav,start_sample,baseline_pred) in enumerate(captures):
            overlap=0.0
            if start_sample is not None and label not in ("no_command","no_wake"):
                overlap=max(0,min(int(start_sample)+len(wav),3*SR)-max(int(start_sample),2*SR))/SR
            assigned=label if label in COMMANDS and overlap>=0.5 else "no_command"
            audio_rows.append(wav); labels.append(CLASSES.index(assigned))
            metadata.append({"split":split,"speaker_id":sid,"scene_label":label,"train_label":assigned,
                             "stage4_source_start_seconds":None if start_sample is None else float(start_sample)/SR,
                             "target_command_overlap_seconds":overlap,"baseline_head_prediction":CLASSES[baseline_pred] if 0<=baseline_pred<len(CLASSES) else None,
                             "noise_snr_db":snr,"sample_index":len(labels)-1})
        if (si+1)%50==0 or si+1==len(scene_rows):
            print(f"runtime capture {split}: scenes {si+1}/{len(scene_rows)}, Stage-4 examples {len(audio_rows)}",flush=True)
    if not audio_rows: raise RuntimeError(f"No Stage-4 captures were emitted for runtime {split}; adjust the gate settings")
    audio_arr=np.asarray(audio_rows,np.float32); label_arr=np.asarray(labels,np.int64)
    cache_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache_path,audio=audio_arr,labels=label_arr)
    manifest_path.write_text("\n".join(json.dumps(row) for row in metadata)+"\n",encoding="utf-8")
    return audio_arr,label_arr,metadata


@torch.inference_mode()
def waveform_features(waveforms: np.ndarray, encoder: torch.nn.Module, device: torch.device) -> torch.Tensor:
    features=[]
    for offset in range(0,len(waveforms),64):
        wav=torch.from_numpy(waveforms[offset:offset+64]).to(device)
        wav=wav-wav.mean(dim=1,keepdim=True)
        wav=wav/wav.std(dim=1,keepdim=True).clamp_min(1e-5)
        features.append(encoder(wav).last_hidden_state.mean(dim=1).float().cpu())
    return torch.cat(features,dim=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, default=REPO_ROOT / "configs/paths.e_research.yaml")
    parser.add_argument("--pipeline-config", type=Path, default=REPO_ROOT / "configs/eval_continuous_pipeline_p0_e.yaml")
    parser.add_argument("--root", type=Path, default=Path("E:/car-tse-data/speech_commands_v2"))
    parser.add_argument("--noise-root", type=Path, default=Path("E:/car-tse-data/cavemove/extracted/cavemove_dataset"))
    parser.add_argument("--max-per-class", type=int, default=3)
    parser.add_argument("--runtime-train-speakers", type=int, default=400)
    parser.add_argument("--runtime-val-speakers", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--presence-threshold", type=float, default=None, help="Override Stage-2 cutoff for controlled diagnostics")
    parser.add_argument("--command-presence-threshold", type=float, default=None, help="Override post-wake command gate for controlled diagnostics")
    parser.add_argument("--kws-threshold", type=float, default=None, help="Override Stage-3 cutoff for controlled diagnostics")
    parser.add_argument("--capture-presence-threshold", type=float, default=None, help="Temporary Stage-2 cutoff used only to collect Stage-4 training inputs")
    parser.add_argument("--capture-command-presence-threshold", type=float, default=None, help="Temporary post-wake cutoff used only to collect Stage-4 training inputs")
    parser.add_argument("--intent-action-threshold", type=float, default=None,
                        help="Use a confidence cutoff selected on validation (e.g. a safer, higher-precision operating point)")
    parser.add_argument("--wandb-project", default="car-tse")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if not torch.cuda.is_available(): raise RuntimeError("audioml_base did not expose CUDA")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    cfg = load_stage_config(args.pipeline_config, args.paths)
    all_rows, noises, eval_speakers = discover(args.root, args.noise_root, args.max_per_class, args.seed)
    count_by_split = {k: len(v) for k,v in all_rows.items()}
    if not all_rows["train"] or not all_rows["val"]: raise RuntimeError(f"Insufficient speaker-disjoint data: {count_by_split}")
    out_dir = REPO_ROOT / "checkpoints" / "stage4_gsc_command_linear"
    cache_dir = REPO_ROOT / "tmp" / "gsc_command_linear_cache"
    manifest_dir = REPO_ROOT / "tmp" / "gsc_command_linear_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in all_rows.items():
        with (manifest_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps({"wav": str(row["wav"]), "speaker_id": row["speaker"],
                                         "enrollment_wav": str(row["enroll"]), "label": int(row["label"]),
                                         "label_name": row["label_name"]}) + "\n")
    run_name=(f"gsc-runtime-trained-linear-head-stage2-{args.presence_threshold if args.presence_threshold is not None else cfg.get('presence_threshold',.9):.2f}"
              f"-command-{args.command_presence_threshold if args.command_presence_threshold is not None else cfg.get('command_presence_threshold',.9):.2f}"
              f"-abstain-{args.intent_action_threshold if args.intent_action_threshold is not None else 'valcal'}-seed-{args.seed}")
    run = wandb.init(project=args.wandb_project, name=run_name, job_type="train-and-pipeline-eval",
                     mode=args.wandb_mode, config={"task": "8 GSC command words + no_command", "frozen_stages": [1,2,3,"wav2vec encoder"],
                     "train_examples": count_by_split["train"], "validation_examples": count_by_split["val"],
                     "reserved_independent_enrollment_test_speakers": len(eval_speakers), "max_examples_per_speaker_class": args.max_per_class,
                     "head": "single Linear(768,9), no hidden layer", "head_parameters": 768*9+9,
                     "command_capture_preroll_seconds": 0.25,
                     "command_capture_start_rule": "end of earliest positive supporting KWS crop minus pre-roll",
                     "runtime_capture_train_speakers": args.runtime_train_speakers,
                     "runtime_capture_validation_speakers": args.runtime_val_speakers,
                     "sample_rate": SR, "stage1_config": cfg["stage1_config"], "stage1_checkpoint": cfg["checkpoints"]["spex"],
                     "stage2_checkpoint": cfg["checkpoints"]["presence"], "stage3_checkpoint": cfg["checkpoints"]["kws"],
                     "diagnostic_presence_threshold": args.presence_threshold if args.presence_threshold is not None else cfg.get("presence_threshold", .9),
                     "diagnostic_command_presence_threshold": args.command_presence_threshold if args.command_presence_threshold is not None else cfg.get("command_presence_threshold"),
                     "diagnostic_kws_threshold": args.kws_threshold if args.kws_threshold is not None else cfg.get("kws_threshold", .29),
                     "runtime_capture_presence_threshold": args.capture_presence_threshold,
                     "runtime_capture_command_presence_threshold": args.capture_command_presence_threshold,
                     "note": "Synthetic continuous replay from same-speaker GSC clips with CAVEMOVE noise; pilot, not natural in-car benchmark."})
    try:
        manifest_artifact = wandb.Artifact("gsc-command-speaker-disjoint-manifests", type="dataset-manifest",
                                           metadata={"labels": list(CLASSES), "speaker_disjoint": True,
                                                     "reserved_full_pipeline_speakers": len(eval_speakers)})
        for split in all_rows:
            manifest_artifact.add_file(str(manifest_dir / f"{split}.jsonl"), name=f"{split}.jsonl")
        run.log_artifact(manifest_artifact)
        stage1, stage1_cfg = make_stage1(cfg, args.paths, device)
        encoder_path = Path(cfg["intent_encoder"])
        encoder_ckpt = Path(stage1_cfg["paths"]["checkpoint_root"]) / cfg["intent_encoder_checkpoint"]
        encoder = AutoModel.from_pretrained(str(encoder_path)).to(device)
        payload = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)
        encoder.load_state_dict(payload["model_state_dict"], strict=True)
        encoder.eval().requires_grad_(False)
        run.log({"data/train_examples": count_by_split["train"], "data/validation_examples": count_by_split["val"],
                 "data/test_examples": count_by_split["test"], "data/reserved_full_pipeline_speakers": len(eval_speakers)}, step=0)
        xtr, ytr, _ = extract_features(all_rows["train"], noises, stage1, encoder, device, cache_dir, args.seed, "train")
        xval, yval, _ = extract_features(all_rows["val"], noises, stage1, encoder, device, cache_dir, args.seed+1, "val")
        model, best_metrics, cm = train_head(xtr, ytr, xval, yval, out_dir / "best.pt", run, args.epochs, args.batch_size, args.seed)
        run.log({"best_validation/"+k:v for k,v in best_metrics.items()})
        run.log({"validation_confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=yval.numpy().tolist(), preds=model(xval.cuda()).argmax(1).cpu().tolist(), class_names=list(CLASSES))})
        # Evaluate speaker-held-out head data, not used for selection.
        if all_rows["test"]:
            xt, yt, _ = extract_features(all_rows["test"], noises, stage1, encoder, device, cache_dir, args.seed+2, "test")
            model.eval()
            with torch.no_grad(): test_logits=model(xt.cuda())
            test_metrics, test_cm=scores(test_logits,yt)
            run.log({"heldout_test/"+k:v for k,v in test_metrics.items()})
            np.savetxt(REPO_ROOT/"reports"/"evaluation"/"stage4_gsc_command_test_confusion.csv",test_cm,delimiter=",",fmt="%d")
        else:
            test_metrics = {"note": "no independent test rows after speaker reservation"}
        # Reload exact saved linear head into the full Stage 1 -> Stage 2 -> Stage 3 -> Stage 4 runtime.
        head_ckpt = out_dir / "best.pt"
        encoder_payload_copy = encoder_ckpt
        presence_path = Path(stage1_cfg["paths"]["checkpoint_root"]) / cfg["checkpoints"]["presence"]
        kws_path = Path(stage1_cfg["paths"]["checkpoint_root"]) / cfg["checkpoints"]["kws"]
        presence = FramePresenceHead(sample_rate=SR,n_mels=40,hidden=64,n_fft=512,hop=160).to(device)
        load_checkpoint(presence_path,presence); presence.eval()
        kws=KWSModel(sample_rate=SR,num_classes=2).to(device); load_checkpoint(kws_path,kws); kws.eval()
        load_checkpoint(head_ckpt,model)
        label_map={str(i):(name if name!="no_command" else "noop") for i,name in enumerate(CLASSES)}
        pipeline=InCarPipeline(stage1,presence,kws, __import__('car_tse.models.wav2vec_intent',fromlist=['Wav2VecIntentClassifier']).Wav2VecIntentClassifier(
            str(encoder_path),encoder_payload_copy,head_ckpt,num_classes=len(CLASSES),input_sample_rate=SR,input_seconds=2.0,head_hidden=0),
            presence_threshold=float(args.presence_threshold if args.presence_threshold is not None else cfg.get("presence_threshold",.9)),
            kws_threshold=float(args.kws_threshold if args.kws_threshold is not None else cfg.get("kws_threshold",.29)),
            kws_smoothing_window=int(cfg.get("kws_smoothing_window",3)),kws_required_hits=int(cfg.get("kws_required_hits",2)),
            kws_refractory_calls=int(cfg.get("kws_refractory_calls",2)),wake_class_id=1,kws_sample_rate=SR,kws_input_seconds=1.0,
            intent_sample_rate=SR,intent_input_seconds=2.0,command_timeout_calls=4,command_end_silence_calls=2,
            command_min_speech_seconds=.25,
            command_presence_threshold=(args.command_presence_threshold if args.command_presence_threshold is not None else cfg.get("command_presence_threshold")),
            label_map=label_map,device=device)
        marvin_by_spk: dict[str,list[Path]]={}
        for p in sorted((args.root/"marvin").glob("*.wav")): marvin_by_spk.setdefault(speaker(p),[]).append(p)
        runtime_cache=REPO_ROOT/"tmp"/"gsc_runtime_capture_cache"
        eval_presence_threshold=pipeline.presence_threshold
        eval_command_presence_threshold=pipeline.command_presence_threshold
        if args.capture_presence_threshold is not None:
            pipeline.presence_threshold=float(args.capture_presence_threshold)
        if args.capture_command_presence_threshold is not None:
            pipeline.command_presence_threshold=float(args.capture_command_presence_threshold)
        rt_train_audio,rt_train_y,rt_train_manifest=collect_runtime_captures(
            all_rows["train"],"train",marvin_by_spk,noises,pipeline,device,runtime_cache,args.seed+11,args.runtime_train_speakers)
        rt_val_audio,rt_val_y,rt_val_manifest=collect_runtime_captures(
            all_rows["val"],"val",marvin_by_spk,noises,pipeline,device,runtime_cache,args.seed+12,args.runtime_val_speakers)
        pipeline.presence_threshold=eval_presence_threshold
        pipeline.command_presence_threshold=eval_command_presence_threshold
        rt_xtrain=waveform_features(rt_train_audio,pipeline.intent.encoder,device)
        rt_xval=waveform_features(rt_val_audio,pipeline.intent.encoder,device)
        runtime_head,runtime_val_metrics,runtime_val_cm=train_head(
            rt_xtrain,torch.from_numpy(rt_train_y),rt_xval,torch.from_numpy(rt_val_y),
            REPO_ROOT/"checkpoints"/"stage4_gsc_runtime_capture"/"best.pt",run,args.epochs,args.batch_size,args.seed+1,
            metric_prefix="runtime_capture_head",step_offset=100)
        pipeline.intent.head.load_state_dict(runtime_head.state_dict(),strict=True)
        calibrated_abstention_threshold,abstention_sweep=calibrate_abstention(runtime_head,rt_xval,torch.from_numpy(rt_val_y))
        abstention_threshold=(float(args.intent_action_threshold) if args.intent_action_threshold is not None
                              else calibrated_abstention_threshold)
        pipeline.intent_action_threshold=abstention_threshold
        run.config.update({"calibrated_stage4_abstention_threshold":calibrated_abstention_threshold,
                           "applied_stage4_abstention_threshold":abstention_threshold},allow_val_change=True)
        run.log({"runtime_capture/data/train_examples":len(rt_train_y),"runtime_capture/data/validation_examples":len(rt_val_y),
                 "runtime_capture/validation_macro_f1":runtime_val_metrics["macro_f1"],
                 "runtime_capture/selected_abstention_threshold":abstention_threshold},step=150)
        threshold_table=wandb.Table(columns=list(abstention_sweep[0]))
        for row in abstention_sweep: threshold_table.add_data(*[row[k] for k in abstention_sweep[0]])
        run.log({"runtime_capture/abstention_threshold_sweep":threshold_table})
        runtime_manifest_path=runtime_cache/"runtime_capture_manifest_run.jsonl"
        runtime_manifest_path.write_text("\n".join(json.dumps(x) for x in rt_train_manifest+rt_val_manifest)+"\n",encoding="utf-8")
        runtime_manifest_artifact=wandb.Artifact("gsc-stage4-runtime-capture-manifest",type="dataset-manifest",
                                                 metadata={"train_capture_count":len(rt_train_y),"val_capture_count":len(rt_val_y),
                                                           "label_rule":"command label only when exact Stage-4 input overlaps at least 0.5 seconds of the target word; otherwise no_command"})
        runtime_manifest_artifact.add_file(str(runtime_manifest_path)); run.log_artifact(runtime_manifest_artifact)
        run.log({"runtime_capture/validation_confusion_matrix":wandb.plot.confusion_matrix(
            probs=None,y_true=rt_val_y.tolist(),preds=runtime_head(rt_xval.cuda()).argmax(1).cpu().tolist(),class_names=list(CLASSES))})
        # Independent enrollment is mandatory here. Use held-out speakers with
        # at least two Marvin recordings; one scene per speaker keeps this
        # first complete-cascade pass fast and limits repeated-speaker bias.
        marvin_by_spk: dict[str,list[Path]]={}
        for p in sorted((args.root/"marvin").glob("*.wav")): marvin_by_spk.setdefault(speaker(p),[]).append(p)
        full_rows=[]; rng=np.random.default_rng(args.seed+3)
        target_words={c:sorted((args.root/c).glob("*.wav")) for c in COMMANDS}
        noncommand_dirs=[p for p in args.root.iterdir() if p.is_dir() and p.name not in set(COMMANDS)|{"marvin","_background_noise_"}]
        noncommand_by_speaker: dict[str,list[Path]]={}
        for folder in noncommand_dirs:
            for p in folder.glob("*.wav"):
                noncommand_by_speaker.setdefault(speaker(p),[]).append(p)
        eligible=sorted(eval_speakers)
        for j,sid in enumerate(eligible):
            marv=marvin_by_spk[sid]
            if len(marv)<2: continue
            wake_path, enroll_path=marv[0],marv[1]
            label=COMMANDS[j%len(COMMANDS)]
            candidates=[p for p in target_words[label] if speaker(p)==sid]
            if not candidates: continue
            full_rows.append((sid,label,wake_path,enroll_path,candidates[0]))
            unrelated=sorted(noncommand_by_speaker.get(sid,[]))
            if unrelated:
                # A post-wake unrelated word is the essential negative case:
                # Stage 4 should return no_command instead of an action.
                full_rows.append((sid,"no_command",wake_path,enroll_path,unrelated[0]))
            other_speaker=eligible[(j+1)%len(eligible)] if len(eligible)>1 else sid
            other_wake=marvin_by_spk[other_speaker][0]
            full_rows.append((sid,"no_wake",other_wake,enroll_path,unrelated[0] if unrelated else cmd_path))
        # Small held-out evaluation includes all classes; replay at actual
        # runtime hop/configuration with the existing Stage 2/3 thresholds.
        full_results=[]
        audio_artifact_dir=REPO_ROOT/"reports"/"evaluation"/"gsc_full_pipeline_stage4_inputs"
        audio_artifact_dir.mkdir(parents=True,exist_ok=True)
        for j,(sid,label,wake_path,enroll_path,cmd_path) in enumerate(full_rows):
            def read1(p):
                z,sr=sf.read(str(p),dtype="float32"); z=z.mean(1) if z.ndim==2 else z
                if sr!=SR:
                    nn=max(1,round(len(z)*SR/sr)); z=np.interp(np.linspace(0,1,nn,endpoint=False),np.linspace(0,1,len(z),endpoint=False),z).astype(np.float32)
                return z
            wake=fit(read1(wake_path),SR); command=fit(read1(cmd_path),SR)
            passenger=fit(read1(marvin_by_spk[eligible[(j+1)%len(eligible)]][0]),SR) if len(eligible)>1 else np.zeros(SR,np.float32)
            enroll=fit(read1(enroll_path),int(1.5*SR))
            duration=9*SR; speech=np.zeros(duration,np.float32)
            if label!="no_wake":
                speech[SR:2*SR]=wake
                speech[2*SR:3*SR]=command
            speech[5*SR:6*SR]=passenger
            noise=load_noise(noises[j%len(noises)],rng)
            reps=int(np.ceil(duration/len(noise))); noise=np.tile(noise,(reps,1))[:duration]
            speech_rms=float(np.sqrt(np.mean(speech[:3*SR]**2)+1e-12)); nrms=float(np.sqrt(np.mean(noise**2)+1e-12))
            noise=noise*(speech_rms*10**(-5/20)/max(nrms,1e-8)); mc=speech[:,None]+noise; mono=mc.mean(1).astype(np.float32)
            state=PipelineState(); wake_hit=False; pred=None; confidence=None; actions=[]; presence_values=[]; wake_values=[]; accepted_windows=0; stage4_captures=[]
            original_infer=pipeline._infer_command
            def capture_exact_stage4(command_state,presence_probability,speaker_similarity,s1):
                if command_state.command_audio is not None:
                    exact=pipeline._prepare_intent_input(command_state.command_audio).detach().cpu().squeeze(0).numpy().astype(np.float32)
                    start_sample=command_state.command_capture_start_sample
                    stage4_captures.append({"wav":exact,"start_sample":start_sample,"buffer_samples":int(command_state.command_audio.shape[-1])})
                return original_infer(command_state,presence_probability,speaker_similarity,s1)
            pipeline._infer_command=capture_exact_stage4
            hop=SR
            for start in range(0,duration,hop):
                end=min(duration,start+3*SR); w=np.zeros(3*SR,np.float32); m=np.zeros((4,3*SR),np.float32)
                if end>start: w[:end-start]=mono[start:end]; m[:,:end-start]=mc[start:end].T
                with torch.inference_mode(): out,state=pipeline.forward(torch.from_numpy(w).to(device),torch.from_numpy(enroll).to(device),state,multichannel=torch.from_numpy(m).to(device),audio_start_sample=start)
                wake_hit |= bool(out.wake_detected)
                accepted_windows += int(bool(out.accepted))
                if out.presence_probability is not None: presence_values.append(float(out.presence_probability))
                if out.wake_probability is not None: wake_values.append(float(out.wake_probability))
                if out.intent_ready and out.action is not None:
                    confidence=out.intent_confidence
                    if out.action is not None and str(out.action.intent_id).isdigit():
                        selected_class=int(out.action.intent_id)
                        pred=CLASSES[selected_class] if 0<=selected_class<len(CLASSES) else None
                    elif out.intent_logits is not None:
                        pred=CLASSES[int(out.intent_logits.argmax(dim=-1).item())]
                    else:
                        pred=label_map.get(str(out.action.intent_id),None)
                    actions.append(pred)
            pipeline._infer_command=original_infer
            captured_meta=[]
            for ci,capture in enumerate(stage4_captures):
                x=capture["wav"]; start_sample=capture["start_sample"]
                path=audio_artifact_dir/f"scene_{j:03d}_capture_{ci:02d}_{label}.wav"
                sf.write(str(path),x,SR,subtype="FLOAT")
                if start_sample is None:
                    overlap=None; start_seconds=None
                else:
                    start_seconds=float(start_sample)/SR
                    used=len(x)
                    overlap=max(0,min(start_sample+used,3*SR)-max(start_sample,2*SR))/SR
                captured_meta.append({"wav":str(path.relative_to(REPO_ROOT)),"source_start_seconds":start_seconds,
                                      "stage4_input_seconds":len(x)/SR,"target_command_overlap_seconds":overlap,
                                      "mapping_is_exact_for_no_skipped_gate":bool(args.presence_threshold==0.0 and accepted_windows==9)})
            correct=(pred==label) or (label=="no_wake" and pred in (None,"no_command"))
            full_results.append({"speaker":sid,"expected":label,"predicted":pred,"wake_detected":wake_hit,"intent_correct":correct,
                                 "action_count":len(actions),"false_action":bool(label in ("no_command","no_wake") and pred not in (None,"no_command")),
                                 "max_presence_probability":max(presence_values,default=None),"max_wake_probability":max(wake_values,default=None),
                                 "presence_accepted_windows":accepted_windows,"snr_db":-5,
                                 "presence_threshold":pipeline.presence_threshold,"kws_threshold":pipeline.kws_threshold,
                                 "stage4_abstention_threshold":pipeline.intent_action_threshold,
                                 "stage4_confidence":confidence,
                                 "stage4_captures":captured_meta})
            print(f"full cascade {j+1}/{len(full_rows)} speaker={sid} wake={wake_hit} expected={label} pred={pred}",flush=True)
        if full_results:
            positive_rows=[r for r in full_results if r["expected"] in COMMANDS]
            no_command_rows=[r for r in full_results if r["expected"]=="no_command"]
            wake_rows=[r for r in full_results if r["expected"] in COMMANDS or r["expected"]=="no_command"]
            no_wake_rows=[r for r in full_results if r["expected"]=="no_wake"]
            wake_recall=sum(r["wake_detected"] for r in wake_rows)/max(1,len(wake_rows))
            false_wake_rate=sum(r["wake_detected"] for r in no_wake_rows)/max(1,len(no_wake_rows))
            command_acc=sum(r["intent_correct"] for r in positive_rows)/max(1,len(positive_rows))
            no_command_recall=sum(not r["false_action"] for r in no_command_rows)/max(1,len(no_command_rows))
            false_action_rate=sum(r["false_action"] for r in no_command_rows)/max(1,len(no_command_rows))
            no_wake_false_action_rate=sum(r["false_action"] for r in no_wake_rows)/max(1,len(no_wake_rows))
            command_acc_if_action=sum(r["intent_correct"] for r in positive_rows if r["action_count"])/max(1,sum(bool(r["action_count"]) for r in positive_rows))
            run.log({"full_pipeline/scenes":len(full_results),"full_pipeline/target_wake_recall":wake_recall,
                     "full_pipeline/no_wake_false_wake_rate":false_wake_rate,"full_pipeline/target_command_accuracy":command_acc,
                     "full_pipeline/no_command_safe_rejection_rate":no_command_recall,"full_pipeline/no_command_false_action_rate":false_action_rate,
                     "full_pipeline/no_wake_false_action_rate":no_wake_false_action_rate,
                     "full_pipeline/target_command_accuracy_when_action_emitted":command_acc_if_action,
                     "full_pipeline/actions_emitted":sum(r["action_count"] for r in full_results),
                     "full_pipeline/scenes_with_no_action":sum(not r["action_count"] for r in full_results)})
            table=wandb.Table(columns=list(full_results[0]))
            for row in full_results: table.add_data(*[row[k] for k in full_results[0]])
            run.log({"full_pipeline/heldout_same_speaker_scenes":table})
            report={"training_examples":count_by_split["train"],"validation_examples":count_by_split["val"],"heldout_examples":count_by_split["test"],
                    "reserved_full_pipeline_speakers":len(eval_speakers),"head_validation_best":best_metrics,"head_test":test_metrics,
                    "runtime_capture_validation_best":runtime_val_metrics,"runtime_capture_train_examples":len(rt_train_y),
                    "runtime_capture_validation_examples":len(rt_val_y),"selected_abstention_threshold":abstention_threshold,
                    "full_pipeline_summary":{"target_wake_recall":wake_recall,"no_wake_false_wake_rate":false_wake_rate,
                        "target_command_accuracy":command_acc,"no_command_safe_rejection_rate":no_command_recall,
                        "no_command_false_action_rate":false_action_rate,"no_wake_false_action_rate":no_wake_false_action_rate,
                        "target_command_accuracy_when_action_emitted":command_acc_if_action},
                    "full_pipeline":full_results,
                    "warning":"Small synthetic proxy: same GSC speakers for wake/enroll/command, different held-out passenger, CAVEMOVE noise; not a natural in-car benchmark."}
            report_path=REPO_ROOT/"reports"/"evaluation"/"gsc_command_full_cascade_report.json"; report_path.parent.mkdir(parents=True,exist_ok=True); report_path.write_text(json.dumps(report,indent=2),encoding="utf-8")
            artifact=wandb.Artifact("gsc-command-full-cascade-report","evaluation-report"); artifact.add_file(str(report_path)); run.log_artifact(artifact)
            if any(r["stage4_captures"] for r in full_results):
                input_artifact=wandb.Artifact("gsc-command-stage4-runtime-inputs","stage4-audio-inputs")
                for result in full_results:
                    for capture in result["stage4_captures"]:
                        input_artifact.add_file(str(REPO_ROOT/capture["wav"]),name=Path(capture["wav"]).name)
                run.log_artifact(input_artifact)
        run.summary["best_validation_macro_f1"]=best_metrics.get("macro_f1")
        run.summary["head_test_macro_f1"]=test_metrics.get("macro_f1") if "macro_f1" in test_metrics else None
        print(f"Training complete; best validation macro-F1={best_metrics.get('macro_f1')}; test={test_metrics}; full pipeline scenes={len(full_results)}",flush=True)
    finally:
        run.finish()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
