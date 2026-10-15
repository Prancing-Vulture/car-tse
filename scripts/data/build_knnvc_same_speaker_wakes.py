"""Build clean, speaker-matched Marvin wake clips with official kNN-VC.

This is a research-data utility for the English clean benchmark.  It converts
Speech Commands ``marvin`` recordings into each FSC test speaker's voice using
the official kNN-VC implementation installed separately on the D: drive.  It
does not create real-driver recordings, and generated clips must remain marked
as synthetic voice-conversion data in every manifest and report.

The script is intentionally kept separate from the main project's Python
environment because kNN-VC needs a Python/torchaudio combination that is not
available in the repository's Python 3.14 runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--knn-vc-root",
        type=Path,
        default=Path(os.environ.get("KNN_VC_ROOT", r"D:\datasets\knn-vc")),
    )
    parser.add_argument(
        "--wake-manifest",
        type=Path,
        default=Path("manifests/test/stage3_kws_cabin_test.jsonl"),
    )
    parser.add_argument(
        "--intent-manifest",
        type=Path,
        default=Path("manifests/official/stage4_intent_fsc_test.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/e2e_v2_clean_same_speaker/knnvc_wakes"),
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("manifests/clean_e2e_v2/wake_knnvc_same_speaker_test.jsonl"),
    )
    parser.add_argument("--wakes-per-speaker", type=int, default=20)
    parser.add_argument("--references-per-speaker", type=int, default=3)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    wake_manifest = resolve_repo_path(repo_root, args.wake_manifest)
    intent_manifest = resolve_repo_path(repo_root, args.intent_manifest)
    output_dir = resolve_repo_path(repo_root, args.output_dir)
    output_manifest = resolve_repo_path(repo_root, args.output_manifest)
    knn_root = args.knn_vc_root.resolve()
    if not (knn_root / "hubconf.py").exists():
        raise FileNotFoundError(f"kNN-VC hubconf.py not found under {knn_root}")
    if args.wakes_per_speaker <= 0 or args.references_per_speaker <= 0:
        raise ValueError("wakes-per-speaker and references-per-speaker must be positive")

    sys.path.insert(0, str(knn_root))
    import torch  # type: ignore
    import torchaudio  # type: ignore
    import soundfile as sf  # type: ignore

    wake_rows = [row for row in load_jsonl(wake_manifest) if int(row.get("label", 0)) == 1]
    wake_rows.sort(key=lambda row: str(row.get("wav", "")))
    if not wake_rows:
        raise RuntimeError(f"No positive wake rows in {wake_manifest}")
    intent_rows = load_jsonl(intent_manifest)
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for row in intent_rows:
        speaker = str(row.get("speaker_id", "")).strip()
        if speaker:
            by_speaker[speaker].append(row)
    speakers = sorted(by_speaker)
    if not speakers:
        raise RuntimeError(f"No speaker_id rows in {intent_manifest}")

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        args.device = "cpu"
    print(f"Loading kNN-VC from {knn_root} on {args.device}", flush=True)
    knn_vc = torch.hub.load(
        str(knn_root),
        "knn_vc",
        source="local",
        pretrained=True,
        prematched=True,
        trust_repo=True,
        device=args.device,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []
    for speaker_index, speaker in enumerate(speakers):
        speaker_rows = sorted(by_speaker[speaker], key=lambda row: str(row.get("wav", "")))
        references = speaker_rows[: args.references_per_speaker]
        reference_paths = [str(resolve_repo_path(repo_root, row["wav"])) for row in references]
        matching_set = knn_vc.get_matching_set(reference_paths)
        speaker_dir = output_dir / speaker
        speaker_dir.mkdir(parents=True, exist_ok=True)
        for wake_index in range(args.wakes_per_speaker):
            source_row = wake_rows[(speaker_index * args.wakes_per_speaker + wake_index) % len(wake_rows)]
            source_path = resolve_repo_path(repo_root, source_row["wav"])
            query = knn_vc.get_features(str(source_path))
            info = sf.info(str(source_path))
            duration = float(info.frames) / max(1, int(info.samplerate))
            converted = knn_vc.match(
                query,
                matching_set,
                topk=max(1, int(args.topk)),
                target_duration=duration,
            ).detach().cpu()
            output_path = speaker_dir / f"wake_{wake_index:04d}_{source_path.stem}.wav"
            torchaudio.save(str(output_path), converted.unsqueeze(0), 16000)
            manifest_rows.append(
                {
                    "wav": str(output_path),
                    "label": 1,
                    "split": "test",
                    "wake_word": "marvin",
                    "noise_only": False,
                    "speaker_id": speaker,
                    "voice_conversion": "official_knn_vc_prematched_hifigan",
                    "voice_conversion_source": str(source_path),
                    "voice_conversion_references": reference_paths,
                    "voice_conversion_topk": int(args.topk),
                    "synthetic": True,
                    "real_driver": False,
                }
            )
            print(
                f"speaker={speaker_index + 1}/{len(speakers)} "
                f"wake={wake_index + 1}/{args.wakes_per_speaker} "
                f"source={source_path.name}",
                flush=True,
            )

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    metadata = {
        "source_wake_manifest": str(wake_manifest),
        "source_intent_manifest": str(intent_manifest),
        "output_manifest": str(output_manifest),
        "output_dir": str(output_dir),
        "speakers": speakers,
        "speaker_count": len(speakers),
        "wakes_per_speaker": int(args.wakes_per_speaker),
        "generated_rows": len(manifest_rows),
        "references_per_speaker": int(args.references_per_speaker),
        "topk": int(args.topk),
        "device": str(args.device),
        "synthetic": True,
        "real_driver": False,
        "license_note": "Generated research benchmark artifacts; verify source and model licenses before distribution.",
    }
    metadata_path = output_manifest.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
