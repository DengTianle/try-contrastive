from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np


def load_dali(
    dali_data_dir: Path,
    gt_file: Path | None,
    keep_ids: set[str] | None = None,
) -> dict[str, Any]:
    try:
        import DALI as dali_code
    except ImportError as exc:
        raise SystemExit(
            "Could not import DALI. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    keep = sorted(keep_ids) if keep_ids else []
    if gt_file is None:
        return dali_code.get_the_DALI_dataset(str(dali_data_dir), skip=[], keep=keep)
    return dali_code.get_the_DALI_dataset(str(dali_data_dir), gt_file=str(gt_file), skip=[], keep=keep)


def build_audio_index(audio_dir: Path) -> tuple[dict[str, Path], list[Path]]:
    files = [p for p in audio_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS]
    by_stem: dict[str, Path] = {}
    for path in files:
        by_stem.setdefault(path.stem, path)
    return by_stem, files


def resolve_user_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return expanded.resolve(strict=False)


def prepare_input_dirs(dali_data_dir: Path, audio_dir: Path) -> None:
    created_dirs = []
    for label, path in (("DALI annotation", dali_data_dir), ("audio", audio_dir)):
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created_dirs.append((label, path))
        elif not path.is_dir():
            raise SystemExit(f"{label} path is not a directory: {path}")

    if created_dirs:
        print("Created missing input directories:")
        for label, path in created_dirs:
            print(f"  {label}: {path}")


def validate_input_files(
    dali_data_dir: Path,
    audio_dir: Path,
    audio_files: list[Path],
    gt_file: Path | None,
) -> None:
    dali_files = list(dali_data_dir.rglob("*.gz"))
    problems = []
    if not dali_files:
        problems.append(f"No DALI .gz annotation files found in: {dali_data_dir}")
    if not audio_files:
        problems.append(
            f"No audio files found in: {audio_dir}. "
            f"Supported extensions: {', '.join(sorted(AUDIO_EXTENSIONS))}"
        )
    if gt_file is not None and not gt_file.is_file():
        problems.append(f"Ground-truth file does not exist: {gt_file}")

    if problems:
        raise SystemExit(
            "Input data is not ready yet.\n"
            + "\n".join(f"- {problem}" for problem in problems)
            + "\n\nPut DALI annotation .gz files in --dali-data-dir and local audio files in --audio-dir, then rerun."
        )


def read_ground_truth_ids(gt_file: Path) -> set[str]:
    with gzip.open(gt_file, "rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise SystemExit(f"Expected ground-truth dict, got {type(data).__name__}: {gt_file}")
    return set(data.keys())


def find_audio_file(entry: Any, by_stem: dict[str, Path]) -> Path | None:
    dali_id = entry.info["id"]
    candidates = [dali_id]

    audio_path = entry.info.get("audio", {}).get("path")
    if audio_path and str(audio_path).lower() != "none":
        path = Path(audio_path)
        if path.exists():
            return path
        candidates.append(path.stem)

    for stem in candidates:
        if stem in by_stem:
            return by_stem[stem]
    return None


def audio_duration_seconds(audio_path: Path) -> float:
    import librosa
    import soundfile as sf

    try:
        info = sf.info(str(audio_path))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return float(librosa.get_duration(path=str(audio_path)))

def split_track_ids(
    track_ids: list[str],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, str]:
    ids = list(track_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    if n == 1:
        return {ids[0]: "train"}

    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio))) if n >= 3 else 0
    if n_train + n_val >= n:
        n_train = max(1, n - n_val - 1)

    split_by_id = {}
    for index, dali_id in enumerate(ids):
        if index < n_train:
            split_by_id[dali_id] = "train"
        elif index < n_train + n_val:
            split_by_id[dali_id] = "val"
        else:
            split_by_id[dali_id] = "test"
    return split_by_id


def segment_quotas(
    max_segments: int,
    present_splits: set[str],
    train_ratio: float,
    val_ratio: float,
) -> dict[str, int]:
    ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "test": max(0.0, 1.0 - train_ratio - val_ratio),
    }
    total_ratio = sum(ratios[split] for split in present_splits)
    quotas = {
        split: max(1, int(round(max_segments * ratios[split] / total_ratio)))
        for split in present_splits
    }
    while sum(quotas.values()) > max_segments:
        largest = max(quotas, key=quotas.get)
        quotas[largest] -= 1
    while sum(quotas.values()) < max_segments:
        largest = max(present_splits, key=lambda split: ratios[split])
        quotas[largest] += 1
    return quotas


def stable_sample_id(dali_id: str, start_seconds: float) -> str:
    digest = hashlib.sha1(f"{dali_id}:{start_seconds:.3f}".encode("utf-8")).hexdigest()[:10]
    return f"{dali_id}_{int(round(start_seconds * 1000)):09d}_{digest}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dali-data-dir", type=Path, default=Path("data/DALI_v1"))
    parser.add_argument("--audio-dir", type=Path, default=Path("data/audio"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--gt-file", type=Path, default=None, help="Optional DALI ground-truth gzip file.")
    parser.add_argument(
        "--ground-truth-only",
        action="store_true",
        help="Use only DALI ids present in --gt-file. Useful for small aligned experiments.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--segment-seconds", type=float, default=10.0)
    parser.add_argument("--hop-seconds", type=float, default=10.0)
    parser.add_argument("--max-tracks", type=int, default=20)
    parser.add_argument("--max-segments", type=int, default=200)
    parser.add_argument("--min-vocal-ratio", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dali_data_dir = resolve_user_path(args.dali_data_dir)
    args.audio_dir = resolve_user_path(args.audio_dir)
    args.output_dir = resolve_user_path(args.output_dir)
    if args.gt_file is not None:
        args.gt_file = resolve_user_path(args.gt_file)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = args.output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    prepare_input_dirs(args.dali_data_dir, args.audio_dir)
    by_stem, audio_files = build_audio_index(args.audio_dir)
    validate_input_files(args.dali_data_dir, args.audio_dir, audio_files, args.gt_file)

    import librosa
    from tqdm import tqdm

    keep_ids = None
    if args.ground_truth_only:
        if args.gt_file is None:
            raise SystemExit("--ground-truth-only requires --gt-file")
        keep_ids = read_ground_truth_ids(args.gt_file)

    print("Loading DALI annotations...")
    dali_data = load_dali(args.dali_data_dir, args.gt_file, keep_ids=keep_ids)
    print(f"Indexed {len(audio_files)} audio files.")
    if keep_ids is not None:
        print(f"Restricted to {len(dali_data)} ground-truth DALI entries.")

    entries = list(dali_data.values())
    entries.sort(key=lambda entry: entry.info["id"])
    rng = random.Random(args.seed)
    rng.shuffle(entries)

    usable_entries: list[tuple[str, Any, Path, list[dict[str, float]], float]] = []
    skipped_no_audio = 0
    skipped_no_notes = 0
    skipped_bad_audio = 0
    for entry in entries:
        audio_path = find_audio_file(entry, by_stem)
        if audio_path is None:
            skipped_no_audio += 1
            continue
        notes = notes_from_entry(entry)
        if not notes:
            skipped_no_notes += 1
            continue
        try:
            duration = audio_duration_seconds(audio_path)
        except Exception as exc:
            print(f"Skipping {entry.info['id']}: could not read duration for {audio_path}: {exc}")
            skipped_bad_audio += 1
            continue
        usable_entries.append((entry.info["id"], entry, audio_path, notes, duration))
        if len(usable_entries) >= args.max_tracks:
            break

    if not usable_entries:
        raise SystemExit("No usable DALI entries found. Check audio filenames and --audio-dir.")
    print(
        f"Found {len(usable_entries)} usable tracks "
        f"(max_tracks={args.max_tracks}, indexed_audio_files={len(audio_files)}, "
        f"skipped_no_audio={skipped_no_audio}, skipped_no_notes={skipped_no_notes}, "
        f"skipped_bad_audio={skipped_bad_audio})"
    )

    split_by_id = split_track_ids(
        [dali_id for dali_id, _, _, _, _ in usable_entries],
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    #deleted

    manifest_path = args.output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} samples to {args.output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Split quotas: {quotas}")
    print(f"Split counts: {split_counts}")
    unmet = {split: quotas[split] - split_counts.get(split, 0) for split in quotas if split_counts.get(split, 0) < quotas[split]}
    if unmet:
        print(
            "Some split quotas were not filled because there were not enough eligible segments "
            f"after audio matching and min_vocal_ratio filtering: {unmet}"
        )
    print(f"Label frames per {args.segment_seconds:g}s segment: {hubert_num_frames(segment_samples)}")


if __name__ == "__main__":
    main()
