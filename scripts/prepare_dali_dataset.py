from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import pickle
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac"}
PREPARED_AUDIO_FORMATS = {"flac", "wav"}
SEGMENT_TIME_DECIMALS = 6 #6dp


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


def read_ids_file(ids_file: Path) -> set[str]:
    if not ids_file.is_file():
        raise SystemExit(f"IDs file does not exist: {ids_file}")
    ids = {line.strip() for line in ids_file.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not ids:
        raise SystemExit(f"IDs file is empty: {ids_file}")
    return ids


def find_audio_file(entry: Any, by_stem: dict[str, Path]) -> Path | None:
    dali_id = entry.info["id"]
    candidates = [dali_id]

    audio_path = entry.info.get("audio", {}).get("path")
    if audio_path and str(audio_path).lower() != "none":
        path = Path(audio_path)
        if path.exists():
            return path.resolve(strict=False)
        candidates.append(path.stem)

    for stem in candidates:
        if stem in by_stem:
            return by_stem[stem].resolve(strict=False)
    return None


def audio_duration_seconds(audio_path: Path, fallback_duration: float | None = None) -> float:
    try:
        import soundfile as sf

        info = sf.info(str(audio_path))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        try:
            if fallback_duration is not None and fallback_duration > 0:
                return float(fallback_duration)
            os.environ.setdefault("NUMBA_CACHE_DIR", str(Path.cwd() / ".numba_cache"))
            import librosa

            return float(librosa.get_duration(path=str(audio_path)))
        except Exception:
            raise


def prepared_audio_is_valid(audio_path: Path, sample_rate: int, audio_format: str) -> bool:
    if not audio_path.exists():
        return False
    if audio_path.suffix.lower() != f".{audio_format}":
        return False
    try:
        import soundfile as sf

        info = sf.info(str(audio_path))
    except Exception:
        return False
    return info.samplerate == sample_rate and info.channels == 1


def prepare_audio_file(
    input_path: Path,
    output_dir: Path,
    dali_id: str,
    sample_rate: int,
    audio_format: str,
    overwrite: bool = False,
) -> Path:
    if audio_format not in PREPARED_AUDIO_FORMATS:
        raise ValueError(f"Unsupported audio format: {audio_format}")

    output_path = output_dir / f"{dali_id}.{audio_format}"
    if (
        not overwrite
        and prepared_audio_is_valid(
            output_path,
            sample_rate=sample_rate,
            audio_format=audio_format,
        )
    ):
        return output_path.resolve(strict=False)

    os.environ.setdefault("NUMBA_CACHE_DIR", str((output_dir / ".numba_cache").resolve(strict=False)))
    import librosa
    import soundfile as sf

    waveform, _ = librosa.load(str(input_path), sr=sample_rate, mono=True)
    subtype = "PCM_16" if audio_format == "wav" else None
    sf.write(str(output_path), waveform, sample_rate, subtype=subtype)
    return output_path.resolve(strict=False)


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


def stable_sample_id(dali_id: str, start_seconds: float) -> str:
    digest = hashlib.sha1(f"{dali_id}:{start_seconds:.3f}".encode("utf-8")).hexdigest()[:10]
    return f"{dali_id}_{int(round(start_seconds * 1000)):09d}_{digest}"


def extract_dali_notes(entry: Any) -> list[dict[str, Any]]:
    annotations = getattr(entry, "annotations", {})
    annot = annotations.get("annot", {}) if isinstance(annotations, dict) else {}
    notes = annot.get("notes", []) if isinstance(annot, dict) else []

    valid_notes: list[dict[str, Any]] = []
    for note in notes:
        times = np.asarray(note.get("time", []), dtype=np.float32)
        freqs = np.asarray(note.get("freq", []), dtype=np.float32)
        if times.size < 2 or freqs.size < 1:
            continue
        if times.size != freqs.size:
            if freqs.size == 1:
                freqs = np.repeat(freqs, times.size)
            else:
                continue
        order = np.argsort(times)
        times = times[order]
        freqs = freqs[order]
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(freqs)):
            continue
        if float(times[-1]) <= float(times[0]):
            continue
        if float(np.nanmax(freqs)) <= 0.0:
            continue
        valid_notes.append(
            {
                "time": times,
                "freq": freqs,
                "text": str(note.get("text", "")),
                "index": note.get("index"),
            }
        )
    valid_notes.sort(key=lambda item: float(item["time"][0]))
    return valid_notes


def render_melody_frames(
    notes: list[dict[str, Any]],
    start_seconds: float,
    segment_seconds: float,
    frame_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_frames = int(round(segment_seconds * frame_rate))
    frame_times = start_seconds + (np.arange(num_frames, dtype=np.float32) + 0.5) / frame_rate
    f0_hz = np.zeros(num_frames, dtype=np.float32)
    voiced = np.zeros(num_frames, dtype=np.bool_)

    segment_end = start_seconds + segment_seconds
    for note in notes:
        times = note["time"]
        freqs = note["freq"]
        note_start = float(times[0])
        note_end = float(times[-1])
        if note_end <= start_seconds or note_start >= segment_end:
            continue
        mask = (frame_times >= note_start) & (frame_times < note_end)
        if not np.any(mask):
            continue
        values = np.interp(frame_times[mask], times, freqs).astype(np.float32)
        valid = np.isfinite(values) & (values > 0.0)
        if not np.any(valid):
            continue
        masked_indices = np.flatnonzero(mask)
        valid_indices = masked_indices[valid]
        f0_hz[valid_indices] = values[valid]
        voiced[valid_indices] = True

    relative_times = frame_times - start_seconds
    return relative_times.astype(np.float32), f0_hz, voiced


def iter_segment_starts(duration: float, segment_seconds: float, hop_seconds: float) -> list[float]:
    last_start = duration - segment_seconds
    if last_start < 0:
        return []
    count = int(math.floor(last_start / hop_seconds)) + 1
    return [round(index * hop_seconds, SEGMENT_TIME_DECIMALS) for index in range(count)]


def annotation_duration_seconds(notes: list[dict[str, Any]]) -> float:
    if not notes:
        return 0.0
    return max(float(note["time"][-1]) for note in notes)


def write_melody_npz(
    melody_dir: Path,
    sample_id: str,
    frame_times: np.ndarray,
    f0_hz: np.ndarray,
    voiced: np.ndarray,
    frame_rate: float,
    start_seconds: float,
    segment_seconds: float,
) -> Path:
    melody_path = melody_dir / f"{sample_id}.npz"
    np.savez_compressed(
        melody_path,
        frame_times=frame_times,
        f0_hz=f0_hz,
        voiced=voiced.astype(np.uint8),
        frame_rate=np.asarray(frame_rate, dtype=np.float32),
        start_seconds=np.asarray(start_seconds, dtype=np.float32),
        duration_seconds=np.asarray(segment_seconds, dtype=np.float32),
    )
    return melody_path.resolve(strict=False)


def row_for_segment(
    segment: dict[str, Any],
    split: str,
    track_info: dict[str, Any],
    manifest_dir: Path,
) -> dict[str, Any]:
    return {
        "sample_id": segment["sample_id"],
        "split": split,
        "dali_id": track_info["dali_id"],
        "artist": track_info["artist"],
        "title": track_info["title"],
        "audio_path": manifest_relative_path(track_info["audio_path"], manifest_dir),
        "raw_audio_path": manifest_relative_path(track_info["raw_audio_path"], manifest_dir),
        "melody_path": manifest_relative_path(segment["melody_path"], manifest_dir),
        "start_seconds": f"{segment['start_seconds']:.6f}",
        "end_seconds": f"{segment['end_seconds']:.6f}",
        "segment_seconds": f"{segment['segment_seconds']:.6f}",
        "melody_frame_rate": f"{segment['frame_rate']:.6f}",
        "voiced_ratio": f"{segment['voiced_ratio']:.6f}",
    }


def manifest_relative_path(path: str | Path, manifest_dir: Path) -> str:
    path = Path(path).expanduser().resolve(strict=False)
    manifest_dir = Path(manifest_dir).expanduser().resolve(strict=False)
    return Path(os.path.relpath(path, manifest_dir)).as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare DALI vocal-melody/audio contrastive data. The script writes one "
            "canonical frame-based melody .npz per eligible segment and a segment manifest."
        )
    )
    parser.add_argument("--dali-data-dir", type=Path, default=Path("data/DALI_v1"))
    parser.add_argument("--audio-dir", type=Path, default=Path("data/audio"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/prepared/dali"))
    parser.add_argument("--gt-file", type=Path, default=None, help="Optional DALI ground-truth gzip file.")
    parser.add_argument(
        "--ground-truth-only",
        action="store_true",
        help="Use only DALI ids present in --gt-file. Useful for small aligned experiments.",
    )
    parser.add_argument(
        "--ids-file",
        type=Path,
        default=None,
        help="Optional text file of DALI ids to load, one id per line.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target audio rate for future dataloaders.")
    parser.add_argument(
        "--prepared-audio-dir",
        type=Path,
        default=None,
        help="Where to write per-song mono resampled audio. Defaults to <output-dir>/audio_16k.",
    )
    parser.add_argument("--audio-format", choices=sorted(PREPARED_AUDIO_FORMATS), default="flac")
    parser.add_argument(
        "--skip-audio-prep",
        action="store_true",
        help="Keep manifest audio paths pointing at original files instead of writing 16 kHz per-song audio.",
    )
    parser.add_argument(
        "--overwrite-audio",
        action="store_true",
        help="Regenerate prepared audio files even if they already exist.",
    )
    parser.add_argument("--segment-seconds", type=float, default=10.0)
    parser.add_argument("--hop-seconds", type=float, default=10.0)
    parser.add_argument("--melody-frame-rate", type=float, default=50.0, help="Number of frames per second.")
    parser.add_argument("--max-tracks", type=int, default=0, help="0 means no track limit.")
    parser.add_argument("--max-segments", type=int, default=0, help="0 means no segment limit.")
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
    if args.ids_file is not None:
        args.ids_file = resolve_user_path(args.ids_file)
    if args.prepared_audio_dir is None:
        args.prepared_audio_dir = args.output_dir / f"audio_{args.sample_rate // 1000}k"
    args.prepared_audio_dir = resolve_user_path(args.prepared_audio_dir)
    if args.segment_seconds <= 0:
        raise SystemExit("--segment-seconds must be positive")
    if args.hop_seconds <= 0:
        raise SystemExit("--hop-seconds must be positive")
    if args.melody_frame_rate <= 0:
        raise SystemExit("--melody-frame-rate must be positive")
    if not 0.0 <= args.min_vocal_ratio <= 1.0:
        raise SystemExit("--min-vocal-ratio must be between 0 and 1")
    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise SystemExit("--train-ratio and --val-ratio must leave a positive test split")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    melody_dir = args.output_dir / "melodies"
    melody_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_audio_prep:
        args.prepared_audio_dir.mkdir(parents=True, exist_ok=True)

    prepare_input_dirs(args.dali_data_dir, args.audio_dir)
    by_stem, audio_files = build_audio_index(args.audio_dir)
    validate_input_files(args.dali_data_dir, args.audio_dir, audio_files, args.gt_file)

    keep_ids = None
    if args.ground_truth_only:
        if args.gt_file is None:
            raise SystemExit("--ground-truth-only requires --gt-file")
        keep_ids = read_ground_truth_ids(args.gt_file)
    if args.ids_file is not None:
        file_ids = read_ids_file(args.ids_file)
        keep_ids = file_ids if keep_ids is None else keep_ids & file_ids
        if not keep_ids:
            raise SystemExit("No ids remain after applying --ids-file and other filters.")

    print("Loading DALI annotations...")
    dali_data = load_dali(args.dali_data_dir, args.gt_file, keep_ids=keep_ids)
    print(f"Loaded {len(dali_data)} DALI entries and indexed {len(audio_files)} audio files.")

    entries = list(dali_data.values())
    entries.sort(key=lambda entry: entry.info["id"])
    random.Random(args.seed).shuffle(entries)

    usable_tracks: list[dict[str, Any]] = []
    skipped = Counter()
    duration_sources = Counter()
    prepared_audio_count = 0
    for entry in entries:
        dali_id = entry.info["id"]
        audio_path = find_audio_file(entry, by_stem)
        if audio_path is None:
            skipped["no_audio"] += 1
            continue

        notes = extract_dali_notes(entry)
        if not notes:
            skipped["no_notes"] += 1
            continue

        annotation_duration = annotation_duration_seconds(notes)
        try:
            duration = audio_duration_seconds(audio_path, fallback_duration=annotation_duration)
        except Exception as exc:
            print(f"Skipping {dali_id}: could not read duration for {audio_path}: {exc}")
            skipped["bad_audio"] += 1
            continue
        if abs(duration - annotation_duration) < 1e-6:
            duration_sources["annotations"] += 1
        else:
            duration_sources["audio_metadata"] += 1

        if duration < args.segment_seconds:
            skipped["too_short"] += 1
            continue

        prepared_audio_path = audio_path
        if not args.skip_audio_prep:
            try:
                prepared_audio_path = prepare_audio_file(
                    input_path=audio_path,
                    output_dir=args.prepared_audio_dir,
                    dali_id=dali_id,
                    sample_rate=args.sample_rate,
                    audio_format=args.audio_format,
                    overwrite=args.overwrite_audio,
                )
                prepared_audio_count += 1
                duration = audio_duration_seconds(prepared_audio_path, fallback_duration=duration)
            except Exception as exc:
                print(f"Skipping {dali_id}: could not prepare 16 kHz audio for {audio_path}: {exc}")
                skipped["audio_prep_failed"] += 1
                continue

        usable_tracks.append(
            {
                "dali_id": dali_id,
                "entry": entry,
                "raw_audio_path": audio_path,
                "audio_path": prepared_audio_path,
                "notes": notes,
                "duration": duration,
                "artist": entry.info.get("artist", ""),
                "title": entry.info.get("title", ""),
            }
        )
        if args.max_tracks > 0 and len(usable_tracks) >= args.max_tracks:
            break

    if not usable_tracks:
        raise SystemExit("No usable DALI entries found. Check audio filenames and --audio-dir.")

    split_by_id = split_track_ids(
        [track["dali_id"] for track in usable_tracks],
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    all_segments: list[dict[str, Any]] = []
    skipped_segments = Counter()
    for track in usable_tracks:
        starts = iter_segment_starts(track["duration"], args.segment_seconds, args.hop_seconds)
        for start_seconds in starts:
            frame_times, f0_hz, voiced = render_melody_frames(
                track["notes"],
                start_seconds=start_seconds,
                segment_seconds=args.segment_seconds,
                frame_rate=args.melody_frame_rate,
            )
            voiced_ratio = float(np.mean(voiced)) if voiced.size else 0.0
            if voiced_ratio < args.min_vocal_ratio:
                skipped_segments["low_vocal_ratio"] += 1
                continue

            sample_id = stable_sample_id(track["dali_id"], start_seconds)
            melody_path = write_melody_npz(
                melody_dir=melody_dir,
                sample_id=sample_id,
                frame_times=frame_times,
                f0_hz=f0_hz,
                voiced=voiced,
                frame_rate=args.melody_frame_rate,
                start_seconds=start_seconds,
                segment_seconds=args.segment_seconds,
            )
            segment = {
                "sample_id": sample_id,
                "dali_id": track["dali_id"],
                "melody_path": melody_path,
                "start_seconds": start_seconds,
                "end_seconds": start_seconds + args.segment_seconds,
                "segment_seconds": args.segment_seconds,
                "frame_rate": args.melody_frame_rate,
                "voiced_ratio": voiced_ratio,
            }
            all_segments.append(segment)
            if args.max_segments > 0 and len(all_segments) >= args.max_segments:
                break

        if args.max_segments > 0 and len(all_segments) >= args.max_segments:
            break

    if not all_segments:
        raise SystemExit(
            "No eligible segments found. Try lowering --min-vocal-ratio or checking DALI/audio alignment."
        )

    track_by_id = {track["dali_id"]: track for track in usable_tracks}
    segment_manifest_path = args.output_dir / "segments_manifest.csv"
    segment_rows: list[dict[str, Any]] = []
    for segment in all_segments:
        track = track_by_id[segment["dali_id"]]
        split = split_by_id[track["dali_id"]]
        segment_rows.append(
            row_for_segment(
                segment=segment,
                split=split,
                track_info=track,
                manifest_dir=segment_manifest_path.parent,
            )
        )

    with segment_manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(segment_rows[0].keys()))
        writer.writeheader()
        writer.writerows(segment_rows)

    metadata_path = args.output_dir / "metadata.json"
    metadata = {
        "source": "DALI",
        "dali_data_dir": manifest_relative_path(args.dali_data_dir, metadata_path.parent),
        "audio_dir": manifest_relative_path(args.audio_dir, metadata_path.parent),
        "ids_file": (
            manifest_relative_path(args.ids_file, metadata_path.parent)
            if args.ids_file is not None
            else None
        ),
        "prepared_audio_dir": manifest_relative_path(args.prepared_audio_dir, metadata_path.parent),
        "audio_format": args.audio_format,
        "skip_audio_prep": args.skip_audio_prep,
        "sample_rate": args.sample_rate,
        "segment_seconds": args.segment_seconds,
        "hop_seconds": args.hop_seconds,
        "melody_frame_rate": args.melody_frame_rate,
        "min_vocal_ratio": args.min_vocal_ratio,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "num_tracks": len(usable_tracks),
        "num_segments": len(all_segments),
        "num_segment_rows": len(segment_rows),
        "num_prepared_audio_files": prepared_audio_count,
        "skipped_tracks": dict(skipped),
        "duration_sources": dict(duration_sources),
        "skipped_segments": dict(skipped_segments),
        "split_counts": dict(Counter(row["split"] for row in segment_rows)),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    print(f"Wrote {len(all_segments)} melody segments to {melody_dir}")
    if not args.skip_audio_prep:
        print(f"Wrote/reused {prepared_audio_count} prepared audio files in {args.prepared_audio_dir}")
    print(f"Wrote {len(segment_rows)} segment rows to {segment_manifest_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Track skips: {dict(skipped)}")
    print(f"Duration sources: {dict(duration_sources)}")
    print(f"Segment skips: {dict(skipped_segments)}")
    print(f"Splits: {metadata['split_counts']}")


if __name__ == "__main__":
    main()
