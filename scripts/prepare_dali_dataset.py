from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import pickle
import random
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prosodia.melody_encoder import (
    MELODY_FEATURE_DIM,
    MELODY_REPRESENTATION,
    hz_to_midi_pitch,
)


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac"}
PREPARED_AUDIO_FORMATS = {"flac", "wav"}


def repeat_segments_path(path: Path) -> Path:
    resolved = resolve_user_path(path)
    if resolved.is_dir():
        resolved = resolved / "segments.jsonl.gz"
    if not resolved.is_file():
        raise SystemExit(f"Repeat-grouping results do not exist: {resolved}")
    return resolved


def load_line_repeat_groupings(
    path: Path,
) -> tuple[dict[tuple[str, int], dict[str, Any]], set[str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    groupings: dict[tuple[str, int], dict[str, Any]] = {}
    grouped_song_ids: set[str] = set()
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if row.get("level") != "line":
                continue
            try:
                key = (str(row["song_id"]), int(row["segment_index"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise SystemExit(
                    f"Invalid repeat-grouping row at {path}:{line_number}"
                ) from exc
            if key in groupings:
                raise SystemExit(f"Duplicate repeat-grouping assignment for {key}")
            groupings[key] = row
            grouped_song_ids.add(key[0])
    if not groupings:
        raise SystemExit(f"No line-level repeat groupings found in: {path}")
    return groupings, grouped_song_ids


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


def read_keep_file(path: Path) -> list[str]:
    return [
        line.split("#", 1)[0].strip()
        for line in resolve_user_path(path).read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    ]


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


def stable_sample_id(
    dali_id: str,
    start_seconds: float,
    line_index: int | None = None,
) -> str:
    identity = f"{dali_id}:{start_seconds:.3f}"
    if line_index is not None:
        identity = f"{dali_id}:{start_seconds:.6f}:{line_index}"
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:10]
    return f"{dali_id}_{int(round(start_seconds * 1000)):09d}_{digest}"


def flatten_dali_text(value: Any) -> str:
    if isinstance(value, dict):
        return flatten_dali_text(value.get("text", ""))
    if isinstance(value, (list, tuple)):
        return " ".join(part for item in value if (part := flatten_dali_text(item)))
    return str(value).strip()


def normalize_line_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def extract_dali_lines(entry: Any) -> list[dict[str, Any]]:
    annotations = getattr(entry, "annotations", {})
    annot = annotations.get("annot", {}) if isinstance(annotations, dict) else {}
    lines = annot.get("lines", []) if isinstance(annot, dict) else []

    if not lines and isinstance(annot, dict):
        hierarchical = annot.get("hierarchical", [])
        lines = [
            line
            for paragraph in hierarchical
            if isinstance(paragraph, dict)
            for line in paragraph.get("text", [])
            if isinstance(line, dict)
        ]

    valid_lines: list[dict[str, Any]] = []
    for line_index, line in enumerate(lines):
        times = np.asarray(line.get("time", []), dtype=np.float64)
        if times.size < 2 or not np.all(np.isfinite(times)):
            continue
        start_seconds = float(times[0])
        end_seconds = float(times[-1])
        if end_seconds <= start_seconds:
            continue
        text = flatten_dali_text(line.get("text", ""))
        normalized_text = normalize_line_text(text)
        if not normalized_text:
            continue
        valid_lines.append(
            {
                "line_index": line_index,
                "text": text,
                "normalized_text": normalized_text,
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
            }
        )
    valid_lines.sort(key=lambda item: (item["start_seconds"], item["line_index"]))
    return valid_lines


def unique_line_segments(
    lines: list[dict[str, Any]],
    audio_duration: float,
    trim_repeated_text: bool = True,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Clip line intervals and optionally keep only the first normalized lyric."""
    seen_text: set[str] = set()
    segments: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for line in lines:
        start_seconds = max(0.0, float(line["start_seconds"]))
        end_seconds = min(float(audio_duration), float(line["end_seconds"]))
        if end_seconds <= start_seconds:
            skipped["outside_audio"] += 1
            continue
        normalized_text = str(line["normalized_text"])
        if trim_repeated_text and normalized_text in seen_text:
            skipped["repeated_line"] += 1
            continue
        seen_text.add(normalized_text)
        segments.append(
            {
                **line,
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "segment_seconds": end_seconds - start_seconds,
            }
        )
    return segments, skipped


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
        if float(freqs[0]) <= 0.0:
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


def extract_segment_note_arrays(
    notes: list[dict[str, Any]],
    start_seconds: float,
    segment_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    segment_end = start_seconds + segment_seconds
    overlapping = [
        note
        for note in notes
        if float(note["time"][-1]) > start_seconds
        and float(note["time"][0]) < segment_end
    ]
    midi_pitches = np.asarray(
        [hz_to_midi_pitch(float(note["freq"][0])) for note in overlapping],
        dtype=np.int16,
    )
    onset_seconds = np.asarray(
        [float(note["time"][0]) - start_seconds for note in overlapping],
        dtype=np.float32,
    )
    duration_seconds = np.asarray(
        [float(note["time"][-1]) - float(note["time"][0]) for note in overlapping],
        dtype=np.float32,
    )
    return midi_pitches, onset_seconds, duration_seconds


def note_coverage_ratio(
    notes: list[dict[str, Any]],
    start_seconds: float,
    segment_seconds: float,
) -> float:
    """Return the fraction of a segment covered by the union of note intervals."""
    if segment_seconds <= 0.0:
        return 0.0
    segment_end = start_seconds + segment_seconds
    intervals = sorted(
        (
            max(start_seconds, float(note["time"][0])),
            min(segment_end, float(note["time"][-1])),
        )
        for note in notes
        if float(note["time"][-1]) > start_seconds
        and float(note["time"][0]) < segment_end
    )
    covered = 0.0
    current_start: float | None = None
    current_end: float | None = None
    for interval_start, interval_end in intervals:
        if current_start is None:
            current_start, current_end = interval_start, interval_end
        elif current_end is not None and interval_start <= current_end:
            current_end = max(current_end, interval_end)
        else:
            assert current_end is not None
            covered += current_end - current_start
            current_start, current_end = interval_start, interval_end
    if current_start is not None and current_end is not None:
        covered += current_end - current_start
    return float(np.clip(covered / segment_seconds, 0.0, 1.0))


def annotation_duration_seconds(notes: list[dict[str, Any]]) -> float:
    if not notes:
        return 0.0
    return max(float(note["time"][-1]) for note in notes)


def count_segment_notes(
    notes: list[dict[str, Any]],
    start_seconds: float,
    segment_seconds: float,
) -> int:
    """Count annotated notes that overlap a segment by any positive duration."""
    segment_end = start_seconds + segment_seconds
    return sum(
        float(note["time"][-1]) > start_seconds
        and float(note["time"][0]) < segment_end
        for note in notes
    )


def segment_quality_skip_reason(
    segment_seconds: float,
    note_count: int,
    max_segment_seconds: float,
    min_segment_notes: int,
) -> str | None:
    if max_segment_seconds > 0 and segment_seconds > max_segment_seconds:
        return "too_long"
    if note_count == 0 or note_count < min_segment_notes:
        return "too_few_notes"
    return None


def write_melody_npz(
    melody_dir: Path,
    sample_id: str,
    midi_pitches: np.ndarray,
    onset_seconds: np.ndarray,
    note_duration_seconds: np.ndarray,
    start_seconds: float,
    segment_seconds: float,
) -> Path:
    melody_path = melody_dir / f"{sample_id}.npz"
    np.savez_compressed(
        melody_path,
        midi_pitches=midi_pitches,
        onset_seconds=onset_seconds,
        note_duration_seconds=note_duration_seconds,
        start_seconds=np.asarray(start_seconds, dtype=np.float32),
        segment_duration_seconds=np.asarray(segment_seconds, dtype=np.float32),
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
        "audio_duration_seconds": f"{track_info['duration']:.6f}",
        "start_seconds": f"{segment['start_seconds']:.6f}",
        "end_seconds": f"{segment['end_seconds']:.6f}",
        "segment_seconds": f"{segment['segment_seconds']:.6f}",
        "voiced_ratio": f"{segment['voiced_ratio']:.6f}",
        "note_count": str(segment["note_count"]),
        "line_index": str(segment["line_index"]),
        "line_text": segment["line_text"],
        "lyric_class": optional_manifest_value(segment.get("lyric_class")),
        "melody_class": optional_manifest_value(segment.get("melody_class")),
        "parent_index": optional_manifest_value(segment.get("parent_index")),
        "transposition_to_representative": optional_manifest_value(
            segment.get("transposition_to_representative")
        ),
        "onset_mae_to_representative": optional_manifest_value(
            segment.get("onset_mae_to_representative")
        ),
        "duration_mae_to_representative": optional_manifest_value(
            segment.get("duration_mae_to_representative")
        ),
        "group_quality_flags": json.dumps(
            segment.get("group_quality_flags", []),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }


def manifest_relative_path(path: str | Path, manifest_dir: Path) -> str:
    path = Path(path).expanduser().resolve(strict=False)
    manifest_dir = Path(manifest_dir).expanduser().resolve(strict=False)
    return Path(os.path.relpath(path, manifest_dir)).as_posix()


def optional_manifest_value(value: Any) -> Any:
    return "" if value is None else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare DALI vocal-melody/audio contrastive data. The script writes one "
            "canonical note-level melody .npz per retained DALI lyric-line occurrence "
            "and a segment manifest."
        )
    )
    parser.add_argument("--dali-data-dir", type=Path, default=Path("data/DALI_v1"))
    parser.add_argument("--audio-dir", type=Path, default=Path("data/audio"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/prepared/dali"))
    parser.add_argument("--gt-file", type=Path, default=None, help="Optional DALI ground-truth gzip file.")
    parser.add_argument(
        "--keep-file",
        type=Path,
        default=None,
        help="Prepare only DALI ids listed in this text file (one id per line).",
    )
    parser.add_argument(
        "--ground-truth-only",
        action="store_true",
        help="Use only DALI ids present in --gt-file. Useful for small aligned experiments.",
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
    parser.add_argument("--max-tracks", type=int, default=0, help="0 means no track limit.")
    parser.add_argument("--max-segments", type=int, default=0, help="0 means no segment limit.")
    parser.add_argument(
        "--max-segment-seconds",
        type=float,
        default=10.0,
        help="Discard line segments longer than this duration. 0 disables the limit.",
    )
    parser.add_argument(
        "--min-segment-notes",
        type=int,
        default=3,
        help="Discard line segments with fewer than this many overlapping notes.",
    )
    parser.add_argument(
        "--repeat-groupings",
        type=Path,
        default=None,
        help=(
            "Optional find_dali_repeats.py results directory or segments.jsonl.gz. "
            "Grouped songs retain all line occurrences and write lyric/melody class IDs."
        ),
    )
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
    if args.keep_file is not None:
        args.keep_file = resolve_user_path(args.keep_file)
    repeat_groupings_path = None
    repeat_groupings: dict[tuple[str, int], dict[str, Any]] = {}
    grouped_song_ids: set[str] = set()
    if args.repeat_groupings is not None:
        repeat_groupings_path = repeat_segments_path(args.repeat_groupings)
        repeat_groupings, grouped_song_ids = load_line_repeat_groupings(
            repeat_groupings_path
        )
        print(
            f"Loaded {len(repeat_groupings)} line group assignments for "
            f"{len(grouped_song_ids)} songs from {repeat_groupings_path}."
        )
    if args.prepared_audio_dir is None:
        args.prepared_audio_dir = args.output_dir / f"audio_{args.sample_rate // 1000}k"
    args.prepared_audio_dir = resolve_user_path(args.prepared_audio_dir)
    if args.max_segment_seconds < 0:
        raise SystemExit("--max-segment-seconds must be nonnegative")
    if args.min_segment_notes < 0:
        raise SystemExit("--min-segment-notes must be nonnegative")
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
    if args.keep_file is not None:
        keep_ids = set(read_keep_file(args.keep_file))
        if not keep_ids:
            raise SystemExit(f"Keep file contains no DALI ids: {args.keep_file}")
    if args.ground_truth_only:
        if args.gt_file is None:
            raise SystemExit("--ground-truth-only requires --gt-file")
        ground_truth_ids = read_ground_truth_ids(args.gt_file)
        keep_ids = (
            ground_truth_ids
            if keep_ids is None
            else keep_ids.intersection(ground_truth_ids)
        )
        if not keep_ids:
            raise SystemExit(
                "No DALI ids remain after applying --keep-file and "
                "--ground-truth-only"
            )

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
        lines = extract_dali_lines(entry)
        if not lines:
            skipped["no_lines"] += 1
            continue
        has_repeat_groupings = dali_id in grouped_song_ids
        if has_repeat_groupings:
            for line in lines:
                grouping = repeat_groupings.get((dali_id, int(line["line_index"])))
                if grouping is None:
                    raise SystemExit(
                        "Repeat groupings are incomplete for "
                        f"{dali_id} line {line['line_index']}"
                    )
                line["repeat_grouping"] = grouping

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
                "lines": lines,
                "has_repeat_groupings": has_repeat_groupings,
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
        line_segments, line_skips = unique_line_segments(
            track["lines"],
            audio_duration=track["duration"],
            trim_repeated_text=not track["has_repeat_groupings"],
        )
        skipped_segments.update(line_skips)
        for line in line_segments:
            start_seconds = line["start_seconds"]
            segment_seconds = line["segment_seconds"]
            note_count = count_segment_notes(
                track["notes"],
                start_seconds=start_seconds,
                segment_seconds=segment_seconds,
            )
            skip_reason = segment_quality_skip_reason(
                segment_seconds=segment_seconds,
                note_count=note_count,
                max_segment_seconds=args.max_segment_seconds,
                min_segment_notes=args.min_segment_notes,
            )
            if skip_reason is not None:
                skipped_segments[skip_reason] += 1
                continue
            (
                midi_pitches,
                onset_seconds,
                note_duration_seconds,
            ) = extract_segment_note_arrays(
                track["notes"],
                start_seconds=start_seconds,
                segment_seconds=segment_seconds,
            )
            voiced_ratio = note_coverage_ratio(
                track["notes"],
                start_seconds=start_seconds,
                segment_seconds=segment_seconds,
            )
            grouping = line.get("repeat_grouping", {})

            sample_id = stable_sample_id(
                track["dali_id"],
                start_seconds,
                line_index=line["line_index"],
            )
            melody_path = write_melody_npz(
                melody_dir=melody_dir,
                sample_id=sample_id,
                midi_pitches=midi_pitches,
                onset_seconds=onset_seconds,
                note_duration_seconds=note_duration_seconds,
                start_seconds=start_seconds,
                segment_seconds=segment_seconds,
            )
            segment = {
                "sample_id": sample_id,
                "dali_id": track["dali_id"],
                "melody_path": melody_path,
                "start_seconds": start_seconds,
                "end_seconds": line["end_seconds"],
                "segment_seconds": segment_seconds,
                "voiced_ratio": voiced_ratio,
                "line_index": line["line_index"],
                "line_text": line["text"],
                "lyric_class": grouping.get("lyric_class"),
                "melody_class": grouping.get("melody_class"),
                "parent_index": grouping.get("parent_index"),
                "transposition_to_representative": grouping.get(
                    "transposition_to_representative"
                ),
                "onset_mae_to_representative": grouping.get(
                    "onset_mae_to_representative"
                ),
                "duration_mae_to_representative": grouping.get(
                    "duration_mae_to_representative"
                ),
                "group_quality_flags": grouping.get("quality_flags", []),
                "note_count": note_count,
            }
            all_segments.append(segment)
            if args.max_segments > 0 and len(all_segments) >= args.max_segments:
                break

        if args.max_segments > 0 and len(all_segments) >= args.max_segments:
            break

    if not all_segments:
        raise SystemExit(
            "No eligible DALI line segments found. Check DALI/audio alignment."
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
        "keep_file": (
            manifest_relative_path(args.keep_file, metadata_path.parent)
            if args.keep_file is not None
            else None
        ),
        "prepared_audio_dir": manifest_relative_path(args.prepared_audio_dir, metadata_path.parent),
        "audio_format": args.audio_format,
        "skip_audio_prep": args.skip_audio_prep,
        "sample_rate": args.sample_rate,
        "segmentation_strategy": "one_dali_line_occurrence_per_segment",
        "repeated_line_policy": (
            "retain_grouped_occurrences_else_keep_first_normalized_text"
            if repeat_groupings_path is not None
            else "keep_first_casefolded_nfkc_text"
        ),
        "repeat_groupings": (
            manifest_relative_path(repeat_groupings_path, metadata_path.parent)
            if repeat_groupings_path is not None
            else None
        ),
        "num_grouped_songs_available": len(grouped_song_ids),
        "num_grouped_tracks_prepared": sum(
            bool(track["has_repeat_groupings"]) for track in usable_tracks
        ),
        "melody_representation": {
            "name": MELODY_REPRESENTATION,
            "level": "note",
            "feature_dimension": MELODY_FEATURE_DIM,
            "pitch_change_bins": 128,
            "duration_bins": 24,
            "onset_shift_bins": 24,
        },
        "max_segment_seconds": args.max_segment_seconds,
        "min_segment_notes": args.min_segment_notes,
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
