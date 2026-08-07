#!/usr/bin/env python3
"""Find within-song lyric and melody equivalence classes in DALI.

Each line or paragraph occurrence is assigned independently to a lyric class
and, when it passes quality gates, a melody class. The tuple of these class IDs
defines a duplicate repeat. Melody classes with multiple lyric classes are
retexted melodies; lyric classes with multiple melody classes are remelodied
lyrics.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import DALI as dali_code


TOOL_VERSION = "1"
LEVEL_TO_DALI_KEY = {"line": "lines", "paragraph": "paragraphs"}
MELODY_BLOCKING_FLAGS = {
    "no_pitched_notes",
    "too_few_notes",
    "low_pitch_range",
    "low_voiced_coverage",
}


@dataclass(frozen=True)
class MelodyFeature:
    pitches: tuple[int, ...]
    pitch_intervals: tuple[int, ...]
    relative_onsets: tuple[float, ...]
    relative_durations: tuple[float, ...]
    pitch_range: float
    voiced_coverage: float

    @property
    def note_count(self) -> int:
        return len(self.pitches)


@dataclass
class Segment:
    index: int
    start: float
    end: float
    text: str
    parent_index: int | None
    lyric_key: str
    lyric_class: int = -1
    melody_class: int | None = None
    melody_feature: MelodyFeature | None = None
    quality_flags: list[str] = field(default_factory=list)
    onset_mae_to_representative: float | None = None
    duration_mae_to_representative: float | None = None
    transposition_to_representative: int | None = None


@dataclass
class MelodyCluster:
    class_id: int
    members: list[Segment]

    @property
    def representative(self) -> Segment:
        return self.members[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or inspect within-song DALI repeat equivalence classes."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build segment class assignments.")
    build.add_argument("--data-dir", type=Path, default=Path("data/DALI_v1"))
    build.add_argument(
        "--output-dir", type=Path, default=Path("outputs/repeated_segments")
    )
    build.add_argument(
        "--levels",
        nargs="+",
        choices=tuple(LEVEL_TO_DALI_KEY),
        default=list(LEVEL_TO_DALI_KEY),
    )
    build.add_argument("--keep", nargs="*", default=None)
    build.add_argument("--keep-file", type=Path, default=None)
    build.add_argument("--max-songs", type=int, default=None)
    build.add_argument("--min-notes", type=int, default=4)
    build.add_argument("--min-pitch-range", type=float, default=2.0)
    build.add_argument("--min-voiced-coverage", type=float, default=0.25)
    build.add_argument("--onset-tolerance", type=float, default=0.02)
    build.add_argument("--duration-tolerance", type=float, default=0.02)
    build.add_argument("--near-tolerance-fraction", type=float, default=0.8)

    inspect_parser = subparsers.add_parser(
        "inspect", help="Print nontrivial classes for one processed song."
    )
    inspect_parser.add_argument("--data-dir", type=Path, default=Path("data/DALI_v1"))
    inspect_parser.add_argument(
        "--results-dir", type=Path, default=Path("outputs/repeated_segments")
    )
    inspect_parser.add_argument("--song-id", required=True)
    inspect_parser.add_argument(
        "--levels",
        nargs="+",
        choices=tuple(LEVEL_TO_DALI_KEY),
        default=list(LEVEL_TO_DALI_KEY),
    )
    inspect_parser.add_argument(
        "--include-remelodied",
        action="store_true",
        help="Also print lyric classes connected to multiple melody classes.",
    )
    return parser.parse_args()


def normalized_lyric_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in normalized if character.isalnum())


def hz_to_midi(freq_hz: float) -> float:
    return 69.0 + 12.0 * math.log2(freq_hz / 440.0)


def mean_pitch_hz(freq_field: Any) -> float | None:
    if freq_field is None:
        return None
    if isinstance(freq_field, (int, float)):
        value = float(freq_field)
        return value if value > 0 else None
    values = [float(value) for value in freq_field if value is not None and value > 0]
    return sum(values) / len(values) if values else None


def overlap_duration(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def parent_paragraph_index(
    segment: dict[str, Any], paragraphs: list[dict[str, Any]]
) -> int | None:
    interval = (float(segment["time"][0]), float(segment["time"][1]))
    overlaps = [
        overlap_duration(
            interval, (float(paragraph["time"][0]), float(paragraph["time"][1]))
        )
        for paragraph in paragraphs
    ]
    if not overlaps or max(overlaps) <= 0:
        return None
    return max(range(len(overlaps)), key=overlaps.__getitem__)


def melody_feature_for_segment(
    segment: Segment, notes: list[dict[str, Any]]
) -> MelodyFeature | None:
    segment_duration = segment.end - segment.start
    if segment_duration <= 0:
        return None

    pitches: list[int] = []
    relative_onsets: list[float] = []
    relative_durations: list[float] = []
    covered_intervals: list[tuple[float, float]] = []
    for note in sorted(notes, key=lambda item: item["time"][0]):
        note_start, note_end = map(float, note["time"])
        clipped_start = max(segment.start, note_start)
        clipped_end = min(segment.end, note_end)
        if clipped_end <= clipped_start:
            continue
        pitch_hz = mean_pitch_hz(note.get("freq"))
        if pitch_hz is None:
            continue
        pitches.append(int(round(hz_to_midi(pitch_hz))))
        relative_onsets.append((clipped_start - segment.start) / segment_duration)
        relative_durations.append((clipped_end - clipped_start) / segment_duration)
        covered_intervals.append((clipped_start, clipped_end))

    if not pitches:
        return None

    merged: list[list[float]] = []
    for start, end in sorted(covered_intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    merged_coverage = sum(end - start for start, end in merged) / segment_duration

    intervals = tuple(
        pitches[index] - pitches[index - 1] for index in range(1, len(pitches))
    )
    return MelodyFeature(
        pitches=tuple(pitches),
        pitch_intervals=intervals,
        relative_onsets=tuple(relative_onsets),
        relative_durations=tuple(relative_durations),
        pitch_range=float(max(pitches) - min(pitches)),
        voiced_coverage=merged_coverage,
    )


def mean_absolute_difference(left: Iterable[float], right: Iterable[float]) -> float:
    pairs = list(zip(left, right))
    if not pairs:
        return 0.0
    return mean(abs(a - b) for a, b in pairs)


def timing_distance(left: Segment, right: Segment) -> tuple[float, float]:
    assert left.melody_feature is not None and right.melody_feature is not None
    return (
        mean_absolute_difference(
            left.melody_feature.relative_onsets,
            right.melody_feature.relative_onsets,
        ),
        mean_absolute_difference(
            left.melody_feature.relative_durations,
            right.melody_feature.relative_durations,
        ),
    )


def melody_signature(segment: Segment) -> tuple[int, tuple[int, ...]]:
    assert segment.melody_feature is not None
    return segment.melody_feature.note_count, segment.melody_feature.pitch_intervals


def classify_segments(
    raw_segments: list[dict[str, Any]],
    notes: list[dict[str, Any]],
    paragraphs: list[dict[str, Any]],
    level: str,
    min_notes: int,
    min_pitch_range: float,
    min_voiced_coverage: float,
    onset_tolerance: float,
    duration_tolerance: float,
) -> list[Segment]:
    segments: list[Segment] = []
    lyric_ids: dict[str, int] = {}
    for index, raw in enumerate(raw_segments):
        start, end = map(float, raw["time"])
        text = str(raw.get("text", ""))
        key = normalized_lyric_key(text)
        segment = Segment(
            index=index,
            start=start,
            end=end,
            text=text,
            parent_index=(
                parent_paragraph_index(raw, paragraphs) if level == "line" else None
            ),
            lyric_key=key,
        )
        if key:
            if key not in lyric_ids:
                lyric_ids[key] = len(lyric_ids)
            segment.lyric_class = lyric_ids[key]
        else:
            segment.quality_flags.append("empty_lyrics")
            segment.lyric_class = len(lyric_ids)
            lyric_ids[f"__empty_{index}"] = segment.lyric_class

        feature = melody_feature_for_segment(segment, notes)
        segment.melody_feature = feature
        if feature is None:
            segment.quality_flags.append("no_pitched_notes")
        else:
            if feature.note_count < min_notes:
                segment.quality_flags.append("too_few_notes")
            if feature.pitch_range < min_pitch_range:
                segment.quality_flags.append("low_pitch_range")
            if feature.voiced_coverage < min_voiced_coverage:
                segment.quality_flags.append("low_voiced_coverage")
        segments.append(segment)

    clusters_by_signature: dict[tuple[int, tuple[int, ...]], list[MelodyCluster]] = {}
    next_class_id = 0
    for segment in segments:
        if segment.melody_feature is None or any(
            flag in MELODY_BLOCKING_FLAGS for flag in segment.quality_flags
        ):
            continue
        signature = melody_signature(segment)
        candidate_clusters = clusters_by_signature.setdefault(signature, [])
        fits: list[tuple[float, MelodyCluster]] = []
        for cluster in candidate_clusters:
            pair_distances = [timing_distance(segment, member) for member in cluster.members]
            max_onset = max(distance[0] for distance in pair_distances)
            max_duration = max(distance[1] for distance in pair_distances)
            if max_onset <= onset_tolerance and max_duration <= duration_tolerance:
                normalized_error = (
                    max_onset / onset_tolerance + max_duration / duration_tolerance
                )
                fits.append((normalized_error, cluster))

        if fits:
            cluster = min(fits, key=lambda item: item[0])[1]
        else:
            cluster = MelodyCluster(class_id=next_class_id, members=[])
            next_class_id += 1
            candidate_clusters.append(cluster)

        representative = cluster.representative if cluster.members else segment
        onset_mae, duration_mae = timing_distance(segment, representative)
        segment.melody_class = cluster.class_id
        segment.onset_mae_to_representative = onset_mae
        segment.duration_mae_to_representative = duration_mae
        assert segment.melody_feature is not None
        assert representative.melody_feature is not None
        segment.transposition_to_representative = (
            segment.melody_feature.pitches[0]
            - representative.melody_feature.pitches[0]
        )
        cluster.members.append(segment)

    return segments


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def cross_class_pair_count(groups: dict[int, list[Segment]], attribute: str) -> int:
    total = 0
    for members in groups.values():
        counts: dict[int, int] = defaultdict(int)
        for member in members:
            value = getattr(member, attribute)
            if value is not None:
                counts[int(value)] += 1
        all_pairs = len(members) * (len(members) - 1) // 2
        within_pairs = sum(count * (count - 1) // 2 for count in counts.values())
        total += all_pairs - within_pairs
    return total


def song_level_stats(
    song_id: str,
    entry: Any,
    level: str,
    segments: list[Segment],
    onset_tolerance: float,
    duration_tolerance: float,
    min_notes: int,
    min_pitch_range: float,
    min_voiced_coverage: float,
    near_tolerance_fraction: float,
) -> dict[str, Any]:
    valid = [segment for segment in segments if segment.melody_class is not None]
    lyric_groups: dict[int, list[Segment]] = defaultdict(list)
    valid_lyric_groups: dict[int, list[Segment]] = defaultdict(list)
    melody_groups: dict[int, list[Segment]] = defaultdict(list)
    joint_groups: dict[tuple[int, int], list[Segment]] = defaultdict(list)
    for segment in segments:
        lyric_groups[segment.lyric_class].append(segment)
    for segment in valid:
        assert segment.melody_class is not None
        valid_lyric_groups[segment.lyric_class].append(segment)
        melody_groups[segment.melody_class].append(segment)
        joint_groups[(segment.lyric_class, segment.melody_class)].append(segment)

    duplicate_groups = [members for members in joint_groups.values() if len(members) > 1]
    retexted_groups = {
        class_id: members
        for class_id, members in melody_groups.items()
        if len({member.lyric_class for member in members}) > 1
    }
    remelodied_groups = {
        class_id: members
        for class_id, members in valid_lyric_groups.items()
        if len(
            {
                member.melody_class
                for member in members
                if member.melody_class is not None
            }
        )
        > 1
    }

    pitch_signature_groups: dict[tuple[int, tuple[int, ...]], list[Segment]] = defaultdict(list)
    for segment in valid:
        pitch_signature_groups[melody_signature(segment)].append(segment)
    split_signature_groups = 0
    same_signature_pairs = 0
    melody_equivalent_pairs = 0
    onset_pair_errors: list[float] = []
    duration_pair_errors: list[float] = []
    near_tolerance_pairs = 0
    for members in pitch_signature_groups.values():
        same_signature_pairs += len(members) * (len(members) - 1) // 2
        if len({member.melody_class for member in members}) > 1:
            split_signature_groups += 1
    for members in melody_groups.values():
        melody_equivalent_pairs += len(members) * (len(members) - 1) // 2
        for left, right in combinations(members, 2):
            onset_error, duration_error = timing_distance(left, right)
            onset_pair_errors.append(onset_error)
            duration_pair_errors.append(duration_error)
            if (
                onset_error >= near_tolerance_fraction * onset_tolerance
                or duration_error >= near_tolerance_fraction * duration_tolerance
            ):
                near_tolerance_pairs += 1

    duplicate_segments = sum(len(group) for group in duplicate_groups)
    duplicate_pairs = sum(len(group) * (len(group) - 1) // 2 for group in duplicate_groups)
    retexted_alternatives = sum(
        len({member.lyric_class for member in members}) - 1
        for members in retexted_groups.values()
    )
    remelodied_alternatives = sum(
        len({member.melody_class for member in members if member.melody_class is not None})
        - 1
        for members in remelodied_groups.values()
    )

    return {
        "song_id": song_id,
        "artist": entry.info.get("artist"),
        "title": entry.info.get("title"),
        "level": level,
        "segments": len(segments),
        "valid_melody_segments": len(valid),
        "unclassified_melody_segments": len(segments) - len(valid),
        "lyric_classes": len(lyric_groups),
        "melody_classes": len(melody_groups),
        "joint_classes": len(joint_groups),
        "lyric_folding_excess": len(segments) - len(lyric_groups),
        "melody_folding_excess": len(valid) - len(melody_groups),
        "joint_folding_excess": len(valid) - len(joint_groups),
        "nontrivial_duplicate_classes": len(duplicate_groups),
        "duplicate_segments": duplicate_segments,
        "duplicate_folding_excess": sum(len(group) - 1 for group in duplicate_groups),
        "duplicate_pairs": duplicate_pairs,
        "retexted_melody_classes": len(retexted_groups),
        "retexted_lyric_alternatives_excess": retexted_alternatives,
        "retexted_segments": sum(len(group) for group in retexted_groups.values()),
        "retexted_cross_lyric_pairs": cross_class_pair_count(
            retexted_groups, "lyric_class"
        ),
        "remelodied_lyric_classes": len(remelodied_groups),
        "remelodied_melody_alternatives_excess": remelodied_alternatives,
        "remelodied_segments": sum(len(group) for group in remelodied_groups.values()),
        "remelodied_cross_melody_pairs": cross_class_pair_count(
            remelodied_groups, "melody_class"
        ),
        "pitch_signature_groups": len(pitch_signature_groups),
        "pitch_signature_groups_split_by_timing": split_signature_groups,
        "same_pitch_signature_pairs": same_signature_pairs,
        "melody_equivalent_pairs": melody_equivalent_pairs,
        "timing_rejected_pairs": same_signature_pairs - melody_equivalent_pairs,
        "mean_within_class_onset_mae": mean(onset_pair_errors) if onset_pair_errors else 0.0,
        "p95_within_class_onset_mae": percentile(onset_pair_errors, 0.95),
        "max_within_class_onset_mae": max(onset_pair_errors, default=0.0),
        "mean_within_class_duration_mae": mean(duration_pair_errors) if duration_pair_errors else 0.0,
        "p95_within_class_duration_mae": percentile(duration_pair_errors, 0.95),
        "max_within_class_duration_mae": max(duration_pair_errors, default=0.0),
        "near_tolerance_pairs": near_tolerance_pairs,
        "onset_tolerance": onset_tolerance,
        "duration_tolerance": duration_tolerance,
        "near_tolerance_fraction": near_tolerance_fraction,
        "min_notes": min_notes,
        "min_pitch_range": min_pitch_range,
        "min_voiced_coverage": min_voiced_coverage,
        "dali_ncc": entry.info.get("scores", {}).get("NCC"),
        "dali_manual_score": entry.info.get("scores", {}).get("manual"),
    }


def rounded(value: Any) -> Any:
    return round(value, 6) if isinstance(value, float) else value


def segment_to_output(song_id: str, level: str, segment: Segment) -> dict[str, Any]:
    feature = segment.melody_feature
    return {
        "song_id": song_id,
        "level": level,
        "segment_index": segment.index,
        "parent_index": segment.parent_index,
        "start": round(segment.start, 6),
        "end": round(segment.end, 6),
        "lyric_class": segment.lyric_class,
        "melody_class": segment.melody_class,
        "note_count": feature.note_count if feature else 0,
        "pitch_range": round(feature.pitch_range, 3) if feature else 0.0,
        "voiced_coverage": round(feature.voiced_coverage, 6) if feature else 0.0,
        "onset_mae_to_representative": (
            round(segment.onset_mae_to_representative, 6)
            if segment.onset_mae_to_representative is not None
            else None
        ),
        "duration_mae_to_representative": (
            round(segment.duration_mae_to_representative, 6)
            if segment.duration_mae_to_representative is not None
            else None
        ),
        "transposition_to_representative": segment.transposition_to_representative,
        "quality_flags": segment.quality_flags,
    }


def read_keep_file(path: Path) -> list[str]:
    return [
        line.split("#", 1)[0].strip()
        for line in path.expanduser().resolve().read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    ]


def selected_song_ids(args: argparse.Namespace, data_dir: Path) -> list[str] | None:
    selected: list[str] = []
    if args.keep_file:
        selected.extend(read_keep_file(args.keep_file))
    if args.keep:
        selected.extend(args.keep)
    if selected:
        ids = list(dict.fromkeys(selected))
    elif args.max_songs is not None:
        ids = sorted(path.stem for path in data_dir.glob("*.gz"))
    else:
        return None
    return ids[: args.max_songs] if args.max_songs is not None else ids


def validate_build_args(args: argparse.Namespace) -> None:
    if args.min_notes < 1:
        raise ValueError("--min-notes must be at least 1")
    if args.min_pitch_range < 0 or not 0 <= args.min_voiced_coverage <= 1:
        raise ValueError("Pitch range must be nonnegative and coverage must be in [0, 1]")
    if args.onset_tolerance <= 0 or args.duration_tolerance <= 0:
        raise ValueError("Timing tolerances must be positive")
    if not 0 < args.near_tolerance_fraction <= 1:
        raise ValueError("--near-tolerance-fraction must be in (0, 1]")


def build_results(args: argparse.Namespace) -> None:
    validate_build_args(args)
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"DALI data directory does not exist: {data_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    keep = selected_song_ids(args, data_dir) or []
    dataset = dali_code.get_the_DALI_dataset(str(data_dir) + "/", keep=keep)

    segments_path = output_dir / "segments.jsonl.gz"
    stats_path = output_dir / "song_stats.csv"
    manifest_path = output_dir / "manifest.json"
    stats_rows: list[dict[str, Any]] = []
    segment_count = 0
    with gzip.open(segments_path, "wt", encoding="utf-8") as handle:
        for song_id, entry in sorted(dataset.items()):
            if entry.annotations["type"] != "horizontal":
                entry.vertical2horizontal()
            annotations = entry.annotations["annot"]
            notes = annotations["notes"]
            paragraphs = annotations["paragraphs"]
            for level in args.levels:
                raw_segments = annotations[LEVEL_TO_DALI_KEY[level]]
                segments = classify_segments(
                    raw_segments,
                    notes,
                    paragraphs,
                    level,
                    args.min_notes,
                    args.min_pitch_range,
                    args.min_voiced_coverage,
                    args.onset_tolerance,
                    args.duration_tolerance,
                )
                for segment in segments:
                    handle.write(
                        json.dumps(
                            segment_to_output(song_id, level, segment),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                segment_count += len(segments)
                stats_rows.append(
                    {
                        key: rounded(value)
                        for key, value in song_level_stats(
                            song_id,
                            entry,
                            level,
                            segments,
                            args.onset_tolerance,
                            args.duration_tolerance,
                            args.min_notes,
                            args.min_pitch_range,
                            args.min_voiced_coverage,
                            args.near_tolerance_fraction,
                        ).items()
                    }
                )

    if stats_rows:
        with stats_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(stats_rows[0]))
            writer.writeheader()
            writer.writerows(stats_rows)

    manifest = {
        "tool": "find_dali_repeats.py",
        "version": TOOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "levels": args.levels,
        "songs": len({row["song_id"] for row in stats_rows}),
        "segments": segment_count,
        "parameters": {
            "lyric_normalization": "NFKC + casefold + Unicode alphanumeric characters",
            "pitch_representation": "rounded MIDI pitch intervals",
            "melody_clustering": "exact pitch intervals + complete-link timing tolerances",
            "min_notes": args.min_notes,
            "min_pitch_range": args.min_pitch_range,
            "min_voiced_coverage": args.min_voiced_coverage,
            "onset_tolerance": args.onset_tolerance,
            "duration_tolerance": args.duration_tolerance,
            "near_tolerance_fraction": args.near_tolerance_fraction,
        },
        "outputs": {
            "segments": segments_path.name,
            "song_stats": stats_path.name,
        },
        "class_semantics": {
            "duplicate_repeat": "same lyric_class and melody_class, multiplicity > 1",
            "retexted_melody": "same melody_class with more than one lyric_class",
            "remelodied_lyrics": "same lyric_class with more than one melody_class",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Processed songs: {manifest['songs']}")
    print(f"Processed segments: {segment_count}")
    print(f"Segment classes: {segments_path}")
    print(f"Per-song stats: {stats_path}")
    print(f"Manifest: {manifest_path}")


def load_song_rows(results_dir: Path, song_id: str) -> list[dict[str, Any]]:
    path = results_dir.expanduser().resolve() / "segments.jsonl.gz"
    if not path.exists():
        raise FileNotFoundError(f"Repeat results do not exist: {path}")
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["song_id"] == song_id:
                rows.append(row)
    return rows


def load_song_stats(results_dir: Path, song_id: str) -> dict[str, dict[str, str]]:
    path = results_dir.expanduser().resolve() / "song_stats.csv"
    if not path.exists():
        raise FileNotFoundError(f"Song stats do not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["level"]: row
            for row in csv.DictReader(handle)
            if row["song_id"] == song_id
        }


def format_occurrence(row: dict[str, Any], text: str) -> str:
    diagnostics = ""
    if row["melody_class"] is not None:
        diagnostics = (
            f" d_on={row['onset_mae_to_representative']:.4f}"
            f" d_dur={row['duration_mae_to_representative']:.4f}"
            f" transpose={row['transposition_to_representative']:+d}"
        )
    return (
        f"    [{row['segment_index']:03d}] {row['start']:.2f}-{row['end']:.2f}s"
        f"{diagnostics}  {text}"
    )


def print_inspection(args: argparse.Namespace) -> None:
    results_dir = args.results_dir.expanduser().resolve()
    rows = load_song_rows(results_dir, args.song_id)
    if not rows:
        raise ValueError(f"Song ID was not found in the results: {args.song_id}")
    stats = load_song_stats(results_dir, args.song_id)
    data_dir = args.data_dir.expanduser().resolve()
    dataset = dali_code.get_the_DALI_dataset(str(data_dir) + "/", keep=[args.song_id])
    if args.song_id not in dataset:
        raise ValueError(f"Song ID was not found in DALI: {args.song_id}")
    entry = dataset[args.song_id]
    if entry.annotations["type"] != "horizontal":
        entry.vertical2horizontal()
    annotations = entry.annotations["annot"]

    print(f"{entry.info.get('artist')} - {entry.info.get('title')}")
    print(f"DALI ID: {args.song_id}")
    for level in args.levels:
        level_rows = [row for row in rows if row["level"] == level]
        if not level_rows:
            continue
        raw_segments = annotations[LEVEL_TO_DALI_KEY[level]]
        text_by_index = {
            index: str(segment.get("text", ""))
            for index, segment in enumerate(raw_segments)
        }
        valid = [row for row in level_rows if row["melody_class"] is not None]
        joint_groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        melody_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        lyric_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in valid:
            joint_groups[(row["lyric_class"], row["melody_class"])].append(row)
            melody_groups[row["melody_class"]].append(row)
            lyric_groups[row["lyric_class"]].append(row)

        print(f"\n{level.upper()} LEVEL")
        if level in stats:
            stat = stats[level]
            print(
                "  "
                f"segments={stat['segments']}, valid_melody={stat['valid_melody_segments']}, "
                f"duplicate_classes={stat['nontrivial_duplicate_classes']}, "
                f"retexted_melody_classes={stat['retexted_melody_classes']}, "
                f"remelodied_lyric_classes={stat['remelodied_lyric_classes']}"
            )
            print(
                "  "
                f"tolerances: onset={stat['onset_tolerance']}, "
                f"duration={stat['duration_tolerance']}; "
                f"timing_rejected_pairs={stat['timing_rejected_pairs']}, "
                f"near_tolerance_pairs={stat['near_tolerance_pairs']}"
            )

        duplicate_groups = [
            (key, members) for key, members in joint_groups.items() if len(members) > 1
        ]
        print(f"\n  DUPLICATE REPEATS ({len(duplicate_groups)} classes)")
        for (lyric_class, melody_class), members in duplicate_groups:
            print(
                f"  lyric L{lyric_class} + melody M{melody_class} "
                f"({len(members)} occurrences)"
            )
            for row in members:
                print(format_occurrence(row, text_by_index[row["segment_index"]]))

        retexted_groups = [
            (class_id, members)
            for class_id, members in melody_groups.items()
            if len({member["lyric_class"] for member in members}) > 1
        ]
        print(f"\n  RETEXTED MELODIES ({len(retexted_groups)} classes)")
        for melody_class, members in retexted_groups:
            alternatives: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for row in members:
                alternatives[row["lyric_class"]].append(row)
            print(
                f"  melody M{melody_class} "
                f"({len(alternatives)} lyric alternatives, {len(members)} occurrences)"
            )
            for lyric_class, alternative_rows in alternatives.items():
                print(f"    lyric L{lyric_class}")
                for row in alternative_rows:
                    print(format_occurrence(row, text_by_index[row["segment_index"]]))

        if args.include_remelodied:
            remelodied_groups = [
                (class_id, members)
                for class_id, members in lyric_groups.items()
                if len({member["melody_class"] for member in members}) > 1
            ]
            print(f"\n  REMELODIED LYRICS ({len(remelodied_groups)} classes)")
            for lyric_class, members in remelodied_groups:
                melodies: dict[int, list[dict[str, Any]]] = defaultdict(list)
                for row in members:
                    melodies[row["melody_class"]].append(row)
                print(
                    f"  lyric L{lyric_class} "
                    f"({len(melodies)} melody alternatives, {len(members)} occurrences)"
                )
                for melody_class, melody_rows in melodies.items():
                    print(f"    melody M{melody_class}")
                    for row in melody_rows:
                        print(format_occurrence(row, text_by_index[row["segment_index"]]))

        excluded = [row for row in level_rows if row["melody_class"] is None]
        if excluded:
            flag_counts: dict[str, int] = defaultdict(int)
            for row in excluded:
                for flag in row["quality_flags"]:
                    flag_counts[flag] += 1
            print(
                f"\n  UNCLASSIFIED MELODY ({len(excluded)} segments): "
                + ", ".join(f"{flag}={count}" for flag, count in sorted(flag_counts.items()))
            )


def main() -> None:
    args = parse_args()
    if args.command == "build":
        build_results(args)
    else:
        print_inspection(args)


if __name__ == "__main__":
    main()
