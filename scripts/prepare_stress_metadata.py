"""Backfill word metadata for an existing note-level manifest without audio work."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from prepare_dali_dataset import (
    extract_dali_notes, extract_segment_note_arrays, load_dali,
    manifest_relative_path, resolve_user_path,
)
from prosodia.datasets.stress_metadata import build_stress_metadata, write_stress_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dali-data-dir", type=Path, default=Path("data/DALI_v1"))
    parser.add_argument("--gt-file", type=Path)
    parser.add_argument("--output-manifest", type=Path, help="Default: stress_segments_manifest.csv beside the source")
    args = parser.parse_args()
    source = resolve_user_path(args.manifest)
    destination = resolve_user_path(args.output_manifest) if args.output_manifest else source.with_name("stress_segments_manifest.csv")
    if source == destination:
        raise SystemExit("Use a separate output manifest so the source remains available")
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("The input manifest is empty")
    dataset = load_dali(
        resolve_user_path(args.dali_data_dir),
        resolve_user_path(args.gt_file) if args.gt_file else None,
        {row["dali_id"] for row in rows},
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    notes_by_song = {}
    counts = Counter()
    output_rows = []
    pending_metadata = []
    for row in rows:
        song = row["dali_id"]
        if song not in dataset:
            raise SystemExit(f"Missing DALI annotations for {song}")
        if song not in notes_by_song:
            notes_by_song[song] = extract_dali_notes(dataset[song])
        notes = notes_by_song[song]
        start = float(row["start_seconds"])
        duration = float(row["end_seconds"]) - start if row.get("end_seconds") else float(row["segment_seconds"])
        melody_path = source.parent / row["melody_path"]
        expected = extract_segment_note_arrays(notes, start, duration)
        # Reject mismatched annotation versions/offsets instead of attaching wrong
        # words to an existing melody. No original features are rewritten.
        with np.load(melody_path, allow_pickle=False) as melody:
            for key, values in zip(("midi_pitches", "onset_seconds", "note_duration_seconds"), expected):
                if key not in melody or melody[key].shape != values.shape or not np.allclose(melody[key], values, atol=1e-4, rtol=0):
                    raise SystemExit(
                        f"{melody_path}: {key} does not match DALI annotations. "
                        "Use the original DALI version/--gt-file, or rerun note-level preparation."
                    )
        metadata = build_stress_metadata(dataset[song], notes, start, duration)
        # Use the melody filename rather than arbitrary IDs as an output basename.
        path = destination.parent / f"{destination.stem}_words" / melody_path.name
        pending_metadata.append((path, metadata))
        updated = dict(row)
        for key in ("audio_path", "raw_audio_path", "melody_path", "prosody_target_path"):
            if updated.get(key):
                updated[key] = manifest_relative_path(source.parent / updated[key], destination.parent)
        updated["stress_metadata_path"] = manifest_relative_path(path, destination.parent)
        output_rows.append(updated)
        counts["segments"] += 1
        counts["notes"] += len(metadata["note_word_ids"])
        counts["notes_with_words"] += int((metadata["note_word_ids"] >= 0).sum())
    # Validate the complete source before replacing any sidecars from an earlier
    # run. The manifest-specific folder also avoids touching source sidecars.
    for path, metadata in pending_metadata:
        write_stress_metadata(path, metadata)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    temporary.replace(destination)
    print(json.dumps({"output_manifest": str(destination), **counts}, indent=2))


if __name__ == "__main__":
    main()
