"""Small word annotations shared by preparation and stress-negative generation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


STRESS_METADATA_VERSION = 1
STRESS_METADATA_FIELDS = (
    "note_word_ids", "word_onset_seconds", "word_duration_seconds", "word_line_ids",
)


def build_stress_metadata(
    entry: Any,
    notes: list[dict[str, Any]],
    start_seconds: float,
    segment_seconds: float,
) -> dict[str, np.ndarray]:
    """Keep original word intervals, including words cut by segment boundaries.

    The note order is exactly extract_segment_note_arrays' order. Missing or
    inconsistent DALI parent links remain unknown; timing is not used to guess
    word membership.
    """
    words = entry.annotations.get("annot", {}).get("words", [])
    end_seconds = start_seconds + segment_seconds
    segment_notes = [
        note for note in notes
        if float(note["time"][-1]) > start_seconds
        and float(note["time"][0]) < end_seconds
    ]
    word_indices: list[int] = []
    for note in segment_notes:
        parent = note.get("index")
        if isinstance(parent, (int, np.integer)) and 0 <= parent < len(words):
            word_indices.append(int(parent))
    word_indices = sorted(set(word_indices))
    valid_words: list[tuple[int, float, float, int]] = []
    for index in word_indices:
        word = words[index]
        times = np.asarray(word.get("time", []), dtype=np.float64)
        line = word.get("index")
        if (
            times.size < 2 or not np.isfinite(times).all()
            or times[-1] <= times[0]
            or not isinstance(line, (int, np.integer)) or line < 0
        ):
            continue
        valid_words.append((index, float(times[0]), float(times[-1]), int(line)))
    local_ids = {word[0]: index for index, word in enumerate(valid_words)}
    note_word_ids = []
    for note in segment_notes:
        parent = note.get("index")
        local_id = local_ids.get(parent, -1) if isinstance(parent, (int, np.integer)) else -1
        if local_id >= 0:
            _, begin, end, _ = valid_words[local_id]
            # A generous millisecond tolerance only covers float conversion.
            if float(note["time"][0]) < begin - 0.001 or float(note["time"][-1]) > end + 0.001:
                local_id = -1
        note_word_ids.append(local_id)
    return {
        "format_version": np.asarray(STRESS_METADATA_VERSION, dtype=np.int64),
        "note_word_ids": np.asarray(note_word_ids, dtype=np.int64),
        "word_onset_seconds": np.asarray([w[1] - start_seconds for w in valid_words], dtype=np.float32),
        "word_duration_seconds": np.asarray([w[2] - w[1] for w in valid_words], dtype=np.float32),
        "word_line_ids": np.asarray([w[3] for w in valid_words], dtype=np.int64),
    }


def write_stress_metadata(path: Path, metadata: dict[str, np.ndarray]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **metadata)
    temporary.replace(path)
    return path.resolve(strict=False)


def load_stress_metadata(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        if int(source.get("format_version", -1)) != STRESS_METADATA_VERSION:
            raise ValueError(f"Unsupported stress metadata version: {path}")
        missing = set(STRESS_METADATA_FIELDS).difference(source.files)
        if missing:
            raise ValueError(f"Incomplete stress metadata {path}: missing {sorted(missing)}")
        result = {key: source[key].copy() for key in STRESS_METADATA_FIELDS}
    ids = result["note_word_ids"]
    starts = result["word_onset_seconds"]
    durations = result["word_duration_seconds"]
    lines = result["word_line_ids"]
    if (
        any(value.ndim != 1 for value in result.values())
        or starts.shape != durations.shape or starts.shape != lines.shape
        or not np.issubdtype(ids.dtype, np.integer)
        or not np.issubdtype(lines.dtype, np.integer)
        or not np.isfinite(starts).all() or not np.isfinite(durations).all()
        or np.any(durations <= 0) or np.any(lines < 0)
        or np.any(ids < -1) or np.any(ids >= len(starts))
    ):
        raise ValueError(f"Invalid word/note arrays in stress metadata: {path}")
    return result
