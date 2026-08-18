from __future__ import annotations

import csv
import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from prosodia.melody_encoder import MELODY_FEATURE_DIM, encode_note_sequence


POSITIVE_VARIANT_POLICIES = {"self", "any", "retexted", "retexted-first"}
CANDIDATE_WINDOW_POLICIES = {"line", "match-positive"}


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    normalize_peak: bool = False
    minimum_input_samples: int = 400


@dataclass(frozen=True)
class MelodyConfig:
    feature_dim: int = MELODY_FEATURE_DIM


def audio_start_seconds(row: dict[str, str]) -> float:
    return float(row["start_seconds"])


def audio_end_seconds(row: dict[str, str]) -> float:
    if row.get("end_seconds"):
        return float(row["end_seconds"])
    return audio_start_seconds(row) + float(row["segment_seconds"])


def segment_duration_seconds(row: dict[str, str]) -> float:
    return audio_end_seconds(row) - audio_start_seconds(row)


def segments_overlap(first: dict[str, str], second: dict[str, str]) -> bool:
    return max(audio_start_seconds(first), audio_start_seconds(second)) < min(
        audio_end_seconds(first),
        audio_end_seconds(second),
    )


def melody_sample_id(row: dict[str, str]) -> str:
    return row["sample_id"]


def audio_sample_id(row: dict[str, str]) -> str:
    return row["sample_id"]


def optional_class_id(row: dict[str, str], key: str) -> int | None:
    value = row.get(key)
    if value in (None, "", "None", "null"):
        return None
    return int(value)


def same_melody_class(first: dict[str, str], second: dict[str, str]) -> bool:
    if first["dali_id"] != second["dali_id"]:
        return False
    first_class = optional_class_id(first, "melody_class")
    second_class = optional_class_id(second, "melody_class")
    return (
        first_class is not None
        and second_class is not None
        and first_class == second_class
    )


def same_lyric_class(first: dict[str, str], second: dict[str, str]) -> bool:
    if first["dali_id"] != second["dali_id"]:
        return False
    first_class = optional_class_id(first, "lyric_class")
    second_class = optional_class_id(second, "lyric_class")
    return (
        first_class is not None
        and second_class is not None
        and first_class == second_class
    )


def melody_equivalent(first: dict[str, str], second: dict[str, str]) -> bool:
    return (
        audio_sample_id(first) == audio_sample_id(second)
        or same_melody_class(first, second)
    )


def resolve_manifest_path(path: str, manifest_dir: Path) -> str:
    path_str = str(path).lstrip("\\/")
    resolved = manifest_dir / path_str
    return str(resolved.resolve(strict=False))


def load_melody(path: str | Path, melody_config: MelodyConfig) -> dict[str, torch.Tensor]:
    with np.load(path) as melody:
        required = {"midi_pitches", "onset_seconds", "note_duration_seconds"}
        missing = required.difference(melody.files)
        if missing:
            raise ValueError(
                f"{path} uses the obsolete frame-level melody format or is incomplete; "
                "rerun scripts/prepare_dali_dataset.py to create note-level features "
                f"(missing {sorted(missing)})"
            )
        midi_pitches = melody["midi_pitches"].astype(np.int64)
        onset_seconds = melody["onset_seconds"].astype(np.float32)
        duration_seconds = melody["note_duration_seconds"].astype(np.float32)

    features = encode_note_sequence(
        midi_pitches=midi_pitches,
        onset_seconds=onset_seconds,
        duration_seconds=duration_seconds,
    )
    if features.shape[1] != melody_config.feature_dim:
        raise ValueError(
            f"Expected {melody_config.feature_dim} melody features, got {features.shape[1]}"
        )

    return {
        "features": torch.from_numpy(features),
        "midi_pitches": torch.from_numpy(midi_pitches),
        "onset_seconds": torch.from_numpy(onset_seconds),
        "duration_seconds": torch.from_numpy(duration_seconds),
    }


def stable_row_rng(seed: int, row_id: str) -> random.Random:
    digest = hashlib.sha1(f"{seed}:{row_id}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


class GroupedContrastiveDataset(Dataset[dict[str, Any]]):
    """Grouped view for InfoNCE training.

    Each item contains one melody anchor, one positive audio segment, and zero or
    more same-song negative audio segments. When repeat-grouping columns are in the
    manifest, congruent melody occurrences are positive variants and are never used
    as negatives.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str | None = None,
        audio_config: AudioConfig | None = None,
        melody_config: MelodyConfig | None = None,
        max_negatives: int | None = None,
        min_negative_offset_seconds: float | None = None,
        positive_variant_policy: str = "self",
        candidate_window_policy: str = "line",
        randomize_candidate_windows: bool = False,
        seed: int = 13,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve(strict=False)
        self.manifest_dir = self.manifest_path.parent
        self.audio_config = audio_config or AudioConfig()
        self.melody_config = melody_config or MelodyConfig()
        self.min_negative_offset_seconds = min_negative_offset_seconds
        if positive_variant_policy not in POSITIVE_VARIANT_POLICIES:
            raise ValueError(
                "positive_variant_policy must be one of "
                f"{sorted(POSITIVE_VARIANT_POLICIES)}"
            )
        self.positive_variant_policy = positive_variant_policy
        if candidate_window_policy not in CANDIDATE_WINDOW_POLICIES:
            raise ValueError(
                "candidate_window_policy must be one of "
                f"{sorted(CANDIDATE_WINDOW_POLICIES)}"
            )
        self.candidate_window_policy = candidate_window_policy
        self.randomize_candidate_windows = randomize_candidate_windows
        self.seed = seed
        self.epoch = 0
        self.max_negatives = max_negatives
        self.transform = transform
        self._audio_frames_by_path: dict[str, int] = {}

        with self.manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rows = [self._resolve_row_paths(row) for row in rows]

        if split is not None:
            rows = [row for row in rows if row["split"] == split]
        if self.candidate_window_policy == "match-positive":
            self._cache_audio_metadata(rows)
        self.rows = rows

        self.groups = self._build_groups_from_segment_rows(rows)

    def _resolve_row_paths(self, row: dict[str, str]) -> dict[str, str]:
        resolved = dict(row)
        for key in ("audio_path", "raw_audio_path", "melody_path"):
            if resolved.get(key):
                resolved[key] = resolve_manifest_path(resolved[key], self.manifest_dir)
        return resolved

    def _audio_frame_count(self, audio_path: str | Path) -> int:
        path = str(Path(audio_path))
        if path not in self._audio_frames_by_path:
            import soundfile as sf

            info = sf.info(path)
            if info.samplerate != self.audio_config.sample_rate:
                raise ValueError(
                    f"Expected {self.audio_config.sample_rate} Hz prepared audio, "
                    f"got {info.samplerate} Hz: {path}"
                )
            self._audio_frames_by_path[path] = int(info.frames)
        return self._audio_frames_by_path[path]

    def _cache_audio_metadata(self, rows: list[dict[str, str]]) -> None:
        for audio_path, path_rows in self._rows_by_audio_path(rows).items():
            frame_count = self._audio_frame_count(audio_path)
            duration_seconds = frame_count / self.audio_config.sample_rate
            for row in path_rows:
                if not row.get("audio_duration_seconds"):
                    row["audio_duration_seconds"] = str(duration_seconds)

    def _build_groups_from_segment_rows(
        self,
        rows: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        rows_by_song: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            rows_by_song[row["dali_id"]].append(row)

        built_groups: list[dict[str, Any]] = []
        for song_rows in rows_by_song.values():
            song_rows.sort(key=lambda row: (audio_start_seconds(row), audio_sample_id(row)))
            for anchor in song_rows:
                positive_variants = self._positive_variant_pool(anchor, song_rows)
                protected_rows = [
                    candidate
                    for candidate in song_rows
                    if self._possible_false_negative(anchor, candidate)
                ]
                negative_pool = [
                    candidate
                    for candidate in song_rows
                    if not self._possible_false_negative(anchor, candidate)
                    and any(
                        self._valid_negative(
                            anchor,
                            positive,
                            candidate,
                            protected_rows,
                        )
                        for positive in positive_variants
                    )
                ]
                built_groups.append(
                    {
                        "melody_sample_id": melody_sample_id(anchor),
                        "anchor": anchor,
                        "positive_variants": positive_variants,
                        "protected_rows": protected_rows,
                        "negative_pool": negative_pool,
                    }
                )
                positive, negatives = self._select_group_candidates(built_groups[-1])
                built_groups[-1]["positive"] = positive
                built_groups[-1]["negatives"] = negatives
        return built_groups

    def _positive_variant_pool(
        self,
        anchor: dict[str, str],
        song_rows: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        if self.positive_variant_policy == "self":
            return [anchor]

        alternatives = [
            candidate
            for candidate in song_rows
            if audio_sample_id(candidate) != audio_sample_id(anchor)
            and same_melody_class(anchor, candidate)
        ]
        retexted = [
            candidate
            for candidate in alternatives
            if not same_lyric_class(anchor, candidate)
        ]
        if self.positive_variant_policy == "retexted":
            pool = retexted
        elif self.positive_variant_policy == "retexted-first":
            pool = retexted or alternatives
        else:
            pool = alternatives
        if not pool:
            return [anchor]
        return pool

    def _selection_rng(self, row_id: str) -> random.Random:
        if self.epoch > 0:
            row_id = f"{row_id}:epoch:{self.epoch}"
        return stable_row_rng(self.seed, row_id)

    def _select_positive_variant(
        self,
        anchor: dict[str, str],
        positive_variants: list[dict[str, str]],
    ) -> dict[str, str]:
        if len(positive_variants) == 1:
            return positive_variants[0]
        rng = self._selection_rng(
            f"{audio_sample_id(anchor)}:positive_variant:{self.positive_variant_policy}",
        )
        return positive_variants[rng.randrange(len(positive_variants))]

    def _possible_false_negative(
        self,
        anchor: dict[str, str],
        candidate: dict[str, str],
    ) -> bool:
        if melody_equivalent(anchor, candidate):
            return True
        if same_lyric_class(anchor, candidate) and (
            optional_class_id(anchor, "melody_class") is None
            or optional_class_id(candidate, "melody_class") is None
        ):
            return True
        return False

    def _valid_negative(
        self,
        anchor: dict[str, str],
        positive: dict[str, str],
        candidate: dict[str, str],
        protected_rows: list[dict[str, str]],
    ) -> bool:
        if self.min_negative_offset_seconds is None:
            valid_offset = not segments_overlap(anchor, candidate)
        else:
            valid_offset = (
                abs(audio_start_seconds(candidate) - audio_start_seconds(anchor))
                >= self.min_negative_offset_seconds
            )
        return valid_offset and self._matching_window_is_feasible(
            positive,
            candidate,
            protected_rows,
        )

    def _matching_window_is_feasible(
        self,
        positive: dict[str, str],
        candidate: dict[str, str],
        protected_rows: list[dict[str, str]],
    ) -> bool:
        if self.candidate_window_policy == "line":
            return True
        target_samples = self._segment_sample_count(positive)
        if self._segment_sample_count(candidate) >= target_samples:
            return True

        lower_sample, upper_sample = self._safe_window_start_sample_bounds(
            candidate=candidate,
            protected_rows=protected_rows,
            target_samples=target_samples,
            total_audio_samples=self._audio_frame_count(candidate["audio_path"]),
        )
        return lower_sample <= upper_sample

    def _segment_sample_bounds(
        self,
        row: dict[str, str],
    ) -> tuple[int, int]:
        sample_rate = self.audio_config.sample_rate
        return (
            int(round(audio_start_seconds(row) * sample_rate)),
            int(round(audio_end_seconds(row) * sample_rate)),
        )

    def _segment_sample_count(self, row: dict[str, str]) -> int:
        start_sample, end_sample = self._segment_sample_bounds(row)
        return max(1, end_sample - start_sample)

    def __len__(self) -> int:
        return len(self.groups)

    def set_epoch(self, epoch: int) -> None:
        """Select reproducible positive variants and negatives for an epoch."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def _select_group_candidates(
        self,
        group: dict[str, Any],
    ) -> tuple[dict[str, str], list[dict[str, str]]]:
        anchor = group["anchor"]
        positive = self._select_positive_variant(anchor, group["positive_variants"])
        negatives = [
            candidate
            for candidate in group["negative_pool"]
            if self._valid_negative(
                anchor,
                positive,
                candidate,
                group["protected_rows"],
            )
        ]
        if self.max_negatives is not None and len(negatives) > self.max_negatives:
            rng = self._selection_rng(audio_sample_id(anchor))
            rng.shuffle(negatives)
            negatives = sorted(
                negatives[: self.max_negatives],
                key=lambda row: (audio_start_seconds(row), audio_sample_id(row)),
            )
        return positive, negatives

    def positive_variant_counts(self) -> dict[str, int]:
        counts = {"aligned": 0, "repeated": 0, "retexted": 0}
        for group in self.groups:
            positive = self._select_positive_variant(
                group["anchor"],
                group["positive_variants"],
            )
            variant = self._positive_variant_type(group["anchor"], positive)
            counts[variant] += 1
        return counts

    def __getitem__(self, index: int) -> dict[str, Any]:
        group = self.groups[index]
        anchor = group["anchor"]
        positive, negatives = self._select_group_candidates(group)
        candidate_rows = [positive, *negatives]
        target = 0
        if len(candidate_rows) > 1:
            order = list(range(len(candidate_rows)))
            rng = self._selection_rng(f"{group['melody_sample_id']}:candidate_order")
            rng.shuffle(order)
            target = order.index(0)
            candidate_rows = [candidate_rows[index] for index in order]
        melody = load_melody(anchor["melody_path"], self.melody_config)
        candidate_audio = self._load_candidate_audio(
            positive=positive,
            candidate_rows=candidate_rows,
            protected_rows=group["protected_rows"],
        )
        candidate_input_values, candidate_audio_attention_mask = pad_candidate_audio(
            candidate_audio,
            minimum_samples=self.audio_config.minimum_input_samples,
        )
        candidate_ids = self._candidate_ids(anchor, positive, candidate_rows)
        candidate_types = self._candidate_types(anchor, positive, candidate_rows)

        item: dict[str, Any] = {
            "melody_features": melody["features"],
            "melody_midi_pitches": melody["midi_pitches"],
            "melody_note_onsets": melody["onset_seconds"],
            "melody_note_durations": melody["duration_seconds"],
            "melody_attention_mask": torch.ones(
                melody["features"].shape[0],
                dtype=torch.bool,
            ),
            "candidate_input_values": candidate_input_values,
            "candidate_audio_attention_mask": candidate_audio_attention_mask,
            "candidate_window_seconds": torch.tensor(
                [audio.shape[0] / self.audio_config.sample_rate for audio in candidate_audio],
                dtype=torch.float32,
            ),
            "candidate_mask": torch.ones(len(candidate_rows), dtype=torch.bool),
            "target": torch.tensor(target, dtype=torch.long),
            "melody_sample_id": group["melody_sample_id"],
            "candidate_ids": candidate_ids,
            "candidate_types": candidate_types,
            "anchor_metadata": anchor,
            "metadata": candidate_rows,
        }
        if self.transform is not None:
            item = self.transform(item)
        return item

    def _candidate_ids(
        self,
        anchor: dict[str, str],
        positive: dict[str, str],
        candidate_rows: list[dict[str, str]],
    ) -> list[str]:
        anchor_id = melody_sample_id(anchor)
        return [
            f"{anchor_id}__pos_{self._positive_variant_type(anchor, positive)}"
            if audio_sample_id(candidate) == audio_sample_id(positive)
            else f"{anchor_id}__same_song_neg{index}"
            for index, candidate in enumerate(candidate_rows)
        ]

    def _candidate_types(
        self,
        anchor: dict[str, str],
        positive: dict[str, str],
        candidate_rows: list[dict[str, str]],
    ) -> list[str]:
        return [
            f"positive_{self._positive_variant_type(anchor, positive)}"
            if audio_sample_id(candidate) == audio_sample_id(positive)
            else "hard_negative_same_song"
            for candidate in candidate_rows
        ]

    def _positive_variant_type(
        self,
        anchor: dict[str, str],
        positive: dict[str, str],
    ) -> str:
        if audio_sample_id(anchor) == audio_sample_id(positive):
            return "aligned"
        if same_lyric_class(anchor, positive):
            return "repeated"
        return "retexted"

    def _load_candidate_audio(
        self,
        positive: dict[str, str],
        candidate_rows: list[dict[str, str]],
        protected_rows: list[dict[str, str]],
    ) -> list[torch.Tensor]:
        import soundfile as sf

        loaded_by_path: dict[str, tuple[np.ndarray, int]] = {}
        windows_by_sample_id: dict[str, tuple[int, int]] = {}
        for audio_path, path_rows in self._rows_by_audio_path(candidate_rows).items():
            total_audio_samples = self._audio_frame_count(audio_path)

            path_windows = [
                self._candidate_audio_window(
                    positive=positive,
                    candidate=row,
                    protected_rows=protected_rows,
                    total_audio_samples=total_audio_samples,
                )
                for row in path_rows
            ]
            for row, window in zip(path_rows, path_windows):
                windows_by_sample_id[audio_sample_id(row)] = window
            starts = [window[0] for window in path_windows]
            ends = [start + expected for start, expected in path_windows]
            read_start = max(min(starts), 0)
            read_end = min(max(ends), total_audio_samples)
            audio, _ = sf.read(
                audio_path,
                start=read_start,
                frames=max(read_end - read_start, 0),
                dtype="float32",
                always_2d=False,
            )
            audio_array = np.asarray(audio, dtype=np.float32)
            if audio_array.ndim == 2:
                audio_array = np.mean(audio_array, axis=1, dtype=np.float32)
            loaded_by_path[audio_path] = (audio_array, read_start)

        return [
            self._crop_loaded_audio(
                *loaded_by_path[str(Path(row["audio_path"]))],
                *windows_by_sample_id[audio_sample_id(row)],
            )
            for row in candidate_rows
        ]

    def _candidate_audio_window(
        self,
        positive: dict[str, str],
        candidate: dict[str, str],
        protected_rows: list[dict[str, str]],
        total_audio_samples: int,
    ) -> tuple[int, int]:
        candidate_start, candidate_end = self._segment_sample_bounds(candidate)
        candidate_samples = self._segment_sample_count(candidate)
        if self.candidate_window_policy == "line":
            return max(candidate_start, 0), candidate_samples

        target_samples = self._segment_sample_count(positive)
        if candidate_samples >= target_samples:
            maximum_offset = candidate_samples - target_samples
            offset = self._select_window_offset(maximum_offset)
            return candidate_start + offset, target_samples

        lower_sample, upper_sample = self._safe_window_start_sample_bounds(
            candidate=candidate,
            protected_rows=protected_rows,
            target_samples=target_samples,
            total_audio_samples=total_audio_samples,
        )
        if lower_sample > upper_sample:
            raise RuntimeError(
                "No safe context window can match the positive duration for "
                f"candidate {audio_sample_id(candidate)}"
            )
        offset = self._select_window_offset(upper_sample - lower_sample)
        return lower_sample + offset, target_samples

    def _safe_window_start_sample_bounds(
        self,
        candidate: dict[str, str],
        protected_rows: list[dict[str, str]],
        target_samples: int,
        total_audio_samples: int,
    ) -> tuple[int, int]:
        sample_rate = self.audio_config.sample_rate
        candidate_start, candidate_end = self._segment_sample_bounds(candidate)
        safe_region_start = 0
        safe_region_end = total_audio_samples
        for protected in protected_rows:
            if audio_sample_id(protected) == audio_sample_id(candidate):
                continue
            if audio_end_seconds(protected) <= audio_start_seconds(candidate):
                safe_region_start = max(
                    safe_region_start,
                    int(round(audio_end_seconds(protected) * sample_rate)),
                )
            elif audio_start_seconds(protected) >= audio_end_seconds(candidate):
                safe_region_end = min(
                    safe_region_end,
                    int(round(audio_start_seconds(protected) * sample_rate)),
                )
            elif segments_overlap(protected, candidate):
                return 1, 0

        return (
            max(safe_region_start, candidate_end - target_samples),
            min(candidate_start, safe_region_end - target_samples),
        )

    def _select_window_offset(self, maximum_offset: int) -> int:
        if maximum_offset <= 0:
            return 0
        if self.randomize_candidate_windows:
            return random.randint(0, maximum_offset)
        return maximum_offset // 2

    def _rows_by_audio_path(
        self,
        rows: list[dict[str, str]],
    ) -> dict[str, list[dict[str, str]]]:
        rows_by_path: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            rows_by_path[str(Path(row["audio_path"]))].append(row)
        return rows_by_path

    def _crop_loaded_audio(
        self,
        audio_array: np.ndarray,
        read_start_sample: int,
        window_start_sample: int,
        expected_samples: int,
    ) -> torch.Tensor:
        offset = max(window_start_sample - read_start_sample, 0)
        crop = audio_array[offset : offset + expected_samples]

        if crop.shape[0] < expected_samples:
            crop = np.pad(crop, (0, expected_samples - crop.shape[0]))
        elif crop.shape[0] > expected_samples:
            crop = crop[:expected_samples]

        if self.audio_config.normalize_peak:
            peak = float(np.max(np.abs(crop))) if crop.size else 0.0
            if peak > 0:
                crop = crop / peak

        return torch.from_numpy(crop.astype(np.float32, copy=True))


def pad_candidate_audio(
    candidate_audio: list[torch.Tensor],
    minimum_samples: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not candidate_audio:
        raise ValueError("Expected at least one candidate audio segment")
    if minimum_samples <= 0:
        raise ValueError("minimum_samples must be positive")
    max_samples = max(
        minimum_samples,
        max(audio.shape[0] for audio in candidate_audio),
    )
    values = candidate_audio[0].new_zeros(len(candidate_audio), max_samples)
    attention_mask = torch.zeros(len(candidate_audio), max_samples, dtype=torch.bool)
    for candidate_index, audio in enumerate(candidate_audio):
        sample_count = audio.shape[0]
        values[candidate_index, :sample_count] = audio
        attention_mask[candidate_index, :sample_count] = True
    return values, attention_mask


class MelodyOnlyDataset(Dataset[dict[str, Any]]):
    """Manifest-backed melody segments without loading paired audio."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str | None = None,
        melody_config: MelodyConfig | None = None,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve(strict=False)
        self.manifest_dir = self.manifest_path.parent
        self.melody_config = melody_config or MelodyConfig()
        self.transform = transform

        with self.manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rows = [self._resolve_row_paths(row) for row in rows]
        if split is not None:
            rows = [row for row in rows if row["split"] == split]
        self.rows = rows

    def _resolve_row_paths(self, row: dict[str, str]) -> dict[str, str]:
        resolved = dict(row)
        if resolved.get("melody_path"):
            resolved["melody_path"] = resolve_manifest_path(resolved["melody_path"], self.manifest_dir)
        return resolved

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        melody = load_melody(row["melody_path"], self.melody_config)
        item: dict[str, Any] = {
            "melody_features": melody["features"],
            "melody_midi_pitches": melody["midi_pitches"],
            "melody_note_onsets": melody["onset_seconds"],
            "melody_note_durations": melody["duration_seconds"],
            "melody_attention_mask": torch.ones(
                melody["features"].shape[0],
                dtype=torch.bool,
            ),
            "melody_sample_id": melody_sample_id(row),
            "metadata": row,
        }
        if self.transform is not None:
            item = self.transform(item)
        return item


def melody_only_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_notes = max(item["melody_features"].shape[0] for item in batch)
    batch_size = len(batch)
    feature_dim = batch[0]["melody_features"].shape[-1]

    melody_features = torch.zeros(batch_size, max_notes, feature_dim)
    melody_midi_pitches = torch.zeros(batch_size, max_notes, dtype=torch.long)
    melody_note_onsets = torch.zeros(batch_size, max_notes)
    melody_note_durations = torch.zeros(batch_size, max_notes)
    melody_attention_mask = torch.zeros(batch_size, max_notes, dtype=torch.bool)

    for batch_index, item in enumerate(batch):
        note_count = item["melody_features"].shape[0]
        melody_features[batch_index, :note_count] = item["melody_features"]
        melody_midi_pitches[batch_index, :note_count] = item["melody_midi_pitches"]
        melody_note_onsets[batch_index, :note_count] = item["melody_note_onsets"]
        melody_note_durations[batch_index, :note_count] = item["melody_note_durations"]
        melody_attention_mask[batch_index, :note_count] = item["melody_attention_mask"]

    return {
        "melody_features": melody_features,
        "melody_midi_pitches": melody_midi_pitches,
        "melody_note_onsets": melody_note_onsets,
        "melody_note_durations": melody_note_durations,
        "melody_attention_mask": melody_attention_mask,
        "melody_sample_id": [item["melody_sample_id"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }


def grouped_contrastive_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_candidates = max(item["candidate_input_values"].shape[0] for item in batch)
    batch_size = len(batch)
    max_samples = max(item["candidate_input_values"].shape[-1] for item in batch)

    candidate_input_values = torch.zeros(batch_size, max_candidates, max_samples)
    candidate_audio_attention_mask = torch.zeros(
        batch_size,
        max_candidates,
        max_samples,
        dtype=torch.bool,
    )
    candidate_mask = torch.zeros(batch_size, max_candidates, dtype=torch.bool)
    candidate_window_seconds = torch.zeros(batch_size, max_candidates)

    for batch_index, item in enumerate(batch):
        num_candidates = item["candidate_input_values"].shape[0]
        sample_count = item["candidate_input_values"].shape[-1]
        candidate_input_values[
            batch_index,
            :num_candidates,
            :sample_count,
        ] = item["candidate_input_values"]
        candidate_audio_attention_mask[
            batch_index,
            :num_candidates,
            :sample_count,
        ] = item[
            "candidate_audio_attention_mask"
        ]
        candidate_mask[batch_index, :num_candidates] = item["candidate_mask"].to(
            dtype=torch.bool
        )
        candidate_window_seconds[batch_index, :num_candidates] = item[
            "candidate_window_seconds"
        ]

    melody_batch = melody_only_collate(batch)

    return {
        "melody_features": melody_batch["melody_features"],
        "melody_attention_mask": melody_batch["melody_attention_mask"],
        "melody_midi_pitches": melody_batch["melody_midi_pitches"],
        "melody_note_onsets": melody_batch["melody_note_onsets"],
        "melody_note_durations": melody_batch["melody_note_durations"],
        "candidate_input_values": candidate_input_values,
        "candidate_audio_attention_mask": candidate_audio_attention_mask,
        "candidate_mask": candidate_mask,
        "candidate_window_seconds": candidate_window_seconds,
        "target": torch.stack([item["target"] for item in batch]),
        "melody_sample_id": [item["melody_sample_id"] for item in batch],
        "candidate_ids": [item["candidate_ids"] for item in batch],
        "candidate_types": [item["candidate_types"] for item in batch],
        "anchor_metadata": [item["anchor_metadata"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }
