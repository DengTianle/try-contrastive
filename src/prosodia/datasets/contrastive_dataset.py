from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    normalize_peak: bool = False


@dataclass(frozen=True)
class MelodyConfig:
    quantization_dir: str | Path | None = None
    events_filename: str = "events.jsonl"
    ratio_vocabulary_filename: str = "ratio_vocabulary.json"
    min_pitch_midi: int = 21
    max_pitch_midi: int = 108


@dataclass(frozen=True)
class QuantizedSongEvents:
    events: list[dict[str, Any]]


class NoteEventTokenizer:
    """Tokenize melody events as bounded MIDI-pitch/rest and duration-ratio IDs.

    Event class 0 is a rest. Note event classes 1..pitch_count correspond to
    chromatic MIDI pitches in the configured inclusive range. Note pitches
    outside that range are clipped to the nearest boundary before tokenization.
    """

    pad_token_id = 0
    mask_token_id = 1

    def __init__(
        self,
        ratios: list[str],
        min_pitch_midi: int = 21,
        max_pitch_midi: int = 108,
    ) -> None:
        if not ratios:
            raise ValueError("ratio vocabulary must not be empty")
        if len(set(ratios)) != len(ratios):
            raise ValueError("ratio vocabulary contains duplicate ratios")
        min_pitch_midi = int(min_pitch_midi)
        max_pitch_midi = int(max_pitch_midi)
        if not 0 <= min_pitch_midi <= max_pitch_midi <= 127:
            raise ValueError(
                "pitch range must satisfy 0 <= min_pitch_midi <= max_pitch_midi <= 127"
            )
        if max_pitch_midi - min_pitch_midi + 1 > 88:
            raise ValueError("pitch vocabulary must contain at most 88 note pitches")
        self.ratios = ratios
        self.ratio_to_id = {ratio: index for index, ratio in enumerate(ratios)}
        self.ratio_values = [float(Fraction(ratio)) for ratio in ratios]
        self.min_pitch_midi = min_pitch_midi
        self.max_pitch_midi = max_pitch_midi
        self.pitch_count = self.max_pitch_midi - self.min_pitch_midi + 1
        self.event_class_count = 1 + self.pitch_count
        self.vocab_size = 2 + (self.event_class_count * len(ratios))

    @property
    def ratio_count(self) -> int:
        return len(self.ratios)

    def quantize_pitch_midi(self, pitch_midi: Any) -> int:
        if pitch_midi is None:
            raise ValueError("Note event is missing pitch_midi")
        pitch_value = float(pitch_midi)
        if not math.isfinite(pitch_value):
            raise ValueError(f"Note event has non-finite pitch_midi: {pitch_midi}")
        rounded_pitch = int(math.floor(pitch_value + 0.5))
        clipped_pitch = min(max(rounded_pitch, self.min_pitch_midi), self.max_pitch_midi)
        return clipped_pitch - self.min_pitch_midi

    def token_id(self, onset: int, ratio: str, pitch_id: int | None = None) -> int:
        ratio_id = self.ratio_to_id[ratio]
        if int(onset) == 0:
            event_class_id = 0
        else:
            if pitch_id is None or not 0 <= pitch_id < self.pitch_count:
                raise ValueError(
                    f"Note pitch_id must be in [0, {self.pitch_count - 1}], got {pitch_id}"
                )
            event_class_id = 1 + pitch_id
        return 2 + (event_class_id * self.ratio_count) + ratio_id

    def encode_events(
        self,
        events: list[dict[str, Any]],
        start_seconds: float,
        end_seconds: float,
    ) -> dict[str, torch.Tensor]:
        token_ids: list[int] = []
        onset_ids: list[int] = []
        pitch_ids: list[int] = []
        ratio_ids: list[int] = []
        event_starts: list[float] = []
        event_ends: list[float] = []

        for event in events:
            event_start = float(event["start"])
            event_end = float(event["end"])
            if event_end <= start_seconds or event_start >= end_seconds:
                continue
            kind = str(event.get("kind", ""))
            if kind not in {"note", "rest"}:
                continue
            ratio = str(event.get("reference_ratio", ""))
            if ratio not in self.ratio_to_id:
                raise ValueError(f"Unknown quantized melody ratio: {ratio}")
            onset = 1 if kind == "note" else 0
            pitch_id = self.quantize_pitch_midi(event.get("pitch_midi")) if onset else -1
            token_ids.append(
                self.token_id(
                    onset=onset,
                    ratio=ratio,
                    pitch_id=pitch_id if onset else None,
                )
            )
            onset_ids.append(onset)
            pitch_ids.append(pitch_id)
            ratio_ids.append(self.ratio_to_id[ratio])
            event_starts.append(event_start)
            event_ends.append(event_end)

        return {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "onsets": torch.tensor(onset_ids, dtype=torch.long),
            "pitch_ids": torch.tensor(pitch_ids, dtype=torch.long),
            "ratio_ids": torch.tensor(ratio_ids, dtype=torch.long),
            "event_starts": torch.tensor(event_starts, dtype=torch.float32),
            "event_ends": torch.tensor(event_ends, dtype=torch.float32),
        }


def audio_start_seconds(row: dict[str, str]) -> float:
    return float(row["start_seconds"])


def melody_sample_id(row: dict[str, str]) -> str:
    return row["sample_id"]


def audio_sample_id(row: dict[str, str]) -> str:
    return row["sample_id"]


def resolve_manifest_path(path: str, manifest_dir: Path) -> str:
    path_str = str(path).lstrip("\\/")
    resolved = manifest_dir / path_str
    return str(resolved.resolve(strict=False))


def quantization_dir_for_manifest(manifest_dir: Path, melody_config: MelodyConfig) -> Path:
    if melody_config.quantization_dir is None:
        return manifest_dir / "quantization"
    quantization_dir = Path(melody_config.quantization_dir).expanduser()
    if not quantization_dir.is_absolute():
        quantization_dir = manifest_dir / quantization_dir
    return quantization_dir.resolve(strict=False)


def load_ratio_vocabulary(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Ratio vocabulary does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    ratios = data.get("ratios")
    if not isinstance(ratios, list):
        raise ValueError(f"Expected a 'ratios' list in {path}")
    return [str(item["ratio"]) for item in ratios]


def load_quantized_song_events(path: Path) -> dict[str, QuantizedSongEvents]:
    if not path.is_file():
        raise FileNotFoundError(f"Quantized melody events file does not exist: {path}")
    events_by_song: dict[str, QuantizedSongEvents] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            song_id = str(record.get("song_id", ""))
            if not song_id:
                raise ValueError(f"Missing song_id in {path}:{line_number}")
            events = record.get("events")
            if not isinstance(events, list):
                raise ValueError(f"Expected events list for {song_id} in {path}:{line_number}")
            events_by_song[song_id] = QuantizedSongEvents(
                events=sorted(events, key=lambda event: float(event["start"]))
            )
    return events_by_song


def stable_row_rng(seed: int, row_id: str) -> random.Random:
    digest = hashlib.sha1(f"{seed}:{row_id}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


class GroupedContrastiveDataset(Dataset[dict[str, Any]]):
    """Grouped view for InfoNCE training.

    Each item contains one melody anchor, one positive audio segment, and zero or
    more same-anchor negative audio segments. The positive candidate is always at
    index 0, so the training target is 0 for every grouped item.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str | None = None,
        audio_config: AudioConfig | None = None,
        melody_config: MelodyConfig | None = None,
        max_negatives: int | None = None,
        min_negative_offset_seconds: float | None = None,
        seed: int = 13,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve(strict=False)
        self.manifest_dir = self.manifest_path.parent
        self.audio_config = audio_config or AudioConfig()
        self.melody_config = melody_config or MelodyConfig()
        self.min_negative_offset_seconds = min_negative_offset_seconds
        self.seed = seed
        self.transform = transform
        self.quantization_dir = quantization_dir_for_manifest(self.manifest_dir, self.melody_config)
        self.tokenizer = NoteEventTokenizer(
            load_ratio_vocabulary(self.quantization_dir / self.melody_config.ratio_vocabulary_filename),
            min_pitch_midi=self.melody_config.min_pitch_midi,
            max_pitch_midi=self.melody_config.max_pitch_midi,
        )
        self.quantized_events_by_song = load_quantized_song_events(
            self.quantization_dir / self.melody_config.events_filename
        )

        with self.manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rows = [self._resolve_row_paths(row) for row in rows]

        if split is not None:
            rows = [row for row in rows if row["split"] == split]
        rows = [row for row in rows if self._row_has_quantized_melody(row)]
        self.rows = rows

        self.groups = self._build_groups_from_segment_rows(rows, max_negatives=max_negatives)

    def _resolve_row_paths(self, row: dict[str, str]) -> dict[str, str]:
        resolved = dict(row)
        for key in ("audio_path", "raw_audio_path"):
            if resolved.get(key):
                resolved[key] = resolve_manifest_path(resolved[key], self.manifest_dir)
        return resolved

    @property
    def melody_vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def melody_ratio_count(self) -> int:
        return self.tokenizer.ratio_count

    @property
    def melody_pitch_count(self) -> int:
        return self.tokenizer.pitch_count

    @property
    def melody_ratio_values(self) -> list[float]:
        return self.tokenizer.ratio_values

    def _row_has_quantized_melody(self, row: dict[str, str]) -> bool:
        song_events = self.quantized_events_by_song.get(row["dali_id"])
        if song_events is None:
            return False
        encoded = self.tokenizer.encode_events(
            song_events.events,
            start_seconds=float(row["start_seconds"]),
            end_seconds=float(row["end_seconds"]),
        )
        return encoded["token_ids"].numel() > 0

    def _build_groups_from_segment_rows(
        self,
        rows: list[dict[str, str]],
        max_negatives: int | None,
    ) -> list[dict[str, Any]]:
        rows_by_song: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            rows_by_song[row["dali_id"]].append(row)

        built_groups: list[dict[str, Any]] = []
        for song_rows in rows_by_song.values():
            song_rows.sort(key=lambda row: (audio_start_seconds(row), audio_sample_id(row)))
            for anchor in song_rows:
                min_offset = self.min_negative_offset_seconds
                if min_offset is None:
                    min_offset = float(anchor["segment_seconds"])
                negatives = [
                    candidate
                    for candidate in song_rows
                    if audio_sample_id(candidate) != audio_sample_id(anchor)
                    and abs(audio_start_seconds(candidate) - audio_start_seconds(anchor)) >= min_offset
                ]
                if max_negatives is not None and len(negatives) > max_negatives:
                    rng = stable_row_rng(self.seed, audio_sample_id(anchor))
                    negatives = list(negatives)
                    rng.shuffle(negatives)
                    negatives = sorted(
                        negatives[:max_negatives],
                        key=lambda row: (audio_start_seconds(row), audio_sample_id(row)),
                    )
                built_groups.append(
                    {
                        "melody_sample_id": melody_sample_id(anchor),
                        "positive": anchor,
                        "negatives": negatives,
                    }
                )
        return built_groups

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict[str, Any]:
        group = self.groups[index]
        positive = group["positive"]
        candidate_rows = [positive, *group["negatives"]]
        target = 0
        if len(candidate_rows) > 1:
            order = list(range(len(candidate_rows)))
            rng = stable_row_rng(self.seed, f"{group['melody_sample_id']}:candidate_order")
            rng.shuffle(order)
            target = order.index(0)
            candidate_rows = [candidate_rows[index] for index in order]
        melody = self._load_melody_tokens(positive)
        candidate_audio = self._load_candidate_audio(candidate_rows)
        candidate_ids = self._candidate_ids(positive, candidate_rows)
        candidate_types = self._candidate_types(positive, candidate_rows)

        item: dict[str, Any] = {
            "melody_token_ids": melody["token_ids"],
            "melody_onsets": melody["onsets"],
            "melody_pitch_ids": melody["pitch_ids"],
            "melody_ratio_ids": melody["ratio_ids"],
            "melody_event_starts": melody["event_starts"],
            "melody_event_ends": melody["event_ends"],
            "melody_attention_mask": torch.ones(
                melody["token_ids"].shape[0],
                dtype=torch.bool,
            ),
            "candidate_input_values": torch.stack(candidate_audio),
            "candidate_audio_attention_mask": torch.stack(
                [torch.ones_like(audio, dtype=torch.bool) for audio in candidate_audio]
            ),
            "candidate_mask": torch.ones(len(candidate_rows), dtype=torch.bool),
            "target": torch.tensor(target, dtype=torch.long),
            "melody_sample_id": group["melody_sample_id"],
            "candidate_ids": candidate_ids,
            "candidate_types": candidate_types,
            "metadata": candidate_rows,
        }
        if self.transform is not None:
            item = self.transform(item)
        return item

    def _load_melody_tokens(self, row: dict[str, str]) -> dict[str, torch.Tensor]:
        song_events = self.quantized_events_by_song.get(row["dali_id"])
        if song_events is None:
            raise ValueError(f"No quantized melody events found for song: {row['dali_id']}")
        melody = self.tokenizer.encode_events(
            song_events.events,
            start_seconds=float(row["start_seconds"]),
            end_seconds=float(row["end_seconds"]),
        )
        if melody["token_ids"].numel() == 0:
            raise ValueError(
                f"No quantized melody tokens for {row['sample_id']} "
                f"({row['start_seconds']}--{row['end_seconds']})"
            )
        return melody

    def _candidate_ids(
        self,
        positive: dict[str, str],
        candidate_rows: list[dict[str, str]],
    ) -> list[str]:
        anchor_id = melody_sample_id(positive)
        return [
            f"{anchor_id}__pos"
            if audio_sample_id(candidate) == audio_sample_id(positive)
            else f"{anchor_id}__same_song_neg{index}"
            for index, candidate in enumerate(candidate_rows)
        ]

    def _candidate_types(
        self,
        positive: dict[str, str],
        candidate_rows: list[dict[str, str]],
    ) -> list[str]:
        return [
            "positive"
            if audio_sample_id(candidate) == audio_sample_id(positive)
            else "hard_negative_same_song"
            for candidate in candidate_rows
        ]

    def _load_candidate_audio(self, candidate_rows: list[dict[str, str]]) -> list[torch.Tensor]:
        import soundfile as sf

        loaded_by_path: dict[str, tuple[np.ndarray, int]] = {}
        for audio_path, path_rows in self._rows_by_audio_path(candidate_rows).items():
            info = sf.info(audio_path)
            if info.samplerate != self.audio_config.sample_rate:
                raise ValueError(
                    f"Expected {self.audio_config.sample_rate} Hz prepared audio, "
                    f"got {info.samplerate} Hz: {audio_path}"
                )

            starts = [int(round(audio_start_seconds(row) * info.samplerate)) for row in path_rows]
            ends = [
                start + int(round(float(row["segment_seconds"]) * info.samplerate))
                for start, row in zip(starts, path_rows)
            ]
            read_start = max(min(starts), 0)
            read_end = max(ends)
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
            self._crop_loaded_audio(row, *loaded_by_path[str(Path(row["audio_path"]))])
            for row in candidate_rows
        ]

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
        row: dict[str, str],
        audio_array: np.ndarray,
        read_start_sample: int,
    ) -> torch.Tensor:
        start_sample = int(round(audio_start_seconds(row) * self.audio_config.sample_rate))
        expected_samples = int(round(float(row["segment_seconds"]) * self.audio_config.sample_rate))
        offset = max(start_sample - read_start_sample, 0)
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
        self.quantization_dir = quantization_dir_for_manifest(self.manifest_dir, self.melody_config)
        self.tokenizer = NoteEventTokenizer(
            load_ratio_vocabulary(self.quantization_dir / self.melody_config.ratio_vocabulary_filename),
            min_pitch_midi=self.melody_config.min_pitch_midi,
            max_pitch_midi=self.melody_config.max_pitch_midi,
        )
        self.quantized_events_by_song = load_quantized_song_events(
            self.quantization_dir / self.melody_config.events_filename
        )

        with self.manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rows = [self._resolve_row_paths(row) for row in rows]
        if split is not None:
            rows = [row for row in rows if row["split"] == split]
        rows = [row for row in rows if self._row_has_quantized_melody(row)]
        self.rows = rows

    def _resolve_row_paths(self, row: dict[str, str]) -> dict[str, str]:
        return dict(row)

    @property
    def melody_vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def melody_ratio_count(self) -> int:
        return self.tokenizer.ratio_count

    @property
    def melody_pitch_count(self) -> int:
        return self.tokenizer.pitch_count

    @property
    def melody_ratio_values(self) -> list[float]:
        return self.tokenizer.ratio_values

    def _row_has_quantized_melody(self, row: dict[str, str]) -> bool:
        song_events = self.quantized_events_by_song.get(row["dali_id"])
        if song_events is None:
            return False
        encoded = self.tokenizer.encode_events(
            song_events.events,
            start_seconds=float(row["start_seconds"]),
            end_seconds=float(row["end_seconds"]),
        )
        return encoded["token_ids"].numel() > 0

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        melody = self._load_melody_tokens(row)
        item: dict[str, Any] = {
            "melody_token_ids": melody["token_ids"],
            "melody_onsets": melody["onsets"],
            "melody_pitch_ids": melody["pitch_ids"],
            "melody_ratio_ids": melody["ratio_ids"],
            "melody_event_starts": melody["event_starts"],
            "melody_event_ends": melody["event_ends"],
            "melody_attention_mask": torch.ones(
                melody["token_ids"].shape[0],
                dtype=torch.bool,
            ),
            "melody_sample_id": melody_sample_id(row),
            "metadata": row,
        }
        if self.transform is not None:
            item = self.transform(item)
        return item

    def _load_melody_tokens(self, row: dict[str, str]) -> dict[str, torch.Tensor]:
        song_events = self.quantized_events_by_song.get(row["dali_id"])
        if song_events is None:
            raise ValueError(f"No quantized melody events found for song: {row['dali_id']}")
        melody = self.tokenizer.encode_events(
            song_events.events,
            start_seconds=float(row["start_seconds"]),
            end_seconds=float(row["end_seconds"]),
        )
        if melody["token_ids"].numel() == 0:
            raise ValueError(
                f"No quantized melody tokens for {row['sample_id']} "
                f"({row['start_seconds']}--{row['end_seconds']})"
            )
        return melody


def melody_only_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_tokens = max(item["melody_token_ids"].shape[0] for item in batch)
    batch_size = len(batch)

    melody_token_ids = torch.full(
        (batch_size, max_tokens),
        NoteEventTokenizer.pad_token_id,
        dtype=torch.long,
    )
    melody_onsets = torch.full((batch_size, max_tokens), -1, dtype=torch.long)
    melody_pitch_ids = torch.full((batch_size, max_tokens), -1, dtype=torch.long)
    melody_ratio_ids = torch.full((batch_size, max_tokens), -1, dtype=torch.long)
    melody_event_starts = torch.zeros(batch_size, max_tokens)
    melody_event_ends = torch.zeros(batch_size, max_tokens)
    melody_attention_mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool)

    for batch_index, item in enumerate(batch):
        token_count = item["melody_token_ids"].shape[0]
        melody_token_ids[batch_index, :token_count] = item["melody_token_ids"]
        melody_onsets[batch_index, :token_count] = item["melody_onsets"]
        melody_pitch_ids[batch_index, :token_count] = item["melody_pitch_ids"]
        melody_ratio_ids[batch_index, :token_count] = item["melody_ratio_ids"]
        melody_event_starts[batch_index, :token_count] = item["melody_event_starts"]
        melody_event_ends[batch_index, :token_count] = item["melody_event_ends"]
        melody_attention_mask[batch_index, :token_count] = item["melody_attention_mask"]

    return {
        "melody_token_ids": melody_token_ids,
        "melody_onsets": melody_onsets,
        "melody_pitch_ids": melody_pitch_ids,
        "melody_ratio_ids": melody_ratio_ids,
        "melody_event_starts": melody_event_starts,
        "melody_event_ends": melody_event_ends,
        "melody_attention_mask": melody_attention_mask,
        "melody_sample_id": [item["melody_sample_id"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }


def grouped_contrastive_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_candidates = max(item["candidate_input_values"].shape[0] for item in batch)
    batch_size = len(batch)
    sample_count = batch[0]["candidate_input_values"].shape[-1]

    candidate_input_values = torch.zeros(batch_size, max_candidates, sample_count)
    candidate_audio_attention_mask = torch.zeros(
        batch_size,
        max_candidates,
        sample_count,
        dtype=torch.bool,
    )
    candidate_mask = torch.zeros(batch_size, max_candidates, dtype=torch.bool)

    for batch_index, item in enumerate(batch):
        num_candidates = item["candidate_input_values"].shape[0]
        candidate_input_values[batch_index, :num_candidates] = item["candidate_input_values"]
        candidate_audio_attention_mask[batch_index, :num_candidates] = item[
            "candidate_audio_attention_mask"
        ]
        candidate_mask[batch_index, :num_candidates] = item["candidate_mask"]

    return {
        **melody_only_collate(batch),
        "candidate_input_values": candidate_input_values,
        "candidate_audio_attention_mask": candidate_audio_attention_mask,
        "candidate_mask": candidate_mask,
        "target": torch.stack([item["target"] for item in batch]),
        "candidate_ids": [item["candidate_ids"] for item in batch],
        "candidate_types": [item["candidate_types"] for item in batch],
    }
