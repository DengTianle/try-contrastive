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


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    normalize_peak: bool = False


@dataclass(frozen=True)
class MelodyConfig:
    log_f0_reference_hz: float = 440.0
    include_voiced_feature: bool = True


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


def load_melody(path: str | Path, melody_config: MelodyConfig) -> dict[str, torch.Tensor]:
    with np.load(path) as melody:
        f0_hz = melody["f0_hz"].astype(np.float32)
        voiced = melody["voiced"].astype(bool)
        frame_times = melody["frame_times"].astype(np.float32)

    log_f0 = np.zeros_like(f0_hz, dtype=np.float32)
    valid = voiced & np.isfinite(f0_hz) & (f0_hz > 0.0)
    log_f0[valid] = np.log2(f0_hz[valid] / melody_config.log_f0_reference_hz)

    feature_parts = [log_f0[:, None]]
    if melody_config.include_voiced_feature:
        feature_parts.append(voiced.astype(np.float32)[:, None])
    features = np.concatenate(feature_parts, axis=1).astype(np.float32)

    return {
        "features": torch.from_numpy(features),
        "f0_hz": torch.from_numpy(f0_hz),
        "voiced": torch.from_numpy(voiced),
        "frame_times": torch.from_numpy(frame_times),
    }


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

        with self.manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rows = [self._resolve_row_paths(row) for row in rows]

        if split is not None:
            rows = [row for row in rows if row["split"] == split]
        self.rows = rows

        self.groups = self._build_groups_from_segment_rows(rows, max_negatives=max_negatives)

    def _resolve_row_paths(self, row: dict[str, str]) -> dict[str, str]:
        resolved = dict(row)
        for key in ("audio_path", "raw_audio_path", "melody_path"):
            if resolved.get(key):
                resolved[key] = resolve_manifest_path(resolved[key], self.manifest_dir)
        return resolved

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
        melody = load_melody(positive["melody_path"], self.melody_config)
        candidate_audio = self._load_candidate_audio(candidate_rows)
        candidate_ids = self._candidate_ids(positive, candidate_rows)
        candidate_types = self._candidate_types(candidate_rows)

        item: dict[str, Any] = {
            "melody_features": melody["features"],
            "melody_f0_hz": melody["f0_hz"],
            "melody_voiced": melody["voiced"],
            "melody_frame_times": melody["frame_times"],
            "melody_attention_mask": torch.ones(
                melody["features"].shape[0],
                dtype=torch.bool,
            ),
            "candidate_input_values": torch.stack(candidate_audio),
            "candidate_audio_attention_mask": torch.stack(
                [torch.ones_like(audio, dtype=torch.bool) for audio in candidate_audio]
            ),
            "candidate_mask": torch.ones(len(candidate_rows), dtype=torch.bool),
            "target": torch.tensor(0, dtype=torch.long),
            "melody_sample_id": group["melody_sample_id"],
            "candidate_ids": candidate_ids,
            "candidate_types": candidate_types,
            "metadata": candidate_rows,
        }
        if self.transform is not None:
            item = self.transform(item)
        return item

    def _candidate_ids(
        self,
        positive: dict[str, str],
        candidate_rows: list[dict[str, str]],
    ) -> list[str]:
        anchor_id = melody_sample_id(positive)
        return [
            f"{anchor_id}__pos",
            *[
                f"{anchor_id}__same_song_neg{index}"
                for index in range(len(candidate_rows) - 1)
            ],
        ]

    def _candidate_types(self, candidate_rows: list[dict[str, str]]) -> list[str]:
        return ["positive", *["hard_negative_same_song"] * (len(candidate_rows) - 1)]

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
        "melody_features": torch.stack([item["melody_features"] for item in batch]),
        "melody_attention_mask": torch.stack([item["melody_attention_mask"] for item in batch]),
        "melody_f0_hz": torch.stack([item["melody_f0_hz"] for item in batch]),
        "melody_voiced": torch.stack([item["melody_voiced"] for item in batch]),
        "melody_frame_times": torch.stack([item["melody_frame_times"] for item in batch]),
        "candidate_input_values": candidate_input_values,
        "candidate_audio_attention_mask": candidate_audio_attention_mask,
        "candidate_mask": candidate_mask,
        "target": torch.stack([item["target"] for item in batch]),
        "melody_sample_id": [item["melody_sample_id"] for item in batch],
        "candidate_ids": [item["candidate_ids"] for item in batch],
        "candidate_types": [item["candidate_types"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }
