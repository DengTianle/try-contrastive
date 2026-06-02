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


class ContrastivePairDataset(Dataset[dict[str, Any]]):
    """Dataset for prepared melody/audio contrastive pairs.

    Expected manifest columns come from scripts/prepare_dali_dataset.py. Audio files
    should already be mono and resampled by prep; this class crops and pads only.
    Melody tensors are frame-aligned fixed-length arrays saved as .npz.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str | None = None,
        pair_types: set[str] | None = None,
        audio_config: AudioConfig | None = None,
        melody_config: MelodyConfig | None = None,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve(strict=False)
        self.audio_config = audio_config or AudioConfig()
        self.melody_config = melody_config or MelodyConfig()
        self.transform = transform

        with self.manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        if split is not None:
            rows = [row for row in rows if row["split"] == split]
        if pair_types is not None:
            rows = [row for row in rows if row["pair_type"] in pair_types]

        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        waveform = self._load_audio_crop(row)
        melody = self._load_melody(row)

        item: dict[str, Any] = {
            "input_values": waveform,
            "audio_attention_mask": torch.ones_like(waveform, dtype=torch.bool),
            "melody_features": melody["features"],
            "melody_f0_hz": melody["f0_hz"],
            "melody_voiced": melody["voiced"],
            "melody_frame_times": melody["frame_times"],
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "pair_type": row["pair_type"],
            "pair_id": row["pair_id"],
            "dali_id": row["dali_id"],
            "metadata": row,
        }
        if self.transform is not None:
            item = self.transform(item)
        return item

    def _load_audio_crop(self, row: dict[str, str]) -> torch.Tensor:
        import soundfile as sf

        audio_path = Path(row["audio_path"])
        start_seconds = audio_start_seconds(row)
        segment_seconds = float(row["segment_seconds"])
        expected_samples = int(round(segment_seconds * self.audio_config.sample_rate))

        info = sf.info(str(audio_path))
        if info.samplerate != self.audio_config.sample_rate:
            raise ValueError(
                f"Expected {self.audio_config.sample_rate} Hz prepared audio, "
                f"got {info.samplerate} Hz: {audio_path}"
            )

        start_sample = int(round(start_seconds * info.samplerate))
        audio, _ = sf.read(
            str(audio_path),
            start=start_sample,
            frames=expected_samples,
            dtype="float32",
            always_2d=False,
        )
        audio_array = np.asarray(audio, dtype=np.float32)
        if audio_array.ndim == 2:
            audio_array = np.mean(audio_array, axis=1, dtype=np.float32)

        if audio_array.shape[0] < expected_samples:
            audio_array = np.pad(audio_array, (0, expected_samples - audio_array.shape[0]))
        elif audio_array.shape[0] > expected_samples:
            audio_array = audio_array[:expected_samples]

        if self.audio_config.normalize_peak:
            peak = float(np.max(np.abs(audio_array))) if audio_array.size else 0.0
            if peak > 0:
                audio_array = audio_array / peak

        return torch.from_numpy(audio_array.copy())

    def _load_melody(self, row: dict[str, str]) -> dict[str, torch.Tensor]:
        return load_melody(row["melody_path"], self.melody_config)


def audio_start_seconds(row: dict[str, str]) -> float:
    if "audio_start_seconds" in row:
        return float(row["audio_start_seconds"])
    return float(row["start_seconds"])


def melody_sample_id(row: dict[str, str]) -> str:
    return row.get("melody_sample_id") or row["sample_id"]


def audio_sample_id(row: dict[str, str]) -> str:
    return row.get("audio_sample_id") or row["sample_id"]


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
        self.audio_config = audio_config or AudioConfig()
        self.melody_config = melody_config or MelodyConfig()
        self.min_negative_offset_seconds = min_negative_offset_seconds
        self.seed = seed
        self.transform = transform

        with self.manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        if split is not None:
            rows = [row for row in rows if row["split"] == split]
        self.rows = rows

        fieldnames = set(rows[0]) if rows else set()
        self.manifest_kind = "pair" if {"pair_type", "label", "melody_sample_id"} <= fieldnames else "segment"
        if self.manifest_kind == "pair":
            self.groups = self._build_groups_from_pair_rows(rows, max_negatives=max_negatives)
        else:
            self.groups = self._build_groups_from_segment_rows(rows, max_negatives=max_negatives)

    def _build_groups_from_pair_rows(
        self,
        rows: list[dict[str, str]],
        max_negatives: int | None,
    ) -> list[dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = melody_sample_id(row)
            group = groups.setdefault(key, {"positive": None, "negatives": []})
            if int(row["label"]) == 1:
                group["positive"] = row
            else:
                group["negatives"].append(row)

        built_groups: list[dict[str, Any]] = []
        for group_melody_sample_id, group in groups.items():
            if group["positive"] is None:
                continue
            negative_rows = group["negatives"]
            if max_negatives is not None:
                negative_rows = negative_rows[:max_negatives]
            built_groups.append(
                {
                    "melody_sample_id": group_melody_sample_id,
                    "positive": group["positive"],
                    "negatives": negative_rows,
                }
            )
        return built_groups

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
        pair_ids = self._candidate_pair_ids(positive, candidate_rows)
        pair_types = self._candidate_pair_types(candidate_rows)

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
            "candidate_pair_ids": pair_ids,
            "candidate_pair_types": pair_types,
            "metadata": candidate_rows,
        }
        if self.transform is not None:
            item = self.transform(item)
        return item

    def _candidate_pair_ids(
        self,
        positive: dict[str, str],
        candidate_rows: list[dict[str, str]],
    ) -> list[str]:
        if self.manifest_kind == "pair":
            return [row["pair_id"] for row in candidate_rows]
        anchor_id = melody_sample_id(positive)
        return [
            f"{anchor_id}__pos",
            *[
                f"{anchor_id}__same_song_neg{index}"
                for index in range(len(candidate_rows) - 1)
            ],
        ]

    def _candidate_pair_types(self, candidate_rows: list[dict[str, str]]) -> list[str]:
        if self.manifest_kind == "pair":
            return [row["pair_type"] for row in candidate_rows]
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


def contrastive_pair_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = {
        "input_values",
        "audio_attention_mask",
        "melody_features",
        "melody_f0_hz",
        "melody_voiced",
        "melody_frame_times",
        "label",
    }
    collated: dict[str, Any] = {}
    for key in tensor_keys:
        collated[key] = torch.stack([item[key] for item in batch])

    for key in ("pair_type", "pair_id", "dali_id", "metadata"):
        collated[key] = [item[key] for item in batch]
    return collated


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
        "candidate_pair_ids": [item["candidate_pair_ids"] for item in batch],
        "candidate_pair_types": [item["candidate_pair_types"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }
