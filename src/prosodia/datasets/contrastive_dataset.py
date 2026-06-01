from __future__ import annotations

import csv
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
        start_seconds = float(row["audio_start_seconds"])
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
        with np.load(row["melody_path"]) as melody:
            f0_hz = melody["f0_hz"].astype(np.float32)
            voiced = melody["voiced"].astype(bool)
            frame_times = melody["frame_times"].astype(np.float32)

        log_f0 = np.zeros_like(f0_hz, dtype=np.float32)
        valid = voiced & np.isfinite(f0_hz) & (f0_hz > 0.0)
        log_f0[valid] = np.log2(f0_hz[valid] / self.melody_config.log_f0_reference_hz)

        feature_parts = [log_f0[:, None]]
        if self.melody_config.include_voiced_feature:
            feature_parts.append(voiced.astype(np.float32)[:, None])
        features = np.concatenate(feature_parts, axis=1).astype(np.float32)

        return {
            "features": torch.from_numpy(features),
            "f0_hz": torch.from_numpy(f0_hz),
            "voiced": torch.from_numpy(voiced),
            "frame_times": torch.from_numpy(frame_times),
        }


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
