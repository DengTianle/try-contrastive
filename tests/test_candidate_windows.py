from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_contrastive import segment_length_probe_for_example
from prosodia.datasets import GroupedContrastiveDataset, grouped_contrastive_collate


class CandidateWindowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        sample_rate = 16_000
        self.audio_path = self.root / "song.wav"
        sf.write(
            self.audio_path,
            np.linspace(-0.5, 0.5, sample_rate * 12, dtype=np.float32),
            sample_rate,
        )
        self.rows = [
            self._row("long", 0.0, 3.0, melody_class=1, lyric_class=1),
            self._row("anchor", 4.0, 6.0, melody_class=2, lyric_class=2),
            self._row("short", 8.0, 9.0, melody_class=3, lyric_class=3),
        ]
        self.manifest_path = self.root / "manifest.csv"
        with self.manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _row(
        self,
        sample_id: str,
        start: float,
        end: float,
        melody_class: int,
        lyric_class: int,
    ) -> dict[str, str]:
        melody_path = self.root / f"{sample_id}.npz"
        np.savez_compressed(
            melody_path,
            f0_hz=np.asarray([440.0], dtype=np.float32),
            voiced=np.asarray([1], dtype=np.uint8),
            frame_times=np.asarray([0.0], dtype=np.float32),
        )
        return {
            "sample_id": sample_id,
            "split": "train",
            "dali_id": "song",
            "audio_path": self.audio_path.name,
            "raw_audio_path": self.audio_path.name,
            "melody_path": melody_path.name,
            "audio_duration_seconds": "12.0",
            "start_seconds": str(start),
            "end_seconds": str(end),
            "segment_seconds": str(end - start),
            "melody_class": str(melody_class),
            "lyric_class": str(lyric_class),
        }

    def test_match_positive_makes_all_observed_candidates_equal_length(self) -> None:
        dataset = GroupedContrastiveDataset(
            self.manifest_path,
            split="train",
            candidate_window_policy="match-positive",
            randomize_candidate_windows=False,
        )
        anchor_index = next(
            index
            for index, group in enumerate(dataset.groups)
            if group["anchor"]["sample_id"] == "anchor"
        )
        item = dataset[anchor_index]

        self.assertEqual(item["candidate_input_values"].shape, (3, 32_000))
        self.assertTrue(torch.all(item["candidate_window_seconds"] == 2.0))
        self.assertTrue(
            torch.all(item["candidate_audio_attention_mask"].sum(dim=-1) == 32_000)
        )
        batch = grouped_contrastive_collate([item])
        self.assertTrue(torch.all(batch["candidate_window_seconds"][0, :3] == 2.0))

    def test_negative_is_removed_when_context_would_cross_positive_class(self) -> None:
        repeat = self._row("repeat", 7.5, 9.0, melody_class=2, lyric_class=4)
        squeezed = self._row("squeezed", 6.2, 7.0, melody_class=4, lyric_class=5)
        rows = [*self.rows[:2], squeezed, repeat]
        with self.manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        dataset = GroupedContrastiveDataset(
            self.manifest_path,
            split="train",
            positive_variant_policy="self",
            candidate_window_policy="match-positive",
        )
        anchor_group = next(
            group for group in dataset.groups if group["anchor"]["sample_id"] == "anchor"
        )

        self.assertNotIn(
            "squeezed",
            [negative["sample_id"] for negative in anchor_group["negatives"]],
        )

    def test_epoch_changes_positive_variant_and_negative_subset_reproducibly(self) -> None:
        rows = [
            self._row("anchor", 0.0, 0.5, melody_class=1, lyric_class=1),
            self._row("variant_a", 1.0, 1.5, melody_class=1, lyric_class=2),
            self._row("variant_b", 2.0, 2.5, melody_class=1, lyric_class=3),
            self._row("variant_c", 3.0, 3.5, melody_class=1, lyric_class=4),
            *[
                self._row(
                    f"negative_{index}",
                    4.0 + index,
                    4.5 + index,
                    melody_class=10 + index,
                    lyric_class=10 + index,
                )
                for index in range(6)
            ],
        ]
        with self.manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        dataset = GroupedContrastiveDataset(
            self.manifest_path,
            split="train",
            max_negatives=2,
            positive_variant_policy="any",
            candidate_window_policy="line",
            seed=13,
        )
        anchor_index = next(
            index
            for index, group in enumerate(dataset.groups)
            if group["anchor"]["sample_id"] == "anchor"
        )

        selections: list[tuple[str, frozenset[str]]] = []
        epoch_zero_selection: tuple[list[str], int] | None = None
        for epoch in range(24):
            dataset.set_epoch(epoch)
            item = dataset[anchor_index]
            target = int(item["target"])
            positive_id = item["metadata"][target]["sample_id"]
            negative_ids = frozenset(
                row["sample_id"]
                for index, row in enumerate(item["metadata"])
                if index != target
            )
            selections.append((positive_id, negative_ids))
            if epoch == 0:
                epoch_zero_selection = (
                    [row["sample_id"] for row in item["metadata"]],
                    target,
                )

        self.assertGreater(len({positive for positive, _ in selections}), 1)
        self.assertGreater(len({negatives for _, negatives in selections}), 1)

        dataset.set_epoch(0)
        repeated_epoch_zero = dataset[anchor_index]
        self.assertEqual(
            (
                [row["sample_id"] for row in repeated_epoch_zero["metadata"]],
                int(repeated_epoch_zero["target"]),
            ),
            epoch_zero_selection,
        )


class DurationProbeTest(unittest.TestCase):
    def test_probe_uses_observed_candidate_windows(self) -> None:
        rows = [
            {"start_seconds": "0", "end_seconds": "2", "segment_seconds": "2"},
            {"start_seconds": "0", "end_seconds": "1", "segment_seconds": "1"},
            {"start_seconds": "0", "end_seconds": "3", "segment_seconds": "3"},
        ]
        probe = segment_length_probe_for_example(
            candidate_rows=rows,
            target_index=0,
            scores=torch.tensor([0.1, 0.3, 0.2]),
            anchor_row=rows[0],
            candidate_window_seconds=torch.tensor([2.0, 2.0, 2.0]),
        )

        self.assertAlmostEqual(probe["expected_top1_accuracy"], 1.0 / 3.0)
        self.assertEqual(probe["matching_candidates"], 3.0)
        self.assertTrue(np.isnan(probe["score_proximity_correlation"]))


if __name__ == "__main__":
    unittest.main()
