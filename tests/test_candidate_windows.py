from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_contrastive import segment_length_probe_for_example
from prosodia.datasets import GroupedContrastiveDataset, grouped_contrastive_collate
from prosodia.datasets import contrastive_dataset as dataset_module


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
            midi_pitches=np.asarray([69], dtype=np.int16),
            onset_seconds=np.asarray([0.0], dtype=np.float32),
            note_duration_seconds=np.asarray([end - start], dtype=np.float32),
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

        timing_by_id = {
            row["sample_id"]: (
                item["candidate_note_onsets"][index][
                    item["candidate_note_attention_mask"][index]
                ],
                item["candidate_note_durations"][index][
                    item["candidate_note_attention_mask"][index]
                ],
            )
            for index, row in enumerate(item["metadata"])
        }
        torch.testing.assert_close(timing_by_id["long"][0], torch.tensor([0.0]))
        torch.testing.assert_close(timing_by_id["long"][1], torch.tensor([2.0]))
        torch.testing.assert_close(timing_by_id["anchor"][0], torch.tensor([0.0]))
        torch.testing.assert_close(timing_by_id["anchor"][1], torch.tensor([2.0]))
        torch.testing.assert_close(timing_by_id["short"][0], torch.tensor([0.5]))
        torch.testing.assert_close(timing_by_id["short"][1], torch.tensor([1.0]))

        self.assertEqual(batch["candidate_note_onsets"].shape, (1, 3, 1))
        self.assertTrue(batch["candidate_note_attention_mask"][0, :3].all())

    def test_segment_policy_preserves_complete_prepared_intervals(self) -> None:
        dataset = GroupedContrastiveDataset(
            self.manifest_path,
            split="train",
            candidate_window_policy="segment",
            randomize_candidate_windows=False,
        )
        anchor_index = next(
            index
            for index, group in enumerate(dataset.groups)
            if group["anchor"]["sample_id"] == "anchor"
        )
        item = dataset[anchor_index]

        self.assertEqual(
            sorted(item["candidate_window_seconds"].tolist()),
            [1.0, 2.0, 3.0],
        )

    def test_match_positive_crop_avoids_gap_between_annotated_lines(self) -> None:
        # A one-second positive can fit entirely in the gap of this longer
        # two-line negative, including at the deterministic center crop.
        np.savez_compressed(
            self.root / "long.npz",
            midi_pitches=np.asarray([60, 64], dtype=np.int16),
            onset_seconds=np.asarray([0.0, 2.75], dtype=np.float32),
            note_duration_seconds=np.asarray([0.25, 0.25], dtype=np.float32),
        )
        for randomized in (False, True):
            with self.subTest(randomized=randomized):
                dataset = GroupedContrastiveDataset(
                    self.manifest_path,
                    candidate_window_policy="match-positive",
                    randomize_candidate_windows=randomized,
                )
                index = next(
                    i for i, group in enumerate(dataset.groups)
                    if group["anchor"]["sample_id"] == "short"
                )
                # Force random training crops into the same gap as evaluation.
                with patch.object(dataset_module.random, "randint", side_effect=lambda lo, hi: (lo + hi) // 2):
                    item = dataset[index]
                self.assertTrue(item["candidate_note_attention_mask"].any(dim=-1).all())
                self.assertTrue((item["candidate_window_seconds"] == 1.0).all())
                self.assertTrue(grouped_contrastive_collate([item])["candidate_notes_validated"])

                positive = next(row for row in dataset.rows if row["sample_id"] == "short")
                candidate = next(row for row in dataset.rows if row["sample_id"] == "long")
                for offset in (0, 4000, 16000, 28000, 32000):
                    with self.subTest(offset=offset), patch.object(
                        dataset, "_select_window_offset", return_value=offset,
                    ):
                        window = dataset._candidate_audio_window(
                            positive, candidate, [positive], 12 * 16000,
                        )
                        self.assertEqual(window[1], 16000)
                        self.assertGreaterEqual(window[0], 0)
                        self.assertLessEqual(sum(window), 3 * 16000)
                        onsets, durations = dataset._candidate_note_timing(candidate, window)
                        self.assertGreater(onsets.numel(), 0)
                        self.assertTrue((durations > 0).all())
                        if offset in (0, 32000):
                            self.assertEqual(window[0], offset)

    def test_match_positive_crop_reports_unannotated_candidate(self) -> None:
        np.savez_compressed(
            self.root / "long.npz",
            midi_pitches=np.asarray([60], dtype=np.int16),
            onset_seconds=np.asarray([0.0], dtype=np.float32),
            note_duration_seconds=np.asarray([0.0], dtype=np.float32),
        )
        dataset = GroupedContrastiveDataset(
            self.manifest_path, candidate_window_policy="match-positive",
        )
        positive = next(row for row in dataset.rows if row["sample_id"] == "short")
        candidate = next(row for row in dataset.rows if row["sample_id"] == "long")
        with self.assertRaisesRegex(ValueError, "No annotated notes.*long"):
            dataset._candidate_audio_window(positive, candidate, [positive], 12 * 16000)

    def test_candidate_timing_cache_keeps_window_clipping_dynamic(self) -> None:
        dataset = GroupedContrastiveDataset(self.manifest_path, split="train")
        row = next(row for row in dataset.rows if row["sample_id"] == "long")
        np.savez_compressed(
            row["melody_path"],
            midi_pitches=np.asarray([60, 62, 64], dtype=np.int16),
            onset_seconds=np.asarray([0.25, 1.25, 2.25], dtype=np.float32),
            note_duration_seconds=np.full(3, 0.5, dtype=np.float32),
        )
        with (
            patch.object(dataset_module.np, "load", wraps=np.load) as read,
            patch.object(dataset_module, "encode_note_sequence") as encode,
        ):
            onsets, durations = dataset._candidate_note_timing(row, (8000, 16000))
            torch.testing.assert_close(onsets, torch.tensor([0.0, 0.75]))
            torch.testing.assert_close(durations, torch.tensor([0.25, 0.25]))
            onsets.fill_(99)
            durations.zero_()

            dataset.set_epoch(1)
            onsets, durations = dataset._candidate_note_timing(row, (16000, 16000))
            torch.testing.assert_close(onsets, torch.tensor([0.25]))
            torch.testing.assert_close(durations, torch.tensor([0.5]))
            onsets, durations = dataset._candidate_note_timing(row, (0, 48000))
            torch.testing.assert_close(onsets, torch.tensor([0.25, 1.25, 2.25]))
            torch.testing.assert_close(durations, torch.full((3,), 0.5))
            self.assertEqual(read.call_count, 1)
            encode.assert_not_called()

    def test_candidate_timings_reuse_anchor_read_and_survive_item_mutation(self) -> None:
        dataset = GroupedContrastiveDataset(
            self.manifest_path, split="train", candidate_window_policy="match-positive",
        )
        index = next(
            i for i, group in enumerate(dataset.groups)
            if group["anchor"]["sample_id"] == "anchor"
        )
        with (
            patch.object(dataset_module.np, "load", wraps=np.load) as read,
            patch.object(
                dataset_module, "encode_note_sequence", wraps=dataset_module.encode_note_sequence,
            ) as encode,
        ):
            item = dataset[index]
            self.assertEqual(read.call_count, 3)  # One read per distinct segment.
            self.assertEqual(encode.call_count, 1)  # Only the anchor needs features.
            item["melody_note_onsets"].fill_(99)
            item["melody_note_durations"].zero_()
            repeated = dataset[index]
            self.assertEqual(read.call_count, 4)  # Only the new anchor read.
            self.assertEqual(encode.call_count, 2)

        for key in (
            "candidate_note_onsets", "candidate_note_durations", "candidate_note_attention_mask",
        ):
            torch.testing.assert_close(repeated[key], item[key])

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

    def test_rounding_infeasible_context_is_removed_before_loading(self) -> None:
        rows = [
            self._row(
                "protected_before",
                0.0,
                0.01849878271358074,
                melody_class=1,
                lyric_class=2,
            ),
            self._row(
                "candidate",
                0.12542893938926097,
                0.44621940941630167,
                melody_class=2,
                lyric_class=3,
            ),
            self._row(
                "protected_after",
                0.5531495660919818,
                0.7,
                melody_class=1,
                lyric_class=4,
            ),
            self._row(
                "anchor",
                1.6688398423883077,
                2.2034875747168066,
                melody_class=1,
                lyric_class=1,
            ),
        ]
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
        anchor_index = next(
            index
            for index, group in enumerate(dataset.groups)
            if group["anchor"]["sample_id"] == "anchor"
        )
        anchor_group = dataset.groups[anchor_index]

        self.assertNotIn(
            "candidate",
            [negative["sample_id"] for negative in anchor_group["negative_pool"]],
        )
        item = dataset[anchor_index]
        self.assertEqual(item["candidate_input_values"].shape[0], 1)

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
