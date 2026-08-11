from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prosodia.training import DifferentSongBatchSampler


class FakeGroupedDataset:
    def __init__(self, song_sizes: dict[str, int]) -> None:
        self.groups: list[dict[str, Any]] = []
        self.song_by_index: dict[int, str] = {}
        for song, size in song_sizes.items():
            for _ in range(size):
                index = len(self.groups)
                self.groups.append({"anchor": {"dali_id": song}})
                self.song_by_index[index] = song

    def __len__(self) -> int:
        return len(self.groups)


class DifferentSongBatchSamplerTest(unittest.TestCase):
    def assert_song_distinct(
        self,
        dataset: FakeGroupedDataset,
        batches: list[list[int]],
    ) -> None:
        for batch in batches:
            songs = [dataset.song_by_index[index] for index in batch]
            self.assertEqual(len(songs), len(set(songs)))

    def test_plans_exact_balanced_batches_without_losing_examples(self) -> None:
        dataset = FakeGroupedDataset({"a": 5, "b": 4, "c": 4, "d": 3, "e": 3})
        sampler = DifferentSongBatchSampler(dataset, batch_size=4, seed=13)

        batches = list(sampler)

        self.assertEqual(len(sampler), 5)
        self.assertEqual(len(batches), len(sampler))
        self.assertEqual(sorted(map(len, batches)), [3, 4, 4, 4, 4])
        self.assertEqual(
            sorted(index for batch in batches for index in batch),
            list(range(19)),
        )
        self.assert_song_distinct(dataset, batches)

    def test_largest_song_sets_batch_count_and_batches_remain_balanced(self) -> None:
        dataset = FakeGroupedDataset({"a": 9, "b": 7, "c": 5, "d": 3})
        sampler = DifferentSongBatchSampler(dataset, batch_size=4, seed=13)

        batches = list(sampler)

        self.assertEqual(len(sampler), 9)
        self.assertEqual(sorted(map(len, batches)), [2, 2, 2, 3, 3, 3, 3, 3, 3])
        self.assertEqual(
            sorted(index for batch in batches for index in batch),
            list(range(24)),
        )
        self.assert_song_distinct(dataset, batches)

    def test_epoch_plan_is_explicit_reproducible_and_changes_between_epochs(self) -> None:
        dataset = FakeGroupedDataset({"a": 5, "b": 5, "c": 5, "d": 5, "e": 5})
        sampler = DifferentSongBatchSampler(dataset, batch_size=4, seed=13)

        epoch_zero = list(sampler)
        self.assertEqual(list(sampler), epoch_zero)

        sampler.set_epoch(1)
        epoch_one = list(sampler)
        epoch_zero_partners = {
            index: frozenset(batch) - {index}
            for batch in epoch_zero
            for index in batch
        }
        epoch_one_partners = {
            index: frozenset(batch) - {index}
            for batch in epoch_one
            for index in batch
        }
        self.assertNotEqual(epoch_one_partners, epoch_zero_partners)
        self.assertEqual(sorted(map(len, epoch_one)), sorted(map(len, epoch_zero)))
        self.assert_song_distinct(dataset, epoch_one)

        sampler.set_epoch(0)
        self.assertEqual(list(sampler), epoch_zero)

    def test_drop_last_reports_and_yields_only_feasible_full_batches(self) -> None:
        dataset = FakeGroupedDataset({"a": 10, "b": 1, "c": 1})
        sampler = DifferentSongBatchSampler(
            dataset,
            batch_size=3,
            seed=13,
            drop_last=True,
        )

        batches = list(sampler)

        self.assertEqual(len(sampler), 1)
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]), 3)
        self.assert_song_distinct(dataset, batches)

    def test_rejects_negative_epoch(self) -> None:
        sampler = DifferentSongBatchSampler(
            FakeGroupedDataset({"a": 1}),
            batch_size=1,
            seed=13,
        )
        with self.assertRaisesRegex(ValueError, "epoch must be non-negative"):
            sampler.set_epoch(-1)


if __name__ == "__main__":
    unittest.main()
