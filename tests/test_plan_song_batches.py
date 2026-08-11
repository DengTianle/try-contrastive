from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from plan_song_batches import plan_summary


class PlanSongBatchesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.manifest_path = Path(self.temporary_directory.name) / "manifest.csv"
        with self.manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "split", "dali_id"])
            writer.writeheader()
            index = 0
            for song, size in {"a": 5, "b": 4, "c": 4, "d": 3, "e": 3}.items():
                for _ in range(size):
                    writer.writerow(
                        {
                            "sample_id": str(index),
                            "split": "train",
                            "dali_id": song,
                        }
                    )
                    index += 1
            writer.writerow({"sample_id": str(index), "split": "val", "dali_id": "v"})

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_reports_balanced_incomplete_batches(self) -> None:
        summary = plan_summary(
            manifest_path=self.manifest_path,
            split="train",
            batch_size=4,
            seed=13,
            epoch=0,
            drop_incomplete_batches=False,
        )

        self.assertEqual(summary["input_examples"], 19)
        self.assertEqual(summary["planned_batches"], 5)
        self.assertEqual(summary["scheduled_examples"], 19)
        self.assertEqual(summary["dropped_examples"], 0)
        self.assertEqual(summary["batch_size_distribution"], {"3": 1, "4": 4})

    def test_reports_dropped_examples_for_full_batches(self) -> None:
        summary = plan_summary(
            manifest_path=self.manifest_path,
            split="train",
            batch_size=4,
            seed=13,
            epoch=0,
            drop_incomplete_batches=True,
        )

        self.assertEqual(summary["planned_batches"], 4)
        self.assertEqual(summary["scheduled_examples"], 16)
        self.assertEqual(summary["dropped_examples"], 3)
        self.assertEqual(summary["batch_size_distribution"], {"4": 4})


if __name__ == "__main__":
    unittest.main()
