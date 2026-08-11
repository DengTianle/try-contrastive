from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from prepare_dali_dataset import read_keep_file, segment_quality_skip_reason


class KeepFileTest(unittest.TestCase):
    def test_reads_one_id_per_line_and_ignores_comments_and_blanks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            keep_file = Path(directory) / "keep.txt"
            keep_file.write_text(
                "song-a\n\n  song-b  # explanation\n# comment\nsong-a\n",
                encoding="utf-8",
            )

            self.assertEqual(read_keep_file(keep_file), ["song-a", "song-b", "song-a"])


class SegmentQualityTest(unittest.TestCase):
    def test_accepts_segment_at_duration_and_note_boundaries(self) -> None:
        self.assertIsNone(
            segment_quality_skip_reason(
                segment_seconds=10.0,
                note_count=3,
                max_segment_seconds=10.0,
                min_segment_notes=3,
            )
        )

    def test_rejects_long_segment_before_rendering(self) -> None:
        self.assertEqual(
            segment_quality_skip_reason(10.001, 8, 10.0, 3),
            "too_long",
        )

    def test_rejects_segment_with_too_few_notes(self) -> None:
        self.assertEqual(
            segment_quality_skip_reason(2.0, 2, 10.0, 3),
            "too_few_notes",
        )

    def test_zero_max_duration_disables_only_the_duration_limit(self) -> None:
        self.assertIsNone(segment_quality_skip_reason(700.0, 3, 0.0, 3))


if __name__ == "__main__":
    unittest.main()
