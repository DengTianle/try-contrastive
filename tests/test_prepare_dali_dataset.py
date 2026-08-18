from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prepare_dali_dataset import (
    extract_segment_note_arrays,
    note_coverage_ratio,
    read_keep_file,
    segment_quality_skip_reason,
)
from prosodia.melody_encoder import (
    DURATION_SLICE,
    MELODY_FEATURE_DIM,
    ONSET_SHIFT_SLICE,
    PITCH_CHANGE_SLICE,
    PITCH_SIGN_INDEX,
    encode_note_sequence,
)


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

    def test_note_representation_always_rejects_empty_segments(self) -> None:
        self.assertEqual(segment_quality_skip_reason(2.0, 0, 10.0, 0), "too_few_notes")


class NoteRepresentationTest(unittest.TestCase):
    def test_encoding_matches_reference_layout_and_segment_normalization(self) -> None:
        features = encode_note_sequence(
            midi_pitches=np.asarray([60, 64, 55]),
            onset_seconds=np.asarray([0.0, 0.5, 1.5]),
            duration_seconds=np.asarray([0.25, 0.5, 1.0]),
        )

        self.assertEqual(features.shape, (3, MELODY_FEATURE_DIM))
        np.testing.assert_array_equal(
            features[:, PITCH_CHANGE_SLICE].argmax(axis=1),
            np.asarray([0, 4, 5]),
        )
        np.testing.assert_array_equal(features[:, PITCH_SIGN_INDEX], [1.0, 1.0, 0.0])
        np.testing.assert_array_equal(
            features[:, DURATION_SLICE].argmax(axis=1),
            np.asarray([0, 11, 22]),
        )
        np.testing.assert_array_equal(
            features[:, ONSET_SHIFT_SLICE].argmax(axis=1),
            np.asarray([0, 21, 22]),
        )

    def test_segment_note_arrays_keep_overlapping_notes_and_raw_durations(self) -> None:
        notes = [
            {
                "time": np.asarray([0.5, 1.5]),
                "freq": np.asarray([440.0, 440.0]),
            },
            {
                "time": np.asarray([2.0, 2.5]),
                "freq": np.asarray([880.0, 880.0]),
            },
        ]

        pitches, onsets, durations = extract_segment_note_arrays(notes, 1.0, 1.25)

        np.testing.assert_array_equal(pitches, [69, 81])
        np.testing.assert_allclose(onsets, [-0.5, 1.0])
        np.testing.assert_allclose(durations, [1.0, 0.5])
        self.assertAlmostEqual(note_coverage_ratio(notes, 1.0, 1.25), 0.6)


if __name__ == "__main__":
    unittest.main()
