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
    build_line_windows,
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


class MultiLineWindowTest(unittest.TestCase):
    @staticmethod
    def _lines(count: int) -> list[dict[str, object]]:
        lines: list[dict[str, object]] = []
        for index in range(count):
            lines.append(
                {
                    "line_index": index,
                    "text": "repeat" if index in {0, 2} else f"line {index}",
                    "normalized_text": (
                        "repeat" if index in {0, 2} else f"line {index}"
                    ),
                    "start_seconds": float(index),
                    "end_seconds": float(index + 1),
                    "repeat_grouping": {
                        "lyric_class": [1, 2, 3, 4, 5][index],
                        "melody_class": [10, 11, 10, 11, 12][index],
                        "parent_index": index // 2,
                        "quality_flags": [],
                    },
                }
            )
        return lines

    @staticmethod
    def _notes(count: int, missing_line: int | None = None) -> list[dict[str, np.ndarray]]:
        notes: list[dict[str, np.ndarray]] = []
        for line_index in range(count):
            note_count = 2 if line_index == missing_line else 3
            for note_index in range(note_count):
                start = line_index + 0.1 + 0.25 * note_index
                notes.append(
                    {
                        "time": np.asarray([start, start + 0.1], dtype=np.float32),
                        "freq": np.asarray([440.0, 440.0], dtype=np.float32),
                    }
                )
        return notes

    def test_builds_non_overlapping_two_line_windows_and_composite_classes(self) -> None:
        windows, skipped = build_line_windows(
            self._lines(4),
            self._notes(4),
            audio_duration=4.0,
            lines_per_segment=2,
            segment_stride_lines=2,
            max_line_seconds=5.0,
            min_line_notes=3,
            trim_repeated_text=True,
        )

        self.assertEqual(skipped, {})
        self.assertEqual([window["line_indices"] for window in windows], [[0, 1], [2, 3]])
        self.assertEqual([window["line_count"] for window in windows], [2, 2])
        self.assertEqual(windows[0]["line_text"], "repeat\nline 1")
        self.assertEqual(windows[0]["melody_class"], windows[1]["melody_class"])
        self.assertNotEqual(windows[0]["lyric_class"], windows[1]["lyric_class"])

    def test_ineligible_line_splits_runs_instead_of_being_bridged(self) -> None:
        windows, skipped = build_line_windows(
            self._lines(5),
            self._notes(5, missing_line=2),
            audio_duration=5.0,
            lines_per_segment=2,
            segment_stride_lines=2,
            max_line_seconds=5.0,
            min_line_notes=3,
        )

        self.assertEqual([window["line_indices"] for window in windows], [[0, 1], [3, 4]])
        self.assertEqual(skipped["too_few_notes"], 1)

    def test_missing_line_index_splits_runs_instead_of_being_bridged(self) -> None:
        lines = [line for line in self._lines(4) if line["line_index"] != 2]
        windows, _ = build_line_windows(
            lines,
            self._notes(4),
            audio_duration=4.0,
            lines_per_segment=2,
            segment_stride_lines=1,
            max_line_seconds=5.0,
            min_line_notes=3,
        )

        self.assertEqual([window["line_indices"] for window in windows], [[0, 1]])

    def test_ungrouped_repeated_full_lyrics_receive_the_same_class(self) -> None:
        lines = self._lines(4)
        lines[3]["text"] = lines[1]["text"]
        lines[3]["normalized_text"] = lines[1]["normalized_text"]
        for line in lines:
            line.pop("repeat_grouping")

        windows, _ = build_line_windows(
            lines,
            self._notes(4),
            audio_duration=4.0,
            lines_per_segment=2,
            segment_stride_lines=2,
            max_line_seconds=5.0,
            min_line_notes=3,
        )

        self.assertEqual(windows[0]["lyric_class"], windows[1]["lyric_class"])
        self.assertIsNone(windows[0]["melody_class"])

    def test_one_line_mode_keeps_legacy_repeat_trimming_and_class_ids(self) -> None:
        windows, skipped = build_line_windows(
            self._lines(3),
            self._notes(3),
            audio_duration=3.0,
            lines_per_segment=1,
            segment_stride_lines=1,
            max_line_seconds=5.0,
            min_line_notes=3,
            trim_repeated_text=True,
        )

        self.assertEqual([window["line_indices"] for window in windows], [[0], [1]])
        self.assertEqual([window["melody_class"] for window in windows], [10, 11])
        self.assertEqual(skipped["repeated_line"], 1)

    def test_optional_window_duration_cap_preserves_complete_lines(self) -> None:
        windows, skipped = build_line_windows(
            self._lines(4),
            self._notes(4),
            audio_duration=4.0,
            lines_per_segment=2,
            segment_stride_lines=2,
            max_line_seconds=5.0,
            min_line_notes=3,
            max_window_seconds=1.5,
        )

        self.assertEqual(windows, [])
        self.assertEqual(skipped["window_too_long"], 2)


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
