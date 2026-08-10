from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from prepare_dali_dataset import read_keep_file


class KeepFileTest(unittest.TestCase):
    def test_reads_one_id_per_line_and_ignores_comments_and_blanks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            keep_file = Path(directory) / "keep.txt"
            keep_file.write_text(
                "song-a\n\n  song-b  # explanation\n# comment\nsong-a\n",
                encoding="utf-8",
            )

            self.assertEqual(read_keep_file(keep_file), ["song-a", "song-b", "song-a"])


if __name__ == "__main__":
    unittest.main()
