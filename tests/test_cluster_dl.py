from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cluster_dl


class BotVerificationTest(unittest.TestCase):
    def test_recognizes_bot_prompt_variants_even_with_429(self) -> None:
        for prompt in (
            "Sign in to confirm you're not a bot.",
            "Sign in to confirm you’re not a bot.",
            "SIGN IN TO CONFIRM YOU ARE NOT A BOT.",
            "Sign in to confirm\nyou're not a bot.",
            "Sign in to confirm \x1b[31myou’re not a bot\x1b[0m.",
        ):
            with self.subTest(prompt=prompt):
                error = RuntimeError(f"HTTP Error 429: Too Many Requests\nERROR: [youtube] id: {prompt}")
                self.assertTrue(cluster_dl.is_bot_verification_error(error))

    def test_other_errors_are_included_regardless_of_status_code(self) -> None:
        for reason in (
            "This video is unavailable",
            "Video unavailable. This video is private",
            "This video is no longer available because the YouTube account associated with this video has been terminated",
            "This video has been removed by the uploader",
            "HTTP Error 429: Too Many Requests",
            "Connection timed out",
        ):
            with self.subTest(reason=reason):
                self.assertFalse(cluster_dl.is_bot_verification_error(RuntimeError(reason)))


class FailedIdsExportTest(unittest.TestCase):
    def test_run_exports_only_attempted_missing_non_bot_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = root / "annotations"
            annotations.mkdir()
            (annotations / "placeholder.gz").touch()
            audio_dir = root / "audio"
            audio_dir.mkdir()
            (audio_dir / "existing.mp3").touch()
            ids = ["existing", "bot", "removed", "converted", "success", "terminated", "unattempted"]
            info = [["DALI_ID", "NAME", "YOUTUBE", "WORKING"]] + [
                [dali_id, dali_id, f"youtube-{dali_id}", True] for dali_id in ids
            ]
            failures = {
                "bot": "HTTP Error 429: Sign in to confirm you’re not a bot.",
                "removed": "HTTP Error 429: This video is unavailable",
                "converted": "Postprocessing failed",
                "terminated": "This video is no longer available because the YouTube account associated with this video has been terminated",
            }

            def download(row, output_dir, *_args):
                dali_id = row[0]
                if dali_id in {"success", "converted"}:
                    (output_dir / f"{dali_id}.mp3").touch()
                if dali_id in failures:
                    return RuntimeError(failures[dali_id])
                return None

            argv = [
                "cluster_dl.py", "--dali-data-dir", str(annotations),
                "--audio-dir", str(audio_dir), "--audio-format", "none",
                "--target-tracks", "3", "--batch-size", "2",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(cluster_dl, "load_or_build_info", return_value=info),
                patch.object(cluster_dl, "download_one_with_ytdlp", side_effect=download) as downloader,
            ):
                cluster_dl.main()

            self.assertEqual(
                [call.args[0][0] for call in downloader.call_args_list],
                ["bot", "removed", "converted", "success", "terminated"],
            )
            self.assertEqual(
                (audio_dir / "non_bot_failed_ids.txt").read_text(encoding="utf-8"),
                "removed\nterminated\n",
            )
            self.assertIn("Sign in to confirm", (audio_dir / "download_errors.txt").read_text(encoding="utf-8"))

            # A later run with every candidate already downloaded clears stale ids.
            for dali_id in ids:
                (audio_dir / f"{dali_id}.mp3").touch()
            with (
                patch.object(sys, "argv", argv),
                patch.object(cluster_dl, "load_or_build_info", return_value=info),
                patch.object(cluster_dl, "download_one_with_ytdlp") as downloader,
            ):
                cluster_dl.main()
            downloader.assert_not_called()
            self.assertEqual((audio_dir / "non_bot_failed_ids.txt").read_text(), "")

    def test_export_is_written_before_incomplete_run_exits_and_dry_run_preserves_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "placeholder.gz").touch()
            info = [["DALI_ID", "NAME", "YOUTUBE", "WORKING"], ["private", "song", "video", True]]
            argv = [
                "cluster_dl.py", "--dali-data-dir", str(root),
                "--audio-dir", str(root), "--audio-format", "none",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(cluster_dl, "load_or_build_info", return_value=info),
                patch.object(cluster_dl, "download_one_with_ytdlp", return_value=RuntimeError("This video is private")),
            ):
                with self.assertRaises(SystemExit):
                    cluster_dl.main()
            output_path = root / "non_bot_failed_ids.txt"
            self.assertEqual(output_path.read_text(), "private\n")
            with (
                patch.object(sys, "argv", argv + ["--dry-run"]),
                patch.object(cluster_dl, "load_or_build_info", return_value=info),
                patch.object(cluster_dl, "download_one_with_ytdlp") as downloader,
            ):
                cluster_dl.main()
            downloader.assert_not_called()
            self.assertEqual(output_path.read_text(), "private\n")


if __name__ == "__main__":
    unittest.main()
