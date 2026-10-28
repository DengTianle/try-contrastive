from __future__ import annotations

import csv
import io
import math
import random
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from prosodia.datasets import GroupedContrastiveDataset, StressNegativeConfig, grouped_contrastive_collate
from prosodia.datasets.stress_augmentation import reverse_word_prominence
from prosodia.datasets.stress_metadata import build_stress_metadata, load_stress_metadata, write_stress_metadata
from prosodia.training import grouped_info_nce_loss
from prepare_stress_metadata import main as backfill_main
from train_contrastive import parse_args, train_one_epoch


def sung_fixture():
    rate = 16_000
    onsets = np.asarray([word + offset for word in range(5) for offset in (0.1, 0.55)], dtype=np.float32)
    durations = np.full(10, 0.3, dtype=np.float32)
    word_ids = np.repeat(np.arange(5), 2)
    levels = np.asarray([-3, -1.5, 0, 1.5, 3])
    waveform = np.zeros(5 * rate, dtype=np.float32)
    for note, start in enumerate(onsets):
        count = round(float(durations[note]) * rate)
        begin = round(float(start) * rate)
        amplitude = 0.1 * 10 ** ((levels[word_ids[note]] + (0.5 if note % 2 else -0.5)) / 20)
        waveform[begin:begin + count] = amplitude * np.sin(2 * np.pi * 220 * np.arange(count) / rate)
    metadata = {
        "format_version": np.asarray(1), "note_word_ids": word_ids,
        "word_onset_seconds": np.arange(5, dtype=np.float32),
        "word_duration_seconds": np.ones(5, dtype=np.float32),
        "word_line_ids": np.zeros(5, dtype=np.int64),
    }
    return torch.from_numpy(waveform), torch.from_numpy(onsets), torch.from_numpy(durations), metadata


class StressEnvelopeTest(unittest.TestCase):
    def test_reverses_prominence_preserves_word_ratios_and_rms(self):
        audio, onsets, durations, metadata = sung_fixture()
        original = audio.clone()
        edited, diagnostics = reverse_word_prominence(audio, onsets, durations, metadata, 16000, StressNegativeConfig(), random.Random(7))
        self.assertIsNotNone(edited)
        torch.testing.assert_close(audio, original, rtol=0, atol=0)
        self.assertLessEqual(diagnostics["rank_correlation"], -0.25)
        self.assertGreaterEqual(diagnostics["changed_note_fraction"], 0.6)
        self.assertLessEqual(diagnostics["changed_note_fraction"], 0.8)
        self.assertAlmostEqual(float(audio.square().mean()), float(edited.square().mean()), places=8)
        self.assertLess(float(edited.abs().max()), 0.99)
        gains = []
        for word in range(5):
            ratios = []
            for note in (word * 2, word * 2 + 1):
                begin = round(float(onsets[note]) * 16000)
                end = begin + round(float(durations[note]) * 16000)
                ratios.append(float(edited[begin:end].norm() / audio[begin:end].norm()))
            self.assertAlmostEqual(*ratios, places=5)
            gains.append(ratios[0])
        self.assertGreater(gains[0], gains[-1])
        again, _ = reverse_word_prominence(audio, onsets, durations, metadata, 16000, StressNegativeConfig(), random.Random(7))
        torch.testing.assert_close(edited, again, rtol=0, atol=0)

    def test_smooth_gain_does_not_introduce_boundary_steps(self):
        audio, onsets, durations, metadata = sung_fixture()
        # A DC floor makes the word-boundary gain itself observable.
        audio += 0.005
        edited, _ = reverse_word_prominence(audio, onsets, durations, metadata, 16000, StressNegativeConfig(), random.Random(7))
        self.assertIsNotNone(edited)
        for boundary in range(1, 5):
            index = boundary * 16000
            self.assertLess(float(abs(edited[index] - edited[index - 1])), 1e-6)
        first = edited[:300] / audio[:300]
        self.assertLess(float(torch.diff(first).abs().max()), 0.02)

    def test_flat_and_silent_clips_are_rejected(self):
        audio, onsets, durations, metadata = sung_fixture()
        for word, db in enumerate((-3, -1.5, 0, 1.5, 3)):
            audio[word * 16000:(word + 1) * 16000] *= 10 ** (-db / 20)
        edited, reason = reverse_word_prominence(audio, onsets, durations, metadata, 16000, StressNegativeConfig(), random.Random(7))
        self.assertIsNone(edited)
        self.assertEqual(reason["reason"], "insufficient_word_contrast")
        edited, _ = reverse_word_prominence(torch.zeros_like(audio), onsets, durations, metadata, 16000, StressNegativeConfig(), random.Random(7))
        self.assertIsNone(edited)

    def test_three_word_fallback_handles_discrete_four_note_coverage(self):
        audio, _, _, _ = sung_fixture()
        # Four complete, one-note words; ordinary pair swaps cannot meet 60-80%.
        audio = audio[:64000]
        onsets = torch.tensor([0.1, 1.1, 2.1, 3.1])
        durations = torch.full((4,), 0.3)
        metadata = {
            "note_word_ids": np.arange(4), "word_onset_seconds": np.arange(4),
            "word_duration_seconds": np.ones(4), "word_line_ids": np.zeros(4),
        }
        edited, diagnostics = reverse_word_prominence(audio, onsets, durations, metadata, 16000, StressNegativeConfig(), random.Random(7))
        self.assertIsNotNone(edited)
        self.assertEqual(diagnostics["changed_note_fraction"], 0.75)
        self.assertLessEqual(diagnostics["rank_correlation"], -0.25)

    def test_incomplete_word_links_are_rejected(self):
        audio, onsets, durations, metadata = sung_fixture()
        metadata["note_word_ids"][:3] = -1
        edited, diagnostics = reverse_word_prominence(audio, onsets, durations, metadata, 16000, StressNegativeConfig(), random.Random(7))
        self.assertIsNone(edited)
        self.assertEqual(diagnostics["reason"], "incomplete_word_links")

    def test_headroom_rejection_does_not_clip_or_mutate_source(self):
        audio, onsets, durations, metadata = sung_fixture()
        audio[800] = 0.9  # Outside the measured note interiors, inside the quiet word.
        edited, reason = reverse_word_prominence(audio, onsets, durations, metadata, 16000, StressNegativeConfig(), random.Random(7))
        self.assertIsNone(edited)
        self.assertEqual(reason["reason"], "insufficient_headroom")
        self.assertAlmostEqual(float(audio[800]), 0.9, places=6)

    def test_partial_words_are_left_intact(self):
        audio, onsets, durations, metadata = sung_fixture()
        metadata["word_onset_seconds"][0] = -0.1
        metadata["word_duration_seconds"][0] = 1.1
        config = replace(StressNegativeConfig(), min_changed_fraction=0.5, max_changed_fraction=1.0)
        edited, _ = reverse_word_prominence(audio, onsets, durations, metadata, 16000, config, random.Random(7))
        self.assertIsNotNone(edited)
        positions = audio[:16000].abs() > 0.01
        ratios = edited[:16000][positions] / audio[:16000][positions]
        self.assertLess(float(ratios.max() - ratios.min()), 1e-5)


class StressScheduleAndLossTest(unittest.TestCase):
    def test_delayed_ramp_and_indefinite_hold(self):
        config = StressNegativeConfig()
        for epoch in range(4):
            self.assertFalse(config.active(epoch))
            self.assertEqual(config.loss_weight(epoch, 10, 10), 0)
        self.assertTrue(config.active(4))
        self.assertAlmostEqual(config.loss_weight(4, 1, 10), 0.0125)
        self.assertAlmostEqual(config.loss_weight(4, 10, 10), 0.125)
        self.assertEqual(config.loss_weight(5, 10, 10), 0.25)
        self.assertEqual(config.loss_weight(99, 10, 10), 0.25)
        self.assertEqual(replace(config, ramp_epochs=0).loss_weight(4, 1, 10), 0.25)
        for changes in ({"weight": float("nan")}, {"probability": -0.1}, {"start_epoch": 0}, {"max_gain_db": 1}):
            with self.assertRaises(ValueError):
                replace(config, **changes)

    def test_cli_is_opt_in_and_defaults_to_epoch_five(self):
        with patch.object(sys, "argv", ["train_contrastive.py"]):
            args = parse_args()
        self.assertFalse(args.stress_negatives)
        self.assertEqual(args.stress_negative_start_epoch, 5)
        self.assertEqual(args.stress_negative_ramp_epochs, 2)

    def test_weighted_denominator_raw_logits_and_gradients(self):
        melody = torch.tensor([[1.0, 0.5]], requires_grad=True)
        candidates = torch.tensor([[[0.5, 0.5], [1.5, 1.0], [-0.5, 0.0], [100.0, 0.0]]], requires_grad=True)
        # Shuffled positive index 1; padded slot 3 must not enter the denominator.
        target = torch.tensor([1])
        valid = torch.tensor([[True, True, True, False]])
        synthetic = torch.tensor([[True, False, False, False]])
        losses = []
        for weight in (0, 0.25, 1):
            loss, raw = grouped_info_nce_loss(melody, candidates, valid, target, temperature=1, stress_negative_mask=synthetic, stress_negative_weight=weight)
            scores = torch.einsum("bd,bcd->bc", melody, candidates)
            expected = torch.log(torch.exp(scores[0, 1]) + torch.exp(scores[0, 2]) + weight * torch.exp(scores[0, 0])) - scores[0, 1]
            torch.testing.assert_close(loss, expected)
            torch.testing.assert_close(raw[0, :3], scores[0, :3])
            losses.append(float(loss.detach()))
        self.assertLess(losses[0], losses[1])
        self.assertLess(losses[1], losses[2])
        loss.backward()
        self.assertGreater(float(melody.grad.norm()), 0)
        self.assertGreater(float(candidates.grad[0, 0].norm()), 0)
        self.assertEqual(float(candidates.grad[0, 3].norm()), 0)
        # Defensively marking a positive as synthetic must never discount it.
        loss, _ = grouped_info_nce_loss(melody, candidates, valid, target, temperature=1, stress_negative_mask=torch.ones_like(synthetic), stress_negative_weight=0)
        self.assertEqual(float(loss.detach()), 0)


class StressDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        audio, onsets, durations, metadata = sung_fixture()
        combined = np.zeros(25 * 16000, dtype=np.float32)
        self.rows = []
        for name, start, melody_class, frequency_scale in (("anchor", 0, 1, 1), ("repeat", 10, 1, -1), ("other", 20, 2, 0.7)):
            combined[start * 16000:(start + 5) * 16000] = audio.numpy() * frequency_scale
            np.savez_compressed(self.root / f"{name}.npz", midi_pitches=np.full(10, 60), onset_seconds=onsets.numpy(), note_duration_seconds=durations.numpy())
            write_stress_metadata(self.root / f"{name}_words.npz", metadata)
            self.rows.append({
                "sample_id": name, "split": "train", "dali_id": "song",
                "audio_path": "song.wav", "melody_path": f"{name}.npz",
                "stress_metadata_path": f"{name}_words.npz",
                "start_seconds": str(start), "end_seconds": str(start + 5),
                "segment_seconds": "5", "melody_class": str(melody_class), "lyric_class": "1" if name == "anchor" else "2",
            })
        sf.write(self.root / "song.wav", combined, 16000, subtype="FLOAT")
        self.manifest = self.root / "manifest.csv"
        self.write_manifest()

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self):
        with self.manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def test_repeat_positive_candidate_masks_padding_and_epoch_boundary(self):
        config = replace(StressNegativeConfig(), probability=1)
        dataset = GroupedContrastiveDataset(self.manifest, split="train", positive_variant_policy="retexted-first", candidate_window_policy="match-positive", stress_negative_config=config)
        index = next(i for i, group in enumerate(dataset.groups) if group["anchor"]["sample_id"] == "anchor")
        dataset.set_epoch(3)
        with patch("prosodia.datasets.contrastive_dataset.reverse_word_prominence", side_effect=AssertionError("warm-up must not generate")):
            warmup = dataset[index]
        self.assertFalse(warmup["candidate_stress_negative_mask"].any())
        dataset.set_epoch(4)
        item = dataset[index]
        target = int(item["target"])
        self.assertEqual(item["metadata"][target]["sample_id"], "repeat")
        self.assertEqual(int(item["candidate_stress_negative_mask"].sum()), 1)
        synthetic = int(torch.nonzero(item["candidate_stress_negative_mask"])[0, 0])
        self.assertNotEqual(synthetic, target)
        self.assertEqual(item["candidate_types"][synthetic], "hard_negative_stress")
        self.assertEqual(item["metadata"][synthetic]["sample_id"], "repeat")
        for key in ("candidate_note_onsets", "candidate_note_durations", "candidate_note_attention_mask", "candidate_audio_attention_mask", "candidate_window_seconds"):
            torch.testing.assert_close(item[key][target], item[key][synthetic])
        source, *_ = sung_fixture()
        torch.testing.assert_close(item["candidate_input_values"][target], -source)
        again = dataset[index]
        torch.testing.assert_close(item["candidate_input_values"], again["candidate_input_values"], rtol=0, atol=0)
        batch = grouped_contrastive_collate([warmup, item])
        self.assertEqual(tuple(batch["candidate_stress_negative_mask"].shape), tuple(batch["candidate_mask"].shape))
        self.assertFalse(batch["candidate_stress_negative_mask"][0].any())
        self.assertFalse(batch["candidate_mask"][0, -1])
        self.assertEqual(int(batch["candidate_stress_negative_mask"].sum()), 1)
        dataset.set_epoch(20)
        self.assertEqual(int(dataset[index]["candidate_stress_negative_mask"].sum()), 1)
        ordinary = GroupedContrastiveDataset(self.manifest, split="train")
        ordinary.set_epoch(20)
        self.assertFalse(ordinary[index]["candidate_stress_negative_mask"].any())

    def test_training_step_updates_melody_and_reports_raw_stress_metrics(self):
        dataset = GroupedContrastiveDataset(self.manifest, split="train", stress_negative_config=replace(StressNegativeConfig(), probability=1))
        dataset.set_epoch(4)
        item = dataset[0]
        batch = grouped_contrastive_collate([item])

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.melody = nn.Parameter(torch.tensor([0.4, 0.8]))

            def forward(self, candidate_input_values, **kwargs):
                values = candidate_input_values
                audio = torch.stack((values.square().mean(-1), values[..., :16000].square().mean(-1)), dim=-1) * 100
                return self.melody[None], audio

        model = TinyModel()
        before = model.melody.detach().clone()
        metrics = train_one_epoch(
            model, [batch], torch.optim.SGD(model.parameters(), lr=0.1), torch.device("cpu"),
            temperature=1, global_loss_weight=0, symmetric_global_loss=False,
            audio_noise_snr_db=None, audio_background_mix_prob=0, audio_background_mix_snr_db=35,
            grad_clip_norm=None, use_amp=False, scaler=torch.amp.GradScaler("cpu", enabled=False),
            scheduler=SimpleNamespace(step=lambda: None), progress=False, desc="test",
            stress_negative_config=StressNegativeConfig(), epoch_index=4,
        )
        self.assertFalse(torch.equal(before, model.melody.detach()))
        self.assertEqual(metrics["stress_negative_count"], 1)
        self.assertEqual(metrics["stress_negative_attempts"], 1)
        self.assertEqual(metrics["stress_negative_accept_rate"], 1)
        self.assertEqual(metrics["stress_negative_weight"], 0.125)
        self.assertTrue(math.isfinite(metrics["stress_negative_score_gap"]))

    def test_metadata_required_only_when_feature_enabled(self):
        for row in self.rows:
            del row["stress_metadata_path"]
        self.write_manifest()
        self.assertGreater(len(GroupedContrastiveDataset(self.manifest)), 0)
        with self.assertRaisesRegex(ValueError, "prepare_stress_metadata.py"):
            GroupedContrastiveDataset(self.manifest, stress_negative_config=StressNegativeConfig())

    def test_metadata_shifts_and_filters_with_candidate_window(self):
        dataset = GroupedContrastiveDataset(self.manifest, split="train", stress_negative_config=StressNegativeConfig())
        dataset[0]  # Populate the timing cache.
        row = dataset.rows[0]
        metadata = dataset._stress_metadata_for_window(row, (16000, 32000))
        np.testing.assert_array_equal(metadata["note_word_ids"], [1, 1, 2, 2])
        np.testing.assert_allclose(metadata["word_onset_seconds"], [-1, 0, 1, 2, 3])
        original = load_stress_metadata(self.root / "anchor_words.npz")
        np.testing.assert_allclose(original["word_onset_seconds"], [0, 1, 2, 3, 4])
        late_row = {**row, "start_seconds": "1000.123456"}
        for boundary in (0.1, 0.4, 1.1, 1.4, 2.1, 2.4):
            for jitter in (-1, 0, 1):
                window = (round((1000.123456 + boundary) * 16000) + jitter, 16000)
                words = dataset._stress_metadata_for_window(late_row, window)
                notes, _ = dataset._candidate_note_timing(late_row, window)
                self.assertEqual(len(words["note_word_ids"]), len(notes))


class StressMetadataTest(unittest.TestCase):
    def entry_and_notes(self):
        notes = [
            {"time": np.asarray([i + 0.1, i + 0.4]), "freq": np.asarray([440.0, 440.0]), "index": i}
            for i in range(3)
        ]
        entry = SimpleNamespace(annotations={"annot": {"words": [
            {"time": [i, i + 1], "index": 0} for i in range(3)
        ]}})
        return entry, notes

    def test_membership_and_truncated_word_boundaries(self):
        entry, notes = self.entry_and_notes()
        result = build_stress_metadata(entry, notes, 0.2, 2.3)
        np.testing.assert_array_equal(result["note_word_ids"], [0, 1, 2])
        np.testing.assert_allclose(result["word_onset_seconds"], [-0.2, 0.8, 1.8])
        notes[0]["index"] = None
        result = build_stress_metadata(entry, notes, 0, 3)
        self.assertEqual(result["note_word_ids"][0], -1)

    def test_backfill_new_manifest_rebases_paths_and_rejects_wrong_annotations(self):
        entry, notes = self.entry_and_notes()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            melody = root / "melody.npz"
            np.savez_compressed(melody, midi_pitches=np.full(3, 69), onset_seconds=np.asarray([0.1, 1.1, 2.1]), note_duration_seconds=np.full(3, 0.3))
            manifest = root / "segments.csv"
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sample_id", "dali_id", "start_seconds", "segment_seconds", "melody_path", "audio_path"])
                writer.writeheader()
                writer.writerow(dict(sample_id="example", dali_id="song", start_seconds=0, segment_seconds=3, melody_path="melody.npz", audio_path="audio.wav"))
            destination = root / "nested" / "stress.csv"
            before = melody.read_bytes()
            argv = ["prepare_stress_metadata.py", "--manifest", str(manifest), "--output-manifest", str(destination)]
            with patch.object(sys, "argv", argv), patch("prepare_stress_metadata.load_dali", return_value={"song": entry}), patch("prepare_stress_metadata.extract_dali_notes", return_value=notes), redirect_stdout(io.StringIO()):
                backfill_main()
            self.assertEqual(melody.read_bytes(), before)
            with destination.open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["melody_path"], "../melody.npz")
            np.testing.assert_array_equal(load_stress_metadata(destination.parent / row["stress_metadata_path"])["note_word_ids"], [0, 1, 2])
            notes[0]["time"] += 0.1
            with patch.object(sys, "argv", argv), patch("prepare_stress_metadata.load_dali", return_value={"song": entry}), patch("prepare_stress_metadata.extract_dali_notes", return_value=notes):
                with self.assertRaisesRegex(SystemExit, "does not match DALI"):
                    backfill_main()


if __name__ == "__main__":
    unittest.main()
