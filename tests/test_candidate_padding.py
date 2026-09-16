from __future__ import annotations

import sys
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import HubertConfig, HubertModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from prosodia.datasets import grouped_contrastive_collate
from prosodia.melody_encoder import MELODY_FEATURE_DIM
from prosodia.training import (
    MelodyAudioContrastiveModel,
    build_contrastive_model_from_checkpoint_args,
    grouped_info_nce_loss,
)
from train_contrastive import save_checkpoint


class CandidatePaddingTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(13)
        # Keep HuBERT's real convolution lengths and masking implementation, but
        # use small randomly initialized layers so no checkpoint download is needed.
        hubert = HubertModel(HubertConfig(
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=32,
            conv_dim=(4,) * 7,
            num_conv_pos_embeddings=16,
            num_conv_pos_embedding_groups=2,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            feat_proj_dropout=0.0,
            layerdrop=0.0,
            apply_spec_augment=False,
        ))
        extractor = SimpleNamespace(
            do_normalize=True, sampling_rate=16000, return_attention_mask=False,
        )
        self.model_args = dict(
            hubert_model_name="test-hubert",
            projection_dim=8,
            freeze_hubert=True,
            melody_d_model=8,
            melody_num_layers=1,
            melody_num_heads=2,
            melody_dim_feedforward=16,
            dropout=0.0,
        )
        with (
            patch("transformers.AutoModel.from_pretrained", return_value=hubert),
            patch("transformers.AutoFeatureExtractor.from_pretrained", return_value=extractor),
        ):
            self.model = MelodyAudioContrastiveModel(**self.model_args)
        self.batch = grouped_contrastive_collate([
            self.make_item(7, 1600), self.make_item(8, 1920),
        ])
        self.inputs = {
            key: self.batch[key]
            for key in (
                "melody_features", "melody_attention_mask",
                "candidate_input_values", "candidate_audio_attention_mask",
                "candidate_note_onsets", "candidate_note_durations",
                "candidate_note_attention_mask",
            )
        }

    @staticmethod
    def make_item(candidates: int, samples: int) -> dict:
        return {
            "melody_features": torch.randn(3, MELODY_FEATURE_DIM),
            "melody_midi_pitches": torch.tensor([60, 62, 64]),
            "melody_note_onsets": torch.tensor([0.0, 0.03, 0.06]),
            "melody_note_durations": torch.full((3,), 0.03),
            "melody_attention_mask": torch.ones(3, dtype=torch.bool),
            "candidate_input_values": torch.randn(candidates, samples),
            "candidate_audio_attention_mask": torch.ones(candidates, samples, dtype=torch.bool),
            "candidate_note_onsets": torch.linspace(0.0, 0.02, candidates)[:, None],
            "candidate_note_durations": torch.full((candidates, 1), 0.06),
            "candidate_note_attention_mask": torch.ones(candidates, 1, dtype=torch.bool),
            "candidate_window_seconds": torch.full((candidates,), samples / 16000),
            "candidate_mask": torch.ones(candidates, dtype=torch.bool),
            "target": torch.tensor(candidates - 1),
            "melody_sample_id": str(candidates),
            "candidate_ids": [str(i) for i in range(candidates)],
            "candidate_types": ["test"] * candidates,
            "anchor_metadata": {},
            "metadata": [{} for _ in range(candidates)],
        }

    def test_mixed_candidate_counts_skip_padding_and_preserve_order(self) -> None:
        self.model.eval()
        encoded_batch_sizes = []
        hook = self.model.audio_encoder.hubert.register_forward_pre_hook(
            lambda module, args, kwargs: encoded_batch_sizes.append(
                kwargs["input_values"].shape[0]
            ),
            with_kwargs=True,
        )
        try:
            with torch.no_grad():
                _, embeddings = self.model(**self.inputs)
        finally:
            hook.remove()

        self.assertEqual(encoded_batch_sizes, [15])
        self.assertEqual(embeddings.shape, (2, 8, 8))
        torch.testing.assert_close(embeddings[0, 7], torch.zeros(8))
        # Compare against separately encoded groups at the same waveform padding
        # length: removing candidate padding must not alter valid embeddings.
        with torch.no_grad():
            for row, count in enumerate((7, 8)):
                expected = self.model.encode_audio(
                    input_values=self.batch["candidate_input_values"][row, :count],
                    attention_mask=self.batch["candidate_audio_attention_mask"][row, :count],
                    note_onsets=self.batch["candidate_note_onsets"][row, :count],
                    note_durations=self.batch["candidate_note_durations"][row, :count],
                    note_attention_mask=self.batch["candidate_note_attention_mask"][row, :count],
                )
                torch.testing.assert_close(embeddings[row, :count], expected)

    def test_full_candidate_group_matches_unmasked_encoding(self) -> None:
        self.model.eval()
        inputs = {key: value[1:] for key, value in self.inputs.items()}
        with torch.no_grad():
            _, masked = self.model(**inputs)
            del inputs["candidate_audio_attention_mask"]
            _, unmasked = self.model(**inputs)
        torch.testing.assert_close(masked, unmasked)

    def test_mixed_candidate_counts_preserve_training_gradients(self) -> None:
        self.model.audio_encoder.set_hubert_trainable_layers(1)
        self.model.train()
        melody, audio = self.model(**self.inputs)
        loss, _ = grouped_info_nce_loss(
            melody, audio,
            candidate_mask=self.batch["candidate_mask"],
            targets=self.batch["target"],
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        for module in (
            self.model.melody_encoder,
            self.model.audio_encoder.projection,
            self.model.audio_encoder.hubert.encoder.layers[-1],
        ):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients))
            self.assertGreater(sum(float(grad.abs().sum()) for grad in gradients), 0.0)

    def test_mean_pooling_with_candidate_padding_and_trainable_hubert(self) -> None:
        self.model.audio_encoder.pooling = "mean"
        with patch.object(
            self.model.audio_encoder, "_pool_frames_to_notes", side_effect=AssertionError,
        ):
            self.test_mixed_candidate_counts_skip_padding_and_preserve_order()
            self.test_mixed_candidate_counts_preserve_training_gradients()

    def test_checkpoint_round_trip_restores_both_pooling_modes(self) -> None:
        optimizer = torch.optim.AdamW(self.model.parameters())
        scheduler = SimpleNamespace(update_step=0, state_dict=lambda: {})
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        for mode in ("note", "mean"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                self.model.audio_encoder.pooling = mode
                self.model.eval()
                save_checkpoint(
                    Path(directory), "test.pt", self.model, optimizer, scheduler,
                    scaler, 1, SimpleNamespace(**self.model_args), {},
                )
                checkpoint = torch.load(Path(directory) / "test.pt", weights_only=False)
                self.assertEqual(checkpoint["args"]["audio_pooling"], mode)
                with (
                    patch(
                        "transformers.AutoModel.from_pretrained",
                        return_value=copy.deepcopy(self.model.audio_encoder.hubert),
                    ),
                    patch(
                        "transformers.AutoFeatureExtractor.from_pretrained",
                        return_value=SimpleNamespace(
                            do_normalize=True, sampling_rate=16000, return_attention_mask=False,
                        ),
                    ),
                ):
                    restored = build_contrastive_model_from_checkpoint_args(checkpoint["args"])
                restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
                restored.eval()
                self.assertEqual(restored.audio_encoder.pooling, mode)
                with torch.no_grad():
                    expected = self.model(**self.inputs)
                    actual = restored(**self.inputs)
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b)


if __name__ == "__main__":
    unittest.main()
