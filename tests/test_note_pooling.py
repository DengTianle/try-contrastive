from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prosodia.prosody_encoder import HubertEncoder, HubertEncoderOutput
from prosodia.training import (
    build_contrastive_model_from_checkpoint_args,
    resolve_checkpoint_audio_pooling,
)


class FakeHubert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(conv_kernel=[1], conv_stride=[1])

    def _get_feature_vector_attention_mask(
        self,
        feature_length: int,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return attention_mask[:, :feature_length]

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        return SimpleNamespace(last_hidden_state=input_values.unsqueeze(-1))


def make_encoder() -> HubertEncoder:
    encoder = HubertEncoder.__new__(HubertEncoder)
    nn.Module.__init__(encoder)
    encoder.hubert = FakeHubert()
    encoder.pooling = "note"
    encoder.sampling_rate = 1
    encoder.normalize_input_waveforms = False
    encoder.hubert_uses_attention_mask = True
    encoder.hubert_trainable = True
    encoder.dropout = nn.Identity()
    encoder.projection = nn.Identity()
    return encoder


class NotePoolingTest(unittest.TestCase):
    def test_mean_pools_valid_frames_before_nonlinear_projection(self) -> None:
        encoder = make_encoder()
        encoder.pooling = "mean"
        encoder.projection = nn.Sequential(nn.Linear(1, 1), nn.ReLU())
        with torch.no_grad():
            encoder.projection[0].weight.fill_(1.0)
            encoder.projection[0].bias.zero_()
        # ReLU(mean(-4, 2)) = 0, while mean(ReLU(-4), ReLU(2)) = 1.
        # An unused note selects only the positive frame; padding is also positive.
        with patch.object(encoder, "_pool_frames_to_notes", side_effect=AssertionError):
            output = encoder(
                input_values=torch.tensor([[-4.0, 2.0, 100.0]]),
                attention_mask=torch.tensor([[True, True, False]]),
                note_onsets=torch.tensor([[1.0]]),
                note_durations=torch.tensor([[1.0]]),
                note_attention_mask=torch.tensor([[True]]),
                normalize=False,
            )
        torch.testing.assert_close(output, torch.tensor([[0.0]]))

    def test_mean_pooling_needs_no_notes_and_can_normalize(self) -> None:
        encoder = make_encoder()
        encoder.pooling = "mean"
        audio = torch.tensor([[2.0, 4.0]], requires_grad=True)
        raw = encoder(audio, normalize=False)
        torch.testing.assert_close(raw, torch.tensor([[3.0]]))
        raw.sum().backward()
        torch.testing.assert_close(audio.grad, torch.tensor([[0.5, 0.5]]))
        torch.testing.assert_close(encoder(audio), torch.tensor([[1.0]]))

    def test_mean_pooling_rejects_note_embedding_output(self) -> None:
        encoder = make_encoder()
        encoder.pooling = "mean"
        with self.assertRaisesRegex(ValueError, "return_note_embeddings requires"):
            encoder(torch.ones(1, 4), return_note_embeddings=True)

    def test_contrastive_embedding_equally_mean_pools_note_embeddings(self) -> None:
        encoder = make_encoder()
        output = encoder(
            input_values=torch.tensor([[0.0, 2.0, 4.0, 100.0]]),
            attention_mask=torch.ones(1, 4, dtype=torch.bool),
            note_onsets=torch.tensor([[0.0, 3.0]]),
            note_durations=torch.tensor([[3.0, 1.0]]),
            note_attention_mask=torch.ones(1, 2, dtype=torch.bool),
            normalize=False,
            return_note_embeddings=True,
        )

        self.assertIsInstance(output, HubertEncoderOutput)
        torch.testing.assert_close(
            output.note_embeddings,
            torch.tensor([[[2.0], [100.0]]]),
        )
        torch.testing.assert_close(output.embeddings, torch.tensor([[51.0]]))
        self.assertNotEqual(float(output.embeddings.item()), 26.5)
        self.assertTrue(output.note_attention_mask.all())

    def test_note_between_frame_centers_uses_nearest_valid_frame(self) -> None:
        encoder = make_encoder()
        note_hidden, note_mask = encoder._pool_frames_to_notes(
            hidden_states=torch.tensor([[[10.0], [20.0]]]),
            attention_mask=torch.ones(1, 2, dtype=torch.bool),
            note_onsets=torch.tensor([[0.4]]),
            note_durations=torch.tensor([[0.1]]),
            note_attention_mask=torch.ones(1, 1, dtype=torch.bool),
        )

        torch.testing.assert_close(note_hidden, torch.tensor([[[10.0]]]))
        self.assertTrue(note_mask.all())

    def test_padded_waveform_and_note_rows_remain_masked(self) -> None:
        encoder = make_encoder()
        output = encoder(
            input_values=torch.tensor([[1.0, 3.0], [0.0, 0.0]]),
            attention_mask=torch.tensor([[True, True], [False, False]]),
            note_onsets=torch.tensor([[0.0], [0.0]]),
            note_durations=torch.tensor([[2.0], [0.0]]),
            note_attention_mask=torch.tensor([[True], [False]]),
            normalize=False,
            return_note_embeddings=True,
        )

        torch.testing.assert_close(output.embeddings, torch.tensor([[2.0], [0.0]]))
        self.assertEqual(output.note_attention_mask.tolist(), [[True], [False]])


class AudioPoolingCheckpointTest(unittest.TestCase):
    def test_saved_mode_is_restored_and_conflicting_modes_are_rejected(self) -> None:
        for mode, other in (("note", "mean"), ("mean", "note")):
            with self.subTest(mode=mode):
                self.assertEqual(resolve_checkpoint_audio_pooling({"audio_pooling": mode}), mode)
                self.assertEqual(
                    resolve_checkpoint_audio_pooling({"audio_pooling": mode}, mode), mode,
                )
                with self.assertRaisesRegex(ValueError, "conflicts with checkpoint"):
                    build_contrastive_model_from_checkpoint_args(
                        {"audio_pooling": mode}, audio_pooling=other,
                    )

    def test_legacy_checkpoints_require_explicit_pooling(self) -> None:
        with self.assertRaisesRegex(ValueError, "no audio_pooling metadata"):
            build_contrastive_model_from_checkpoint_args({})
        for mode in ("note", "mean"):
            self.assertEqual(resolve_checkpoint_audio_pooling({}, mode), mode)

    def test_invalid_saved_pooling_is_not_silently_overridden(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported audio pooling"):
            build_contrastive_model_from_checkpoint_args(
                {"audio_pooling": "unknown"}, audio_pooling="note",
            )


if __name__ == "__main__":
    unittest.main()
