from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prosodia.prosody_encoder import HubertEncoder, HubertEncoderOutput


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
    encoder.sampling_rate = 1
    encoder.normalize_input_waveforms = False
    encoder.hubert_uses_attention_mask = True
    encoder.hubert_trainable = True
    encoder.dropout = nn.Identity()
    encoder.projection = nn.Identity()
    return encoder


class NotePoolingTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
