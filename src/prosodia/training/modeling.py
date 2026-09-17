from __future__ import annotations

import torch
from torch import nn

from prosodia.melody_encoder import MELODY_FEATURE_DIM, MelodyTransformerEncoder
from prosodia.prosody_encoder import HubertEncoderOutput, HubertProsodyEncoder


class MelodyAudioContrastiveModel(nn.Module):
    def __init__(
        self,
        hubert_model_name: str,
        projection_dim: int = 256,
        freeze_hubert: bool = False,
        melody_input_dim: int = MELODY_FEATURE_DIM,
        melody_d_model: int = 256,
        melody_num_layers: int = 4,
        melody_num_heads: int = 4,
        melody_dim_feedforward: int = 1024,
        dropout: float = 0.1,
        audio_pooling: str = "note",
    ) -> None:
        super().__init__()
        self.melody_encoder = MelodyTransformerEncoder(
            input_dim=melody_input_dim,
            projection_dim=projection_dim,
            d_model=melody_d_model,
            num_layers=melody_num_layers,
            num_heads=melody_num_heads,
            dim_feedforward=melody_dim_feedforward,
            dropout=dropout,
        )
        self.audio_encoder = HubertProsodyEncoder(
            model_name=hubert_model_name,
            projection_dim=projection_dim,
            dropout=dropout,
            freeze_hubert=freeze_hubert,
            pooling=audio_pooling,
        )

    def encode_melody(
        self,
        melody_features: torch.Tensor,
        melody_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.melody_encoder(
            melody_features=melody_features,
            melody_attention_mask=melody_attention_mask,
        )

    def encode_audio(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        note_onsets: torch.Tensor | None = None,
        note_durations: torch.Tensor | None = None,
        note_attention_mask: torch.Tensor | None = None,
        return_note_embeddings: bool = False,
        note_inputs_validated: bool = False,
    ) -> torch.Tensor | HubertEncoderOutput:
        return self.audio_encoder(
            input_values=input_values,
            attention_mask=attention_mask,
            note_onsets=note_onsets,
            note_durations=note_durations,
            note_attention_mask=note_attention_mask,
            return_note_embeddings=return_note_embeddings,
            note_inputs_validated=note_inputs_validated,
        )

    def forward(
        self,
        melody_features: torch.Tensor,
        candidate_input_values: torch.Tensor,
        melody_attention_mask: torch.Tensor | None = None,
        candidate_audio_attention_mask: torch.Tensor | None = None,
        candidate_note_onsets: torch.Tensor | None = None,
        candidate_note_durations: torch.Tensor | None = None,
        candidate_note_attention_mask: torch.Tensor | None = None,
        candidate_notes_validated: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        melody_embeddings = self.encode_melody(
            melody_features=melody_features,
            melody_attention_mask=melody_attention_mask,
        )

        batch_size, num_candidates, num_samples = candidate_input_values.shape
        flat_audio = candidate_input_values.reshape(batch_size * num_candidates, num_samples)
        flat_audio_mask = None
        if candidate_audio_attention_mask is not None:
            flat_audio_mask = candidate_audio_attention_mask.reshape(
                batch_size * num_candidates,
                num_samples,
            )
        flat_note_onsets = None
        flat_note_durations = None
        flat_note_mask = None
        note_inputs = (
            candidate_note_onsets,
            candidate_note_durations,
            candidate_note_attention_mask,
        )
        if self.audio_encoder.pooling == "mean":
            note_inputs = (None, None, None)
        if all(value is not None for value in note_inputs):
            flat_note_onsets = candidate_note_onsets.reshape(
                batch_size * num_candidates,
                -1,
            )
            flat_note_durations = candidate_note_durations.reshape(
                batch_size * num_candidates,
                -1,
            )
            flat_note_mask = candidate_note_attention_mask.reshape(
                batch_size * num_candidates,
                -1,
            )
        elif any(value is not None for value in note_inputs):
            raise ValueError(
                "Candidate note onsets, durations, and attention mask must be "
                "provided together"
            )

        # Collation pads the candidate dimension when groups have different
        # sizes. HuBERT's feature-mask helper assumes non-empty waveforms, so
        # exclude those empty slots from both encoding and note pooling.
        valid_indices = None
        if flat_audio_mask is not None:
            indices = flat_audio_mask.any(dim=-1).nonzero(as_tuple=True)[0]
            if indices.numel() != flat_audio.shape[0]:
                valid_indices = indices
                flat_audio = flat_audio.index_select(0, indices)
                flat_audio_mask = flat_audio_mask.index_select(0, indices)
                if flat_note_onsets is not None:
                    flat_note_onsets = flat_note_onsets.index_select(0, indices)
                    flat_note_durations = flat_note_durations.index_select(0, indices)
                    flat_note_mask = flat_note_mask.index_select(0, indices)

        flat_audio_embeddings = self.encode_audio(
            input_values=flat_audio,
            attention_mask=flat_audio_mask,
            note_onsets=flat_note_onsets,
            note_durations=flat_note_durations,
            note_attention_mask=flat_note_mask,
            note_inputs_validated=candidate_notes_validated,
        )
        if valid_indices is not None:
            # Restore candidate order and keep padding at zero. index_copy
            # preserves gradients from the grouped loss to the real candidates.
            flat_audio_embeddings = flat_audio_embeddings.new_zeros(
                batch_size * num_candidates, flat_audio_embeddings.shape[-1],
            ).index_copy(0, valid_indices, flat_audio_embeddings)
        candidate_audio_embeddings = flat_audio_embeddings.reshape(batch_size, num_candidates, -1)
        return melody_embeddings, candidate_audio_embeddings
