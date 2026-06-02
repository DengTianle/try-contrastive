from __future__ import annotations

import torch
from torch import nn

from prosodia.melody_encoder import MelodyTransformerEncoder
from prosodia.prosody_encoder import HubertProsodyEncoder


class MelodyAudioContrastiveModel(nn.Module):
    def __init__(
        self,
        hubert_model_name: str,
        projection_dim: int = 256,
        freeze_hubert: bool = False,
        melody_input_dim: int = 2,
        melody_d_model: int = 256,
        melody_num_layers: int = 4,
        melody_num_heads: int = 4,
        melody_dim_feedforward: int = 1024,
        dropout: float = 0.1,
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
    ) -> torch.Tensor:
        return self.audio_encoder(
            input_values=input_values,
            attention_mask=attention_mask,
        )

    def forward(
        self,
        melody_features: torch.Tensor,
        candidate_input_values: torch.Tensor,
        melody_attention_mask: torch.Tensor | None = None,
        candidate_audio_attention_mask: torch.Tensor | None = None,
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
        flat_audio_embeddings = self.encode_audio(
            input_values=flat_audio,
            attention_mask=flat_audio_mask,
        )
        candidate_audio_embeddings = flat_audio_embeddings.reshape(batch_size, num_candidates, -1)
        return melody_embeddings, candidate_audio_embeddings
