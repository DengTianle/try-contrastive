from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 4096) -> None:
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.encoding.shape[1]:
            raise ValueError(
                f"Sequence length {x.shape[1]} exceeds max positional length {self.encoding.shape[1]}"
            )
        return x + self.encoding[:, : x.shape[1]].to(dtype=x.dtype)


@dataclass
class MelodyEncoderOutput:
    frame_embeddings: torch.Tensor
    pooled_embedding: torch.Tensor
    projected_embedding: torch.Tensor | None


class MelodyTransformerEncoder(nn.Module):
    """Transformer encoder for frame-based vocal melody tokens.

    Input shape is [batch, frames, input_dim]. With the prepared DALI dataset,
    input_dim is normally 2: log-frequency and voiced flag.
    """

    def __init__(
        self,
        input_dim: int = 2,
        projection_dim: int | None = 256,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_length: int = 4096,
        pooling: str = "cls",
    ) -> None:
        super().__init__()
        if pooling not in {"cls", "mean"}:
            raise ValueError(f"Unsupported pooling: {pooling}")
        if d_model % num_heads != 0:
            raise ValueError("-- d_model must be divisible by num_heads")

        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.d_model = d_model
        self.pooling = pooling

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_length=max_length + 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.projection = None
        if projection_dim is not None:
            self.projection = nn.Sequential(
                #nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, projection_dim),
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def encode(
        self,
        melody_features: torch.Tensor,
        melody_attention_mask: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
        normalize: bool = True,
        project: bool = True,
    ) -> MelodyEncoderOutput:
        if melody_features.ndim != 3:
            raise ValueError(
                f"Expected melody_features with shape [batch, frames, features], got {melody_features.shape}"
            )

        x = self.input_projection(melody_features)
        if frame_mask is not None:
            if frame_mask.shape != melody_features.shape[:2]:
                raise ValueError(
                    "Expected frame_mask with shape [batch, frames], "
                    f"got {frame_mask.shape}"
                )
            x = x.masked_fill(frame_mask.to(device=x.device, dtype=torch.bool).unsqueeze(-1), 0.0)

        batch_size = x.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.positional_encoding(x)

        key_padding_mask = None
        pooled_mask = None
        if melody_attention_mask is not None:
            if melody_attention_mask.ndim != 2:
                raise ValueError(
                    "Expected melody_attention_mask with shape [batch, frames]"
                )
            cls_mask = torch.ones(
                batch_size,
                1,
                dtype=torch.bool,
                device=melody_attention_mask.device,
            )
            pooled_mask = melody_attention_mask.to(dtype=torch.bool)
            full_mask = torch.cat([cls_mask, pooled_mask], dim=1)
            key_padding_mask = ~full_mask

        encoded = self.transformer(x, src_key_padding_mask=key_padding_mask)
        encoded = self.output_norm(encoded)
        frame_embeddings = encoded[:, 1:]

        if self.pooling == "cls":
            pooled = encoded[:, 0]
        elif pooled_mask is None:
            pooled = frame_embeddings.mean(dim=1)
        else:
            mask = pooled_mask.to(frame_embeddings.device).unsqueeze(-1)
            summed = (frame_embeddings * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp_min(1)
            pooled = summed / lengths

        if project and self.projection is None:
            raise RuntimeError("This melody encoder has no projection head")
        embeddings = (
            self.projection(pooled)
            if project and self.projection is not None
            else None
        )
        if normalize and embeddings is not None:
            embeddings = F.normalize(embeddings, dim=-1)
        return MelodyEncoderOutput(
            frame_embeddings=frame_embeddings,
            pooled_embedding=pooled,
            projected_embedding=embeddings,
        )

    def forward(
        self,
        melody_features: torch.Tensor,
        melody_attention_mask: torch.Tensor | None = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        projected_embedding = self.encode(
            melody_features=melody_features,
            melody_attention_mask=melody_attention_mask,
            normalize=normalize,
        ).projected_embedding
        if projected_embedding is None:
            raise RuntimeError("Melody projection was unexpectedly disabled")
        return projected_embedding


class MelodyMaskedProsodyModel(nn.Module):
    """Frame-level masked prosody pretraining head for the melody encoder."""

    def __init__(
        self,
        input_dim: int = 2,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_length: int = 4096,
        pooling: str = "cls",
    ) -> None:
        super().__init__()
        self.encoder = MelodyTransformerEncoder(
            input_dim=input_dim,
            projection_dim=None,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_length=max_length,
            pooling=pooling,
        )
        self.delta_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.voiced_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        melody_features: torch.Tensor,
        melody_attention_mask: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encoder.encode(
            melody_features=melody_features,
            melody_attention_mask=melody_attention_mask,
            frame_mask=frame_mask,
            normalize=False,
            project=False,
        )
        frame_embeddings = encoded.frame_embeddings
        return {
            "delta_log_f0": self.delta_head(frame_embeddings).squeeze(-1),
            "voiced_logits": self.voiced_head(frame_embeddings).squeeze(-1),
        }
