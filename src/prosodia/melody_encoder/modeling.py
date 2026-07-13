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
    token_embeddings: torch.Tensor
    pooled_embedding: torch.Tensor
    projected_embedding: torch.Tensor


class MelodyTransformerEncoder(nn.Module):
    """Transformer encoder for tokenized pitch/rest and duration-ratio sequences."""

    def __init__(
        self,
        vocab_size: int,
        projection_dim: int = 256,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_length: int = 4096,
        pooling: str = "cls",
        pad_token_id: int = 0,
        mask_token_id: int = 1,
    ) -> None:
        super().__init__()
        if pooling not in {"cls", "mean"}:
            raise ValueError(f"Unsupported pooling: {pooling}")
        if d_model % num_heads != 0:
            raise ValueError("-- d_model must be divisible by num_heads")
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.d_model = d_model
        self.pooling = pooling
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.input_norm = nn.LayerNorm(d_model)
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
        melody_token_ids: torch.Tensor,
        melody_attention_mask: torch.Tensor | None = None,
        normalize: bool = True,
    ) -> MelodyEncoderOutput:
        if melody_token_ids.ndim != 2:
            raise ValueError(
                f"Expected melody_token_ids with shape [batch, tokens], got {melody_token_ids.shape}"
            )

        x = self.input_norm(self.token_embedding(melody_token_ids))

        batch_size = x.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.positional_encoding(x)

        key_padding_mask = None
        pooled_mask = None
        if melody_attention_mask is not None:
            if melody_attention_mask.ndim != 2:
                raise ValueError(
                    "Expected melody_attention_mask with shape [batch, tokens]"
                )
            if melody_attention_mask.shape != melody_token_ids.shape:
                raise ValueError(
                    "Expected melody_attention_mask to match melody_token_ids, "
                    f"got {melody_attention_mask.shape} and {melody_token_ids.shape}"
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
        token_embeddings = encoded[:, 1:]

        if self.pooling == "cls":
            pooled = encoded[:, 0]
        elif pooled_mask is None:
            pooled = token_embeddings.mean(dim=1)
        else:
            mask = pooled_mask.to(token_embeddings.device).unsqueeze(-1)
            summed = (token_embeddings * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp_min(1)
            pooled = summed / lengths

        embeddings = self.projection(pooled)
        if normalize:
            embeddings = F.normalize(embeddings, dim=-1)
        return MelodyEncoderOutput(
            token_embeddings=token_embeddings,
            pooled_embedding=pooled,
            projected_embedding=embeddings,
        )

    def forward(
        self,
        melody_token_ids: torch.Tensor,
        melody_attention_mask: torch.Tensor | None = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        return self.encode(
            melody_token_ids=melody_token_ids,
            melody_attention_mask=melody_attention_mask,
            normalize=normalize,
        ).projected_embedding


class MelodyMaskedTokenModel(nn.Module):
    """Masked pretraining heads for onset/rest, pitch, and duration ratio."""

    def __init__(
        self,
        vocab_size: int,
        ratio_count: int | None = None,
        pitch_count: int = 88,
        projection_dim: int = 256,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_length: int = 4096,
        pooling: str = "cls",
        pad_token_id: int = 0,
        mask_token_id: int = 1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        if pitch_count <= 0 or pitch_count > 88:
            raise ValueError("pitch_count must be in [1, 88]")
        self.pitch_count = pitch_count
        if ratio_count is None:
            event_class_count = 1 + pitch_count
            if (vocab_size - 2) % event_class_count != 0:
                raise ValueError("Cannot infer ratio_count from vocab_size")
            ratio_count = (vocab_size - 2) // event_class_count
        if ratio_count <= 0:
            raise ValueError("ratio_count must be positive")
        expected_vocab_size = 2 + ((1 + pitch_count) * ratio_count)
        if vocab_size != expected_vocab_size:
            raise ValueError(
                "vocab_size does not match pitch_count and ratio_count: "
                f"expected {expected_vocab_size}, got {vocab_size}"
            )
        self.ratio_count = ratio_count
        self.encoder = MelodyTransformerEncoder(
            vocab_size=vocab_size,
            projection_dim=projection_dim,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_length=max_length,
            pooling=pooling,
            pad_token_id=pad_token_id,
            mask_token_id=mask_token_id,
        )
        self.onset_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )
        self.pitch_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pitch_count),
        )
        self.ratio_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, ratio_count),
        )

    def forward(
        self,
        melody_token_ids: torch.Tensor,
        melody_attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encoder.encode(
            melody_token_ids=melody_token_ids,
            melody_attention_mask=melody_attention_mask,
            normalize=False,
        )
        return {
            "onset_logits": self.onset_head(encoded.token_embeddings),
            "pitch_logits": self.pitch_head(encoded.token_embeddings),
            "ratio_logits": self.ratio_head(encoded.token_embeddings),
            "pooled_embedding": encoded.projected_embedding,
        }


MelodyMaskedProsodyModel = MelodyMaskedTokenModel
