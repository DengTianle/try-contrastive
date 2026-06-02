from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class HubertEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        projection_dim: int = 256,
        dropout: float = 0.1,
        freeze_hubert: bool = False,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        if pooling != "mean":
            raise ValueError(f"Unsupported pooling: {pooling}")

        self.model_name = model_name
        self.projection_dim = projection_dim
        self.pooling = pooling
        self.hubert = AutoModel.from_pretrained(model_name)
        hidden_size = self.hubert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, projection_dim),
        )
        self.set_hubert_trainable(not freeze_hubert)
        
        # Memory optimization: Always freeze the CNN feature extractor, which uses huge amounts of memory 
        # and rarely needs fine-tuning for downstream tasks.
        if hasattr(self.hubert, "freeze_feature_encoder"):
            self.hubert.freeze_feature_encoder()
            
        # Memory optimization: Enable gradient checkpointing for the transformer layers to trade computation for memory.
        #if not freeze_hubert and hasattr(self.hubert, "gradient_checkpointing_enable"):
        #    self.hubert.gradient_checkpointing_enable()

    def set_hubert_trainable(self, trainable: bool) -> None:
        for parameter in self.hubert.parameters():
            parameter.requires_grad = trainable

    def _pool_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states.mean(dim=1)

        feature_attention_mask = self.hubert._get_feature_vector_attention_mask(
            hidden_states.shape[1],
            attention_mask,
        )
        mask = feature_attention_mask.to(hidden_states.device).unsqueeze(-1)
        summed = (hidden_states * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp_min(1)
        return summed / lengths

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        outputs = self.hubert(input_values=input_values, attention_mask=attention_mask)
        hidden_states = self.dropout(outputs.last_hidden_state)
        pooled = self._pool_hidden_states(hidden_states, attention_mask)
        embeddings = self.projection(pooled)
        if normalize:
            embeddings = F.normalize(embeddings, dim=-1)
        return embeddings


HubertProsodyEncoder = HubertEncoder
