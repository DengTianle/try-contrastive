from __future__ import annotations

import torch
from torch import nn


class HubertEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        #num_labels: int = 3, 
        dropout: float = 0.1, #default for HuBERT anyway
        freeze_hubert: bool = False,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.model_name = model_name
        self.num_labels = num_labels
        self.hubert = AutoModel.from_pretrained(model_name)
        hidden_size = self.hubert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        #self.classifier = nn.Linear(hidden_size, num_labels)
        self.set_hubert_trainable(not freeze_hubert)

    def set_hubert_trainable(self, trainable: bool) -> None:
        for parameter in self.hubert.parameters():
            parameter.requires_grad = trainable

    def forward(
        self,
        input_values: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.hubert(input_values=input_values)
        hidden_states = self.dropout(outputs.last_hidden_state)
        #return self.classifier(hidden_states)
