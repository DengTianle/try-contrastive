from __future__ import annotations

import torch
import torch.nn.functional as F


def grouped_info_nce_loss(
    melody_embeddings: torch.Tensor,
    candidate_audio_embeddings: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute one-way melody-to-audio InfoNCE over explicit candidates.

    melody_embeddings: [batch, dim]
    candidate_audio_embeddings: [batch, candidates, dim]
    candidate_mask: [batch, candidates], True for valid candidates

    Candidate index 0 is assumed to be the positive audio.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if melody_embeddings.ndim != 2:
        raise ValueError(f"Expected melody_embeddings [B, D], got {melody_embeddings.shape}")
    if candidate_audio_embeddings.ndim != 3:
        raise ValueError(
            f"Expected candidate_audio_embeddings [B, C, D], got {candidate_audio_embeddings.shape}"
        )

    logits = torch.einsum("bd,bcd->bc", melody_embeddings, candidate_audio_embeddings)
    logits = logits / temperature

    if candidate_mask is not None:
        logits = logits.masked_fill(~candidate_mask.to(dtype=torch.bool), torch.finfo(logits.dtype).min)

    targets = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(logits, targets)
    return loss, logits
