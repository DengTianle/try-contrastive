from __future__ import annotations

import torch
import torch.nn.functional as F


def grouped_info_nce_loss(
    melody_embeddings: torch.Tensor,
    candidate_audio_embeddings: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
    targets: torch.Tensor | None = None,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute one-way melody-to-audio InfoNCE over explicit candidates.

    melody_embeddings: [batch, dim]
    candidate_audio_embeddings: [batch, candidates, dim]
    candidate_mask: [batch, candidates], True for valid candidates
    targets: [batch], positive candidate index for each melody

    Candidate index 0 is assumed to be the positive audio when targets is not supplied.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if melody_embeddings.ndim != 2:
        raise ValueError(f"Expected melody_embeddings [B, D], got {melody_embeddings.shape}")
    if candidate_audio_embeddings.ndim != 3:
        raise ValueError(
            f"Expected candidate_audio_embeddings [B, C, D], got {candidate_audio_embeddings.shape}"
        )

    batch_size, num_candidates, _ = candidate_audio_embeddings.shape

    if targets is None:
        targets = torch.zeros(batch_size, dtype=torch.long, device=melody_embeddings.device)
    else:
        targets = targets.to(device=melody_embeddings.device, dtype=torch.long)

    logits = torch.einsum("bd,bcd->bc", melody_embeddings, candidate_audio_embeddings)
    logits = logits / temperature
    if candidate_mask is not None:
        logits = logits.masked_fill(~candidate_mask.to(dtype=torch.bool), torch.finfo(logits.dtype).min)

    loss = F.cross_entropy(logits, targets)
    return loss, logits


def positive_audio_embeddings(
    candidate_audio_embeddings: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Gather the positive audio embedding from each grouped candidate set."""

    if candidate_audio_embeddings.ndim != 3:
        raise ValueError(
            f"Expected candidate_audio_embeddings [B, C, D], got {candidate_audio_embeddings.shape}"
        )
    targets = targets.to(device=candidate_audio_embeddings.device, dtype=torch.long)
    gather_index = targets[:, None, None].expand(-1, 1, candidate_audio_embeddings.shape[-1])
    return candidate_audio_embeddings.gather(dim=1, index=gather_index).squeeze(1)


def global_in_batch_info_nce_loss(
    melody_embeddings: torch.Tensor,
    positive_audio_embeddings: torch.Tensor,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute melody-to-positive-audio InfoNCE across the minibatch."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if melody_embeddings.ndim != 2:
        raise ValueError(f"Expected melody_embeddings [B, D], got {melody_embeddings.shape}")
    if positive_audio_embeddings.ndim != 2:
        raise ValueError(
            f"Expected positive_audio_embeddings [B, D], got {positive_audio_embeddings.shape}"
        )
    if melody_embeddings.shape != positive_audio_embeddings.shape:
        raise ValueError(
            "melody_embeddings and positive_audio_embeddings must have matching [B, D] shapes, "
            f"got {melody_embeddings.shape} and {positive_audio_embeddings.shape}"
        )

    logits = melody_embeddings @ positive_audio_embeddings.T
    logits = logits / temperature
    targets = torch.arange(logits.shape[0], dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(logits, targets)
    return loss, logits
