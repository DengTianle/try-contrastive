from __future__ import annotations

from typing import Any

from .modeling import MelodyAudioContrastiveModel


def checkpoint_arg(checkpoint_args: dict[str, Any], name: str, default: Any) -> Any:
    value = checkpoint_args.get(name, default)
    return default if value is None else value


def build_contrastive_model_from_checkpoint_args(
    checkpoint_args: dict[str, Any],
) -> MelodyAudioContrastiveModel:
    return MelodyAudioContrastiveModel(
        hubert_model_name=checkpoint_arg(
            checkpoint_args,
            "hubert_model_name",
            "facebook/hubert-base-ls960",
        ),
        projection_dim=checkpoint_arg(checkpoint_args, "projection_dim", 256),
        freeze_hubert=checkpoint_arg(checkpoint_args, "freeze_hubert", False),
        melody_vocab_size=checkpoint_arg(checkpoint_args, "melody_vocab_size", 234),
        melody_d_model=checkpoint_arg(checkpoint_args, "melody_d_model", 256),
        melody_num_layers=checkpoint_arg(checkpoint_args, "melody_num_layers", 4),
        melody_num_heads=checkpoint_arg(checkpoint_args, "melody_num_heads", 4),
        melody_dim_feedforward=checkpoint_arg(checkpoint_args, "melody_dim_feedforward", 1024),
        melody_max_length=checkpoint_arg(checkpoint_args, "melody_max_length", 4096),
        dropout=checkpoint_arg(checkpoint_args, "dropout", 0.1),
    )
