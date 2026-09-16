from __future__ import annotations

from typing import Any

from .modeling import MelodyAudioContrastiveModel


def checkpoint_arg(checkpoint_args: dict[str, Any], name: str, default: Any) -> Any:
    value = checkpoint_args.get(name, default)
    return default if value is None else value


def resolve_checkpoint_audio_pooling(
    checkpoint_args: dict[str, Any],
    audio_pooling: str | None = None,
) -> str:
    """Resolve behavior that cannot be inferred from state-dict tensor shapes."""
    saved = checkpoint_args.get("audio_pooling")
    for value in (saved, audio_pooling):
        if value is not None and value not in {"note", "mean"}:
            raise ValueError(f"Unsupported audio pooling: {value!r}")
    if saved is not None:
        if audio_pooling is not None and audio_pooling != saved:
            raise ValueError(
                f"Requested audio pooling {audio_pooling!r} conflicts with checkpoint "
                f"audio_pooling={saved!r}. Use the saved pooling mode."
            )
        return saved
    if audio_pooling is None:
        raise ValueError(
            "Checkpoint has no audio_pooling metadata; weight shapes cannot distinguish "
            "the two modes. Specify --audio-pooling mean for pre-note-pooling checkpoints "
            "or --audio-pooling note for two-stage note-pooling checkpoints."
        )
    return audio_pooling


def build_contrastive_model_from_checkpoint_args(
    checkpoint_args: dict[str, Any],
    *,
    audio_pooling: str | None = None,
) -> MelodyAudioContrastiveModel:
    audio_pooling = resolve_checkpoint_audio_pooling(checkpoint_args, audio_pooling)
    return MelodyAudioContrastiveModel(
        audio_pooling=audio_pooling,
        hubert_model_name=checkpoint_arg(
            checkpoint_args,
            "hubert_model_name",
            "facebook/hubert-base-ls960",
        ),
        projection_dim=checkpoint_arg(checkpoint_args, "projection_dim", 256),
        freeze_hubert=checkpoint_arg(checkpoint_args, "freeze_hubert", False),
        melody_d_model=checkpoint_arg(checkpoint_args, "melody_d_model", 256),
        melody_num_layers=checkpoint_arg(checkpoint_args, "melody_num_layers", 4),
        melody_num_heads=checkpoint_arg(checkpoint_args, "melody_num_heads", 4),
        melody_dim_feedforward=checkpoint_arg(checkpoint_args, "melody_dim_feedforward", 1024),
        dropout=checkpoint_arg(checkpoint_args, "dropout", 0.1),
    )
