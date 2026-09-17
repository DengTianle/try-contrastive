from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class HubertEncoderOutput:
    """Clip embedding plus the projected note sequence used to construct it."""

    embeddings: torch.Tensor
    note_embeddings: torch.Tensor
    note_attention_mask: torch.Tensor


def normalize_waveforms(
    input_values: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    epsilon: float = 1e-7,
) -> torch.Tensor:
    """Apply per-waveform zero-mean, unit-variance normalization.

    This mirrors Wav2Vec2FeatureExtractor normalization while excluding padded
    samples from the statistics and restoring padding to zero afterwards.

    Note that we did not use their provided feature extractor since that will
    mean moving to CPU numpy operations then move back to GPU.
    """

    if input_values.ndim != 2:
        raise ValueError(
            f"Expected input_values with shape [batch, samples], got {input_values.shape}"
        )
    if input_values.shape[-1] == 0:
        raise ValueError("Expected input_values to contain at least one sample")
    if not torch.is_floating_point(input_values):
        raise ValueError("Expected floating-point waveform input_values")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    working_dtype = (
        torch.float32
        if input_values.dtype in {torch.float16, torch.bfloat16}
        else input_values.dtype
    )
    working = input_values.to(dtype=working_dtype)
    if attention_mask is None:
        mean = working.mean(dim=-1, keepdim=True)
        variance = working.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (working - mean) * torch.rsqrt(variance + epsilon)
        return normalized.to(dtype=input_values.dtype)

    if attention_mask.shape != input_values.shape:
        raise ValueError(
            "Expected attention_mask to match input_values shape, "
            f"got {attention_mask.shape} and {input_values.shape}"
        )
    valid = attention_mask.to(device=working.device, dtype=working.dtype)
    counts = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (working * valid).sum(dim=-1, keepdim=True) / counts
    centered = working - mean
    variance = (centered.square() * valid).sum(dim=-1, keepdim=True) / counts
    normalized = centered * torch.rsqrt(variance + epsilon)
    normalized = normalized * valid
    return normalized.to(dtype=input_values.dtype)


class HubertEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        projection_dim: int = 256,
        dropout: float = 0.1,
        freeze_hubert: bool = False,
        pooling: str = "note",
    ) -> None:
        super().__init__()
        from transformers import AutoFeatureExtractor, AutoModel

        if pooling not in {"note", "mean"}:
            raise ValueError(f"Unsupported pooling: {pooling}")

        self.model_name = model_name
        self.projection_dim = projection_dim
        self.pooling = pooling
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.normalize_input_waveforms = bool(
            getattr(feature_extractor, "do_normalize", False)
        )
        self.sampling_rate = int(getattr(feature_extractor, "sampling_rate", 16_000))
        self.hubert_uses_attention_mask = bool(
            getattr(feature_extractor, "return_attention_mask", False)
        )
        self.hubert = AutoModel.from_pretrained(model_name)
        hidden_size = self.hubert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Sequential(
            #nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, projection_dim),
        )
        self.frozen_modules_eval = True
        self.hubert_trainable_layers = 0
        self.set_hubert_trainable(not freeze_hubert)

        # The raw-waveform CNN remains frozen even when all transformer blocks are
        # fine-tuned. It is expensive to train and usually transfers well.
        self._freeze_feature_encoder()

        # Trade computation for memory whenever HuBERT gradients are enabled.
        if not freeze_hubert and hasattr(self.hubert, "gradient_checkpointing_enable"):
            self.hubert.gradient_checkpointing_enable()

    @property
    def num_hubert_transformer_layers(self) -> int:
        layers = getattr(getattr(self.hubert, "encoder", None), "layers", None)
        if layers is None:
            raise ValueError(
                "Expected the HuBERT backbone to expose transformer blocks as "
                "hubert.encoder.layers"
            )
        return len(layers)

    def _freeze_feature_encoder(self) -> None:
        if hasattr(self.hubert, "freeze_feature_encoder"):
            self.hubert.freeze_feature_encoder()
            return
        feature_extractor = getattr(self.hubert, "feature_extractor", None)
        if feature_extractor is not None:
            for parameter in feature_extractor.parameters():
                parameter.requires_grad = False

    def _stable_encoder_output_norm(self) -> nn.Module | None:
        config = getattr(self.hubert, "config", None)
        if not bool(getattr(config, "do_stable_layer_norm", False)):
            return None
        return getattr(getattr(self.hubert, "encoder", None), "layer_norm", None)

    def hubert_parameters_for_top_layers(
        self,
        trainable_layers: int,
    ) -> list[tuple[str, nn.Parameter]]:
        """Return HuBERT parameters that will eventually be optimized.

        Partial fine-tuning adapts only the top transformer blocks. Full
        transformer fine-tuning also adapts feature projection, positional
        convolution, and encoder input normalization, while the waveform CNN
        remains frozen.
        """

        total_layers = self.num_hubert_transformer_layers
        if not 0 <= trainable_layers <= total_layers:
            raise ValueError(
                f"trainable_layers must be between 0 and {total_layers}, "
                f"got {trainable_layers}"
            )
        if trainable_layers == 0:
            return []

        feature_extractor = getattr(self.hubert, "feature_extractor", None)
        selected_parameter_ids: set[int] = set()
        if trainable_layers == total_layers:
            for parameter in self.hubert.parameters():
                selected_parameter_ids.add(id(parameter))
            if feature_extractor is not None:
                for parameter in feature_extractor.parameters():
                    selected_parameter_ids.discard(id(parameter))
        else:
            layers = self.hubert.encoder.layers
            for layer in layers[-trainable_layers:]:
                for parameter in layer.parameters():
                    selected_parameter_ids.add(id(parameter))
            output_norm = self._stable_encoder_output_norm()
            if output_norm is not None:
                for parameter in output_norm.parameters():
                    selected_parameter_ids.add(id(parameter))

        return [
            (name, parameter)
            for name, parameter in self.hubert.named_parameters()
            if id(parameter) in selected_parameter_ids
        ]

    def set_hubert_trainable_layers(self, trainable_layers: int) -> None:
        """Train the top N transformer blocks and freeze all lower HuBERT modules."""

        selected = self.hubert_parameters_for_top_layers(trainable_layers)
        selected_parameter_ids = {id(parameter) for _, parameter in selected}
        for parameter in self.hubert.parameters():
            parameter.requires_grad = id(parameter) in selected_parameter_ids

        self._freeze_feature_encoder()
        self.hubert_trainable_layers = trainable_layers
        self.hubert_trainable = trainable_layers > 0
        if self.hubert_trainable and hasattr(self.hubert, "gradient_checkpointing_enable"):
            self.hubert.gradient_checkpointing_enable()
        self._apply_hubert_module_modes(self.training)

    def set_hubert_trainable(self, trainable: bool) -> None:
        trainable_layers = self.num_hubert_transformer_layers if trainable else 0
        self.set_hubert_trainable_layers(trainable_layers)

    def configure_hubert_regularization(
        self,
        *,
        apply_spec_augment: bool,
        layerdrop: float,
        frozen_modules_eval: bool = True,
    ) -> None:
        if not 0.0 <= layerdrop < 1.0:
            raise ValueError("layerdrop must be in [0, 1)")
        self.hubert.config.apply_spec_augment = apply_spec_augment
        self.hubert.config.layerdrop = layerdrop
        self.frozen_modules_eval = frozen_modules_eval
        self._apply_hubert_module_modes(self.training)

    def _apply_hubert_module_modes(self, mode: bool) -> None:
        if not self.hubert_trainable:
            self.hubert.eval()
            return

        self.hubert.train(mode)
        if not mode or not self.frozen_modules_eval:
            return

        # Keep deterministic outputs from the frozen prefix while leaving the
        # selected top blocks in training mode. The parent encoder stays in
        # training mode, so its LayerDrop setting is controlled explicitly via
        # configure_hubert_regularization.
        for module_name in ("feature_extractor", "feature_projection"):
            module = getattr(self.hubert, module_name, None)
            if module is not None and not any(
                parameter.requires_grad for parameter in module.parameters()
            ):
                module.eval()

        encoder = getattr(self.hubert, "encoder", None)
        if encoder is None:
            return
        encoder_dropout = getattr(encoder, "dropout", None)
        if (
            encoder_dropout is not None
            and self.hubert_trainable_layers < self.num_hubert_transformer_layers
        ):
            encoder_dropout.eval()
        for module_name in ("pos_conv_embed", "layer_norm"):
            module = getattr(encoder, module_name, None)
            if module is not None and not any(
                parameter.requires_grad for parameter in module.parameters()
            ):
                module.eval()
        for layer in encoder.layers:
            if any(parameter.requires_grad for parameter in layer.parameters()):
                layer.train(True)
            else:
                layer.eval()

    def train(self, mode: bool = True) -> "HubertEncoder":
        super().train(mode)
        self._apply_hubert_module_modes(mode)
        return self

    def _feature_frame_centers(
        self,
        num_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return HuBERT feature-frame centers measured in waveform samples."""

        kernels = getattr(self.hubert.config, "conv_kernel", None)
        strides = getattr(self.hubert.config, "conv_stride", None)
        if kernels is None or strides is None or len(kernels) != len(strides):
            raise ValueError("HuBERT config must define matching conv_kernel/conv_stride")

        receptive_field = 1
        total_stride = 1
        for kernel, stride in zip(kernels, strides):
            receptive_field += (int(kernel) - 1) * total_stride
            total_stride *= int(stride)
        first_center = (receptive_field - 1) / 2.0
        return (
            torch.arange(num_frames, device=device, dtype=dtype) * total_stride
            + first_center
        )

    def _pool_frames_to_notes(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        note_onsets: torch.Tensor,
        note_durations: torch.Tensor,
        note_attention_mask: torch.Tensor,
        note_inputs_validated: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if note_onsets.shape != note_durations.shape:
            raise ValueError("note_onsets and note_durations must have equal shapes")
        if note_attention_mask.shape != note_onsets.shape:
            raise ValueError("note_attention_mask must match the note timing shape")
        if note_onsets.ndim != 2 or note_onsets.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "Expected note timing with shape [batch, notes], got "
                f"{note_onsets.shape}"
            )

        if attention_mask is None:
            feature_attention_mask = torch.ones(
                hidden_states.shape[:2],
                device=hidden_states.device,
                dtype=torch.bool,
            )
        else:
            feature_attention_mask = self.hubert._get_feature_vector_attention_mask(
                hidden_states.shape[1],
                attention_mask,
            ).to(device=hidden_states.device, dtype=torch.bool)

        timing_dtype = hidden_states.dtype
        if timing_dtype in {torch.float16, torch.bfloat16}:
            timing_dtype = torch.float32
        frame_centers = self._feature_frame_centers(
            hidden_states.shape[1],
            device=hidden_states.device,
            dtype=timing_dtype,
        )
        starts = note_onsets.to(device=hidden_states.device, dtype=timing_dtype)
        ends = starts + note_durations.to(
            device=hidden_states.device,
            dtype=timing_dtype,
        )
        valid_notes = note_attention_mask.to(
            device=hidden_states.device,
            dtype=torch.bool,
        ) & (ends > starts)
        frame_membership = (
            (frame_centers[None, None, :] >= starts[:, :, None] * self.sampling_rate)
            & (frame_centers[None, None, :] < ends[:, :, None] * self.sampling_rate)
            & feature_attention_mask[:, None, :]
            & valid_notes[:, :, None]
        )

        # HuBERT's valid frames form a contiguous prefix. Locate the two frame
        # centers around each midpoint and choose the closer one (ties go left).
        # This costs O(notes * log(frames)), without a host sync or a dense
        # note-by-frame distance/one-hot allocation for the rare short notes.
        notes_with_frames = frame_membership.any(dim=-1)
        notes_without_frames = valid_notes & ~notes_with_frames
        midpoints = (starts + ends) * (0.5 * self.sampling_rate)
        last_frame = feature_attention_mask.sum(dim=-1, keepdim=True) - 1
        right = torch.minimum(torch.searchsorted(frame_centers, midpoints), last_frame).clamp_min(0)
        left = (right - 1).clamp_min(0)
        nearest_frames = torch.where(
            (midpoints - frame_centers[left]).abs() <= (frame_centers[right] - midpoints).abs(),
            left, right,
        ).unsqueeze(-1)
        waveform_mask = last_frame.squeeze(-1) >= 0
        fallback = (notes_without_frames & waveform_mask[:, None]).unsqueeze(-1)
        frame_membership.scatter_(
            -1, nearest_frames,
            frame_membership.gather(-1, nearest_frames) | fallback,
        )

        pooled_note_mask = valid_notes & (notes_with_frames | fallback.squeeze(-1))
        # Normal batches already validate note presence during CPU collation.
        # Retain validation for direct encoder calls and unvalidated batches.
        if not note_inputs_validated:
            missing_note_rows = waveform_mask & ~pooled_note_mask.any(dim=-1)
            if bool(missing_note_rows.any()):
                raise ValueError("Every non-empty waveform must contain an annotated note")

        weights = frame_membership.to(dtype=hidden_states.dtype)
        note_sums = torch.einsum("bnt,bth->bnh", weights, hidden_states)
        frame_counts = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        note_hidden_states = note_sums / frame_counts
        note_hidden_states = note_hidden_states * pooled_note_mask.unsqueeze(-1)
        return note_hidden_states, pooled_note_mask

    def _pool_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Mean-pool valid HuBERT frames before the projection head."""
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        feature_mask = self.hubert._get_feature_vector_attention_mask(
            hidden_states.shape[1], attention_mask,
        ).to(device=hidden_states.device, dtype=torch.bool)
        mask = feature_mask.unsqueeze(-1)
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

    def _default_single_note_timing(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preserve compatibility by treating an unannotated waveform as one note."""

        if attention_mask is None:
            sample_counts = input_values.new_full(
                (input_values.shape[0],),
                input_values.shape[1],
            )
        else:
            sample_counts = attention_mask.to(
                device=input_values.device,
                dtype=input_values.dtype,
            ).sum(dim=-1)
        onsets = input_values.new_zeros(input_values.shape[0], 1)
        durations = sample_counts[:, None] / self.sampling_rate
        note_mask = sample_counts[:, None] > 0
        return onsets, durations, note_mask

    def _prepare_input_values(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.normalize_input_waveforms:
            return normalize_waveforms(input_values, attention_mask=attention_mask)
        if attention_mask is None:
            return input_values
        if attention_mask.shape != input_values.shape:
            raise ValueError(
                "Expected attention_mask to match input_values shape, "
                f"got {attention_mask.shape} and {input_values.shape}"
            )
        return input_values.masked_fill(
            ~attention_mask.to(device=input_values.device, dtype=torch.bool),
            0.0,
        )

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        note_onsets: torch.Tensor | None = None,
        note_durations: torch.Tensor | None = None,
        note_attention_mask: torch.Tensor | None = None,
        normalize: bool = True,
        return_note_embeddings: bool = False,
        note_inputs_validated: bool = False,
    ) -> torch.Tensor | HubertEncoderOutput:
        if self.pooling == "mean" and return_note_embeddings:
            raise ValueError("return_note_embeddings requires audio pooling='note'")
        note_inputs = (note_onsets, note_durations, note_attention_mask)
        if self.pooling == "note" and all(value is None for value in note_inputs):
            note_onsets, note_durations, note_attention_mask = (
                self._default_single_note_timing(input_values, attention_mask)
            )
        elif self.pooling == "note" and any(value is None for value in note_inputs):
            raise ValueError(
                "note_onsets, note_durations, and note_attention_mask must be "
                "provided together"
            )

        prepared_input_values = self._prepare_input_values(input_values, attention_mask)
        hubert_attention_mask = (
            attention_mask if self.hubert_uses_attention_mask else None
        )
        if self.hubert_trainable:
            outputs = self.hubert(
                input_values=prepared_input_values,
                attention_mask=hubert_attention_mask,
            )
        else:
            with torch.no_grad():
                outputs = self.hubert(
                    input_values=prepared_input_values,
                    attention_mask=hubert_attention_mask,
                )
        hidden_states = self.dropout(outputs.last_hidden_state)
        if self.pooling == "mean":
            embeddings = self.projection(
                self._pool_hidden_states(hidden_states, attention_mask)
            )
            return F.normalize(embeddings, dim=-1) if normalize else embeddings

        note_hidden_states, pooled_note_mask = self._pool_frames_to_notes(
            hidden_states,
            attention_mask,
            note_onsets,
            note_durations,
            note_attention_mask,
            note_inputs_validated=note_inputs_validated,
        )
        note_embeddings = self.projection(note_hidden_states)
        note_embeddings = note_embeddings * pooled_note_mask.unsqueeze(-1)
        note_counts = pooled_note_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        embeddings = note_embeddings.sum(dim=1) / note_counts
        if normalize:
            embeddings = F.normalize(embeddings, dim=-1)
        if return_note_embeddings:
            return HubertEncoderOutput(
                embeddings=embeddings,
                note_embeddings=note_embeddings,
                note_attention_mask=pooled_note_mask,
            )
        return embeddings


HubertProsodyEncoder = HubertEncoder
