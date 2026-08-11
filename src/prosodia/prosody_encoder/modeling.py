from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


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
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        from transformers import AutoFeatureExtractor, AutoModel

        if pooling != "mean":
            raise ValueError(f"Unsupported pooling: {pooling}")

        self.model_name = model_name
        self.projection_dim = projection_dim
        self.pooling = pooling
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.normalize_input_waveforms = bool(
            getattr(feature_extractor, "do_normalize", False)
        )
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
        normalize: bool = True,
    ) -> torch.Tensor:
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
        pooled = self._pool_hidden_states(hidden_states, attention_mask)
        embeddings = self.projection(pooled)
        if normalize:
            embeddings = F.normalize(embeddings, dim=-1)
        return embeddings


HubertProsodyEncoder = HubertEncoder
