from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prosodia.datasets import GroupedContrastiveDataset, grouped_contrastive_collate
from prosodia.melody_encoder import MELODY_REPRESENTATION
from prosodia.training import (
    DifferentSongBatchSampler,
    MelodyAudioContrastiveModel,
    global_in_batch_info_nce_loss,
    grouped_info_nce_loss,
    positive_audio_embeddings,
    symmetric_global_in_batch_info_nce_loss,
)


try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - convenience fallback for minimal environments.
    tqdm = None


def resolve_user_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return expanded.resolve(strict=False)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_amp_settings(
    amp: bool,
    amp_dtype: str,
    device: torch.device,
) -> tuple[bool, torch.dtype]:
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[amp_dtype]
    if amp_dtype == "bf16":
        if not amp:
            raise SystemExit("--amp-dtype bf16 requires --amp")
        if device.type != "cuda":
            raise SystemExit("--amp-dtype bf16 requires a CUDA device")
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                raise SystemExit(
                    "--amp-dtype bf16 requires a GPU with native BF16 support; "
                    "use --amp-dtype fp16 on this device"
                )
    return amp and device.type == "cuda", dtype


def initialize_distributed(requested_device: str) -> tuple[torch.device, int, int]:
    """Initialize torchrun's process group and select this process's GPU."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return choose_device(requested_device), 0, 1

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = choose_device(requested_device)
    if device.type == "cuda":
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    elif device.type == "cpu":
        backend = "gloo"
    else:
        raise SystemExit("Distributed training supports CUDA or CPU devices only")
    dist.init_process_group(backend=backend)
    return device, dist.get_rank(), dist.get_world_size()


def unwrap_model(model: nn.Module) -> MelodyAudioContrastiveModel:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def wrap_distributed_model(model: nn.Module, device: torch.device) -> nn.Module:
    if not dist.is_initialized():
        return model
    if device.type == "cuda":
        return DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
        )
    return DistributedDataParallel(model)


def distributed_totals(totals: torch.Tensor) -> list[float]:
    if dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return totals.cpu().tolist()


def metric_totals(size: int, device: torch.device) -> torch.Tensor:
    # MPS does not support float64. Use double precision elsewhere to retain
    # the accuracy of the previous Python-float epoch accumulators.
    dtype = torch.float32 if device.type == "mps" else torch.float64
    return torch.zeros(size, device=device, dtype=dtype)


def maybe_progress(iterable: Any, enabled: bool, **kwargs: Any) -> Any:
    if enabled and tqdm is not None:
        return tqdm(iterable, **kwargs)
    return iterable


def masked_rms(audio: torch.Tensor, mask: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    mask_float = mask.to(device=audio.device, dtype=audio.dtype)
    summed = (audio.square() * mask_float).sum(dim=-1, keepdim=keepdim)
    count = mask_float.sum(dim=-1, keepdim=keepdim).clamp_min(1.0)
    return (summed / count).clamp_min(1e-12).sqrt()


def add_soft_audio_noise(
    audio: torch.Tensor,
    sample_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    snr_db: float | None,
) -> torch.Tensor:
    if snr_db is None:
        return audio

    rms = masked_rms(audio, sample_mask, keepdim=True)
    noise = torch.randn_like(audio) * sample_mask.to(dtype=audio.dtype)
    noise_rms = masked_rms(noise, sample_mask, keepdim=True)
    noise_scale = rms * (10.0 ** (-snr_db / 20.0)) / noise_rms
    augmented = audio + noise * noise_scale
    augmented = torch.where(valid_mask.unsqueeze(-1), augmented, audio)
    return augmented.clamp(-1.0, 1.0)


def add_quiet_background_mix(
    audio: torch.Tensor,
    sample_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    probability: float,
    snr_db: float,
) -> torch.Tensor:
    if probability <= 0.0:
        return audio

    batch_size, num_candidates, num_samples = audio.shape
    flat_audio = audio.reshape(batch_size * num_candidates, num_samples)
    flat_mask = sample_mask.reshape(batch_size * num_candidates, num_samples)
    flat_valid = valid_mask.reshape(batch_size * num_candidates)
    valid_indices = torch.nonzero(flat_valid, as_tuple=False).squeeze(-1)
    if valid_indices.numel() < 2:
        return audio

    apply_mask = torch.rand(valid_indices.shape, device=audio.device) < probability
    target_indices = valid_indices[apply_mask]
    if target_indices.numel() == 0:
        return audio

    donor_positions = torch.randint(
        low=0,
        high=valid_indices.numel() - 1,
        size=target_indices.shape,
        device=audio.device,
    )
    donor_indices = valid_indices[donor_positions]
    donor_indices = torch.where(
        donor_indices >= target_indices,
        valid_indices[donor_positions + 1],
        donor_indices,
    )

    targets = flat_audio[target_indices]
    target_masks = flat_mask[target_indices]
    donors = flat_audio[donor_indices] * target_masks.to(dtype=audio.dtype)
    target_rms = masked_rms(targets, target_masks, keepdim=True)
    donor_rms = masked_rms(donors, target_masks, keepdim=True)
    donor_scale = target_rms * (10.0 ** (-snr_db / 20.0)) / donor_rms

    augmented = flat_audio.clone()
    augmented[target_indices] = targets + donors * donor_scale
    return augmented.reshape_as(audio).clamp(-1.0, 1.0)


def augment_candidate_audio(
    candidate_input_values: torch.Tensor,
    candidate_audio_attention_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
    noise_snr_db: float | None,
    background_mix_prob: float,
    background_mix_snr_db: float,
) -> torch.Tensor:
    sample_mask = candidate_audio_attention_mask.to(dtype=torch.bool)
    valid_mask = candidate_mask.to(dtype=torch.bool)
    augmented = add_soft_audio_noise(
        audio=candidate_input_values,
        sample_mask=sample_mask,
        valid_mask=valid_mask,
        snr_db=noise_snr_db,
    )
    return add_quiet_background_mix(
        audio=augmented,
        sample_mask=sample_mask,
        valid_mask=valid_mask,
        probability=background_mix_prob,
        snr_db=background_mix_snr_db,
    )


def parameter_uses_weight_decay(name: str, parameter: torch.Tensor) -> bool:
    """Decay matrix-like weights, but not biases or normalization gain/offset."""

    return parameter.ndim >= 2 and not name.endswith(".bias")


def build_adamw_parameter_groups(
    model: MelodyAudioContrastiveModel,
    *,
    lr: float,
    hubert_lr: float,
    weight_decay: float,
    hubert_trainable_layers: int,
) -> list[dict[str, Any]]:
    eventual_hubert_parameters = (
        model.audio_encoder.hubert_parameters_for_top_layers(hubert_trainable_layers)
    )
    eventual_hubert_parameter_ids = {
        id(parameter) for _, parameter in eventual_hubert_parameters
    }

    grouped: dict[tuple[str, bool], list[torch.Tensor]] = {
        ("head", True): [],
        ("head", False): [],
        ("hubert", True): [],
        ("hubert", False): [],
    }
    for name, parameter in model.named_parameters():
        schedule_name = (
            "hubert" if id(parameter) in eventual_hubert_parameter_ids else "head"
        )
        if schedule_name == "head" and not parameter.requires_grad:
            continue
        #remaining are hubert and head consisting of audio head and whole melody encoder
        uses_decay = parameter_uses_weight_decay(name, parameter)
        grouped[(schedule_name, uses_decay)].append(parameter)

    parameter_groups: list[dict[str, Any]] = []
    for schedule_name in ("head", "hubert"):
        peak_lr = lr if schedule_name == "head" else hubert_lr
        for uses_decay in (True, False):
            parameters = grouped[(schedule_name, uses_decay)]
            if not parameters:
                continue
            parameter_groups.append(
                {
                    "params": parameters,
                    "lr": peak_lr,
                    "peak_lr": peak_lr,
                    "weight_decay": weight_decay if uses_decay else 0.0,
                    "schedule_name": schedule_name,
                    "decay_parameters": uses_decay,
                }
            )

    optimized_parameter_ids = {
        id(parameter)
        for group in parameter_groups
        for parameter in group["params"]
    }
    expected_parameter_ids = {
        id(parameter)
        for _, parameter in model.named_parameters()
        if parameter.requires_grad or id(parameter) in eventual_hubert_parameter_ids
    }
    if optimized_parameter_ids != expected_parameter_ids:
        raise RuntimeError("AdamW parameter grouping omitted or duplicated model parameters")
    return parameter_groups


def warmup_cosine_multiplier(
    update_step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    if total_steps <= 0:
        return 1.0
    bounded_step = min(max(update_step, 0), total_steps - 1)
    if warmup_steps > 0 and bounded_step < warmup_steps:
        return float(bounded_step + 1) / float(warmup_steps)
    decay_steps = max(total_steps - warmup_steps - 1, 1)
    decay_progress = min(
        max((bounded_step - warmup_steps) / decay_steps, 0.0),
        1.0,
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


class StagedWarmupCosineScheduler:
    """Per-update schedules for heads and a later-unfrozen HuBERT parameter group."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        total_steps: int,
        head_warmup_steps: int,
        hubert_unfreeze_step: int | None,
        hubert_warmup_steps: int,
        min_lr_ratio: float,
        enabled: bool,
    ) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.head_warmup_steps = max(head_warmup_steps, 0)
        self.hubert_unfreeze_step = hubert_unfreeze_step
        self.hubert_warmup_steps = max(hubert_warmup_steps, 0)
        self.min_lr_ratio = min_lr_ratio
        self.enabled = enabled
        self.update_step = 0
        self._apply_learning_rates()

    def _multiplier_for_group(self, schedule_name: str) -> float:
        if not self.enabled:
            if schedule_name == "hubert":
                if self.hubert_unfreeze_step is None:
                    return 0.0
                if self.update_step < self.hubert_unfreeze_step:
                    return 0.0
            return 1.0
        if schedule_name == "head":
            return warmup_cosine_multiplier(
                self.update_step,
                total_steps=self.total_steps,
                warmup_steps=self.head_warmup_steps,
                min_lr_ratio=self.min_lr_ratio,
            )
        if schedule_name != "hubert":
            raise ValueError(f"Unknown optimizer schedule group: {schedule_name}")
        if self.hubert_unfreeze_step is None:
            return 0.0
        if self.update_step < self.hubert_unfreeze_step:
            return 0.0
        local_step = self.update_step - self.hubert_unfreeze_step
        remaining_steps = max(self.total_steps - self.hubert_unfreeze_step, 1)
        return warmup_cosine_multiplier(
            local_step,
            total_steps=remaining_steps,
            warmup_steps=self.hubert_warmup_steps,
            min_lr_ratio=self.min_lr_ratio,
        )

    def _apply_learning_rates(self) -> None:
        for group in self.optimizer.param_groups:
            peak_lr = float(group["peak_lr"])
            multiplier = self._multiplier_for_group(str(group["schedule_name"]))
            group["lr"] = peak_lr * multiplier

    def step(self) -> None:
        self.update_step += 1
        self._apply_learning_rates()

    def learning_rates(self) -> dict[str, float]:
        rates: dict[str, float] = {}
        for group in self.optimizer.param_groups:
            schedule_name = str(group["schedule_name"])
            rates[schedule_name] = max(rates.get(schedule_name, 0.0), float(group["lr"]))
        return rates

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "head_warmup_steps": self.head_warmup_steps,
            "hubert_unfreeze_step": self.hubert_unfreeze_step,
            "hubert_warmup_steps": self.hubert_warmup_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "enabled": self.enabled,
            "update_step": self.update_step,
        }


def step_optimizer_and_scheduler(
    scaler: torch.amp.GradScaler,
    optimizer: torch.optim.Optimizer,
    scheduler: StagedWarmupCosineScheduler,
) -> bool:
    """Advance the scheduler only when GradScaler performs the optimizer step."""
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    step_succeeded = scaler.get_scale() >= scale_before
    if step_succeeded:
        scheduler.step()
    return step_succeeded


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float,
    global_loss_weight: float,
    symmetric_global_loss: bool,
    audio_noise_snr_db: float | None,
    audio_background_mix_prob: float,
    audio_background_mix_snr_db: float,
    grad_clip_norm: float | None,
    use_amp: bool,
    scaler: torch.amp.GradScaler,
    scheduler: StagedWarmupCosineScheduler,
    progress: bool,
    desc: str,
    log_every_steps: int = 50,
    amp_dtype: torch.dtype = torch.float16,
) -> dict[str, float]:
    model.train()
    totals = metric_totals(6, device)
    (
        total_loss, total_hard_loss, total_global_loss,
        total_correct, total_global_correct, total_global_audio_correct,
    ) = totals.unbind()
    total_examples = 0
    total_global_examples = 0
    progress_bar = maybe_progress(loader, enabled=progress, desc=desc, leave=False)
    for step, batch in enumerate(progress_bar, start=1):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            candidate_input_values = augment_candidate_audio(
                candidate_input_values=batch["candidate_input_values"],
                candidate_audio_attention_mask=batch["candidate_audio_attention_mask"],
                candidate_mask=batch["candidate_mask"],
                noise_snr_db=audio_noise_snr_db,
                background_mix_prob=audio_background_mix_prob,
                background_mix_snr_db=audio_background_mix_snr_db,
            )
            melody_embeddings, audio_embeddings = model(
                melody_features=batch["melody_features"],
                melody_attention_mask=batch["melody_attention_mask"],
                candidate_input_values=candidate_input_values,
                candidate_audio_attention_mask=batch["candidate_audio_attention_mask"],
                candidate_note_onsets=batch["candidate_note_onsets"],
                candidate_note_durations=batch["candidate_note_durations"],
                candidate_note_attention_mask=batch["candidate_note_attention_mask"],
                candidate_notes_validated=batch.get("candidate_notes_validated", False),
            )
            hard_loss, hard_logits = grouped_info_nce_loss(
                melody_embeddings=melody_embeddings,
                candidate_audio_embeddings=audio_embeddings,
                candidate_mask=batch["candidate_mask"],
                targets=batch["target"],
                temperature=temperature,
            )
            global_loss = hard_loss.new_zeros(())
            global_logits = None
            global_audio_logits = None
            batch_size = batch["melody_features"].shape[0]
            if global_loss_weight > 0.0 and batch_size > 1:
                # DDP does not gather forward outputs, so this remains local to
                # the current GPU; only the resulting gradients are synchronized.
                batch_positive_audio_embeddings = positive_audio_embeddings(
                    candidate_audio_embeddings=audio_embeddings,
                    targets=batch["target"],
                )
                if symmetric_global_loss:
                    global_loss, global_logits, global_audio_logits = (
                        symmetric_global_in_batch_info_nce_loss(
                            melody_embeddings=melody_embeddings,
                            positive_audio_embeddings=batch_positive_audio_embeddings,
                            temperature=temperature,
                        )
                    )
                else:
                    global_loss, global_logits = global_in_batch_info_nce_loss(
                        melody_embeddings=melody_embeddings,
                        positive_audio_embeddings=batch_positive_audio_embeddings,
                        temperature=temperature,
                    )
            loss = hard_loss + global_loss_weight * global_loss

        scaler.scale(loss).backward()
        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        step_optimizer_and_scheduler(scaler, optimizer, scheduler)

        targets = batch["target"].to(dtype=torch.long)
        total_loss.add_(loss.detach(), alpha=batch_size)
        total_hard_loss.add_(hard_loss.detach(), alpha=batch_size)
        total_correct.add_((hard_logits.argmax(dim=-1) == targets).sum())
        if global_logits is not None:
            total_global_loss.add_(global_loss.detach(), alpha=batch_size)
            global_targets = torch.arange(batch_size, device=global_logits.device)
            total_global_correct.add_(
                (global_logits.argmax(dim=-1) == global_targets).sum()
            )
            if global_audio_logits is not None:
                total_global_audio_correct.add_(
                    (global_audio_logits.argmax(dim=-1) == global_targets).sum()
                )
            total_global_examples += batch_size
        total_examples += batch_size
        if (
            progress and log_every_steps > 0 and step % log_every_steps == 0
            and hasattr(progress_bar, "set_postfix")
        ):
            logged = totals.cpu().tolist()
            progress_bar.set_postfix(
                loss=logged[0] / max(total_examples, 1),
                hard=logged[1] / max(total_examples, 1),
                acc=logged[3] / max(total_examples, 1),
                refresh=False,
            )

    (
        total_loss,
        total_hard_loss,
        total_global_loss,
        total_correct,
        total_global_correct,
        total_global_audio_correct,
        total_examples,
        total_global_examples,
    ) = distributed_totals(
        torch.cat((totals, totals.new_tensor([total_examples, total_global_examples]))),
    )
    return {
        "loss": total_loss / max(total_examples, 1),
        "hard_loss": total_hard_loss / max(total_examples, 1),
        "global_loss": total_global_loss / max(total_global_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
        "global_accuracy": total_global_correct / max(total_global_examples, 1),
        "global_audio_accuracy": total_global_audio_correct / max(total_global_examples, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
    use_amp: bool,
    progress: bool,
    desc: str,
    log_every_steps: int = 50,
    amp_dtype: torch.dtype = torch.float16,
) -> dict[str, float]:
    model.eval()
    totals = metric_totals(7, device)
    (
        total_loss, total_correct, total_rank, total_reciprocal_rank,
        total_recall_at_2, total_recall_at_3, total_recall_at_5,
    ) = totals.unbind()
    total_examples = 0

    progress_bar = maybe_progress(loader, enabled=progress, desc=desc, leave=False)
    for step, batch in enumerate(progress_bar, start=1):
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            melody_embeddings, audio_embeddings = model(
                melody_features=batch["melody_features"],
                melody_attention_mask=batch["melody_attention_mask"],
                candidate_input_values=batch["candidate_input_values"],
                candidate_audio_attention_mask=batch["candidate_audio_attention_mask"],
                candidate_note_onsets=batch["candidate_note_onsets"],
                candidate_note_durations=batch["candidate_note_durations"],
                candidate_note_attention_mask=batch["candidate_note_attention_mask"],
                candidate_notes_validated=batch.get("candidate_notes_validated", False),
            )
            loss, logits = grouped_info_nce_loss(
                melody_embeddings=melody_embeddings,
                candidate_audio_embeddings=audio_embeddings,
                candidate_mask=batch["candidate_mask"],
                targets=batch["target"],
                temperature=temperature,
            )

        batch_size = batch["melody_features"].shape[0]
        targets = batch["target"].to(dtype=torch.long)
        target_logits = logits.gather(dim=1, index=targets[:, None])
        ranks = (logits >= target_logits).sum(dim=1)
        total_loss.add_(loss.detach(), alpha=batch_size)
        total_correct.add_((logits.argmax(dim=-1) == targets).sum())
        total_rank.add_(ranks.sum())
        total_reciprocal_rank.add_((1.0 / ranks.to(dtype=torch.float32)).sum())
        total_recall_at_2.add_((ranks <= 2).sum())
        total_recall_at_3.add_((ranks <= 3).sum())
        total_recall_at_5.add_((ranks <= 5).sum())
        total_examples += batch_size
        if (
            progress and log_every_steps > 0 and step % log_every_steps == 0
            and hasattr(progress_bar, "set_postfix")
        ):
            logged = totals.cpu().tolist()
            progress_bar.set_postfix(
                loss=logged[0] / max(total_examples, 1),
                acc=logged[1] / max(total_examples, 1),
                refresh=False,
            )

    (
        total_loss, total_correct, total_rank, total_reciprocal_rank,
        total_recall_at_2, total_recall_at_3, total_recall_at_5,
    ) = totals.cpu().tolist()
    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
        "mean_rank": total_rank / max(total_examples, 1),
        "mrr": total_reciprocal_rank / max(total_examples, 1),
        "recall_at_1": total_correct / max(total_examples, 1),
        "recall_at_2": total_recall_at_2 / max(total_examples, 1),
        "recall_at_3": total_recall_at_3 / max(total_examples, 1),
        "recall_at_5": total_recall_at_5 / max(total_examples, 1),
    }


def save_checkpoint(
    output_dir: Path,
    name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: StagedWarmupCosineScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = unwrap_model(model)
    checkpoint = {
        "epoch": epoch,
        "global_step": scheduler.update_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "args": {**vars(args), "audio_pooling": model.audio_encoder.pooling},
        "metrics": metrics,
        "melody_representation": MELODY_REPRESENTATION,
    }
    torch.save(checkpoint, output_dir / name)


def checkpoint_metric_direction(metric_name: str) -> str:
    if metric_name in {"loss", "mean_rank"}:
        return "min"
    return "max"


def is_better_checkpoint(metric_name: str, value: float, best_value: float) -> bool:
    if checkpoint_metric_direction(metric_name) == "min":
        return value < best_value
    return value > best_value


def melody_encoder_state_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "melody_encoder_state_dict" in checkpoint:
        return checkpoint["melody_encoder_state_dict"]

    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError(
            "Checkpoint must contain either melody_encoder_state_dict or model_state_dict"
        )

    encoder_prefixes = ("encoder.", "melody_encoder.")
    for prefix in encoder_prefixes:
        prefix_state = {
            key.removeprefix(prefix): value
            for key, value in model_state.items()
            if key.startswith(prefix)
        }
        if prefix_state:
            return prefix_state

    raise ValueError("Could not find melody encoder weights in checkpoint")


def load_pretrained_melody_encoder(
    model: MelodyAudioContrastiveModel,
    checkpoint_path: Path,
    strict: bool,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = melody_encoder_state_from_checkpoint(checkpoint)
    projection_keys = [key for key in state_dict if key.startswith("projection.")]
    state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("projection.")
    }
    if projection_keys:
        print(
            "Skipped melody pretrain projection keys "
            f"({len(projection_keys)}); contrastive projection starts from scratch."
        )

    incompatible = model.melody_encoder.load_state_dict(
        state_dict,
        strict=False,
    )
    missing_trunk_keys = [
        key for key in incompatible.missing_keys if not key.startswith("projection.")
    ]
    if strict and (missing_trunk_keys or incompatible.unexpected_keys):
        raise RuntimeError(
            "Strict melody pretrain loading failed: "
            f"missing trunk keys={missing_trunk_keys}, "
            f"unexpected keys={incompatible.unexpected_keys}"
        )
    if missing_trunk_keys:
        print(f"Missing melody pretrain trunk keys: {missing_trunk_keys}")
    if incompatible.unexpected_keys:
        print(f"Unexpected melody pretrain keys: {incompatible.unexpected_keys}")
    print(f"Loaded melody encoder pretrain from {checkpoint_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train melody/audio contrastive encoders.")
    parser.add_argument("--manifest", type=Path, default=Path("data/prepared/dali/segments_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/contrastive"))
    parser.add_argument("--hubert-model-name", default="facebook/hubert-base-ls960")
    parser.add_argument(
        "--audio-pooling", choices=["note", "mean"], default="note",
        help="Pool audio by annotated notes (default), or mean-pool HuBERT frames before projection.",
    )
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument(
        "--freeze-hubert",
        action="store_true",
        help=(
            "Keep the complete HuBERT backbone frozen for the entire run. Without "
            "this flag, HuBERT is initially frozen and its top blocks are later unfrozen."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Peak learning rate for the melody encoder and projection heads.",
    )
    parser.add_argument(
        "--hubert-lr",
        type=float,
        default=1e-5,
        help="Peak learning rate for trainable HuBERT parameters.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument(
        "--hubert-freeze-epochs",
        type=int,
        default=4,
        help=(
            "Number of complete head-only epochs before partial HuBERT fine-tuning. "
            "Ignored by --freeze-hubert."
        ),
    )
    parser.add_argument(
        "--hubert-trainable-layers",
        type=int,
        default=4,
        help=(
            "Number of top HuBERT transformer blocks to unfreeze after the frozen "
            "stage. The waveform CNN always remains frozen."
        ),
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=["warmup-cosine", "constant"],
        default="warmup-cosine",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
        help="Fraction of all optimizer updates used to warm up the non-HuBERT groups.",
    )
    parser.add_argument(
        "--hubert-warmup-ratio",
        type=float,
        default=0.05,
        help=(
            "Fraction of post-unfreeze updates used for HuBERT's independent warmup."
        ),
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.01,
        help="Final cosine learning rate as a fraction of each group's peak rate.",
    )
    parser.add_argument(
        "--hubert-spec-augment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable HuBERT's internal SpecAugment while partially fine-tuning. "
            "Disabled by default to avoid silently stacking it with waveform augmentation."
        ),
    )
    parser.add_argument(
        "--hubert-layerdrop",
        type=float,
        default=0.0,
        help=(
            "HuBERT transformer LayerDrop probability during partial fine-tuning. "
            "Defaults to 0 for deterministic layer selection."
        ),
    )
    parser.add_argument(
        "--hubert-frozen-modules-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep the frozen HuBERT prefix in eval mode while its top blocks train, "
            "so dropout in frozen modules does not make their features drift."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-negatives", type=int, default=7, help="Limit negatives per anchor to avoid OOM.")
    parser.add_argument(
        "--positive-variant-policy",
        choices=["self", "any", "retexted", "retexted-first"],
        default="retexted-first",
        help=(
            "How repeat-grouped melody classes provide the positive audio. "
            "The default prefers a congruent occurrence with different lyrics, then "
            "an ordinary repeat, and finally the aligned occurrence."
        ),
    )
    parser.add_argument(
        "--candidate-window-policy",
        choices=["line", "segment", "match-positive"],
        default="match-positive",
        help=(
            "Audio context shown to the model. The default makes every candidate "
            "match the selected positive's duration: shorter negatives receive "
            "random surrounding context and longer negatives are randomly cropped. "
            "Use segment (or the legacy name line) to preserve every complete "
            "prepared candidate interval."
        ),
    )
    parser.add_argument(
        "--disable-song-balanced-batches",
        action="store_true",
        help="Use ordinary shuffled batches instead of enforcing different songs per training batch.",
    )
    parser.add_argument(
        "--drop-incomplete-batches",
        action="store_true",
        help=(
            "Drop examples that cannot be placed into complete training batches. "
            "Song-balanced training retains the maximum feasible number of full "
            "batches with distinct songs."
        ),
    )
    parser.add_argument(
        "--global-loss-weight",
        type=float,
        default=0.5,
        help="Weight for the global in-batch positive-audio loss added to the hard grouped loss.",
    )
    parser.add_argument(
        "--symmetric-global-loss",
        action="store_true",
        help=(
            "Make the global in-batch loss bidirectional by adding audio-to-melody "
            "classification over the same minibatch positives."
        ),
    )
    parser.add_argument(
        "--no-in-batch-negatives",
        action="store_true",
        help="Deprecated alias for --global-loss-weight=0.0.",
    )
    parser.add_argument(
        "--min-negative-offset-seconds",
        type=float,
        default=None,
        help=(
            "Optional minimum difference between same-song segment start times. "
            "By default, all non-overlapping prepared segments are eligible negatives."
        ),
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--no-val", action="store_true")
    parser.add_argument(
        "--best-checkpoint-metric",
        choices=[
            "loss",
            "accuracy",
            "mrr",
            "mean_rank",
            "recall_at_1",
            "recall_at_2",
            "recall_at_3",
            "recall_at_5",
        ],
        default="mrr",
        help=(
            "Validation metric used to save best.pt. Defaults to mrr because it is "
            "less noisy than top-1 on small validation sets while still rewarding rank."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision (FP16 by default).")
    parser.add_argument(
        "--amp-dtype", choices=["fp16", "bf16"], default="fp16",
        help="CUDA mixed-precision dtype used with --amp. BF16 requires native GPU support.",
    )
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--melody-d-model", type=int, default=256)
    parser.add_argument("--melody-num-layers", type=int, default=4)
    parser.add_argument("--melody-num-heads", type=int, default=4)
    parser.add_argument("--melody-dim-feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--melody-pretrained-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional checkpoint from scripts/pretrain_melody_encoder.py. "
            "Only the melody encoder weights are loaded."
        ),
    )
    parser.add_argument(
        "--melody-pretrained-strict",
        action="store_true",
        help=(
            "Require an exact key match for the transferable melody encoder trunk. "
            "The contrastive projection head is always initialized from scratch."
        ),
    )
    parser.add_argument(
        "--audio-noise-snr-db",
        type=float,
        default=None,
        help=(
            "Training-only audio augmentation. Add white noise at this SNR in dB. "
            "Higher is softer; try 35, or 30 for a stronger setting. Omit to disable."
        ),
    )
    parser.add_argument(
        "--audio-background-mix-prob",
        type=float,
        default=0.0,
        help=(
            "Training-only audio augmentation. Probability of mixing a random in-batch "
            "candidate quietly underneath each candidate. 0 disables it; try 0.25."
        ),
    )
    parser.add_argument(
        "--audio-background-mix-snr-db",
        type=float,
        default=35.0,
        help=(
            "SNR for --audio-background-mix-prob in dB. Higher is softer; "
            "35 is deliberately quiet, 30 is stronger."
        ),
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument(
        "--log-every-steps", type=int, default=50,
        help="Refresh train/validation progress metrics every N batches; 0 logs only epoch results.",
    )
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.manifest = resolve_user_path(args.manifest)
    args.output_dir = resolve_user_path(args.output_dir)
    if args.melody_pretrained_checkpoint is not None:
        args.melody_pretrained_checkpoint = resolve_user_path(args.melody_pretrained_checkpoint)
    if args.global_loss_weight < 0.0:
        raise SystemExit("--global-loss-weight must be non-negative")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    if args.log_every_steps < 0:
        raise SystemExit("--log-every-steps must be non-negative")
    if args.lr <= 0.0 or args.hubert_lr <= 0.0:
        raise SystemExit("--lr and --hubert-lr must be positive")
    if args.weight_decay < 0.0:
        raise SystemExit("--weight-decay must be non-negative")
    if args.hubert_freeze_epochs < 0:
        raise SystemExit("--hubert-freeze-epochs must be non-negative")
    if args.hubert_trainable_layers < 0:
        raise SystemExit("--hubert-trainable-layers must be non-negative")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise SystemExit("--warmup-ratio must be in [0, 1)")
    if not 0.0 <= args.hubert_warmup_ratio < 1.0:
        raise SystemExit("--hubert-warmup-ratio must be in [0, 1)")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise SystemExit("--min-lr-ratio must be in [0, 1]")
    if not 0.0 <= args.hubert_layerdrop < 1.0:
        raise SystemExit("--hubert-layerdrop must be in [0, 1)")
    if (
        not args.freeze_hubert
        and args.hubert_trainable_layers > 0
        and args.hubert_freeze_epochs >= args.epochs
    ):
        raise SystemExit(
            "--hubert-freeze-epochs must be smaller than --epochs when HuBERT "
            "fine-tuning is enabled"
        )
    if args.no_in_batch_negatives:
        args.global_loss_weight = 0.0
    if args.audio_noise_snr_db is not None and args.audio_noise_snr_db <= 0.0:
        raise SystemExit("--audio-noise-snr-db must be positive")
    if not 0.0 <= args.audio_background_mix_prob <= 1.0:
        raise SystemExit("--audio-background-mix-prob must be in [0, 1]")
    if args.audio_background_mix_snr_db <= 0.0:
        raise SystemExit("--audio-background-mix-snr-db must be positive")

    device, rank, world_size = initialize_distributed(args.device)
    is_main_process = rank == 0
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    use_amp, amp_dtype = resolve_amp_settings(args.amp, args.amp_dtype, device)
    progress = not args.no_progress and is_main_process
    if progress and tqdm is None:
        print("tqdm is not installed; continuing without progress bars.")

    train_dataset = GroupedContrastiveDataset(
        manifest_path=args.manifest,
        split=args.train_split,
        max_negatives=args.max_negatives,
        min_negative_offset_seconds=args.min_negative_offset_seconds,
        positive_variant_policy=args.positive_variant_policy,
        candidate_window_policy=args.candidate_window_policy,
        randomize_candidate_windows=args.candidate_window_policy == "match-positive",
        seed=args.seed,
    )
    if len(train_dataset) == 0:
        raise SystemExit(f"No grouped training examples found for split={args.train_split}")
    if is_main_process:
        print(f"Training positive variants: {train_dataset.positive_variant_counts()}")
        if world_size > 1:
            print(
                f"Distributed training on {world_size} processes; "
                f"batch size is {args.batch_size} per GPU."
            )

    train_batch_sampler = None
    train_sampler = None
    if args.global_loss_weight > 0.0 and not args.disable_song_balanced_batches:
        train_batch_sampler = DifferentSongBatchSampler(
            dataset=train_dataset,
            batch_size=args.batch_size,
            seed=args.seed,
            drop_last=args.drop_incomplete_batches,
            num_replicas=world_size,
            rank=rank,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=args.num_workers,
            collate_fn=grouped_contrastive_collate,
        )
    else:
        if world_size > 1:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=args.seed,
                drop_last=args.drop_incomplete_batches,
            )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            drop_last=args.drop_incomplete_batches,
            num_workers=args.num_workers,
            collate_fn=grouped_contrastive_collate,
        )

    val_loader = None
    if not args.no_val and is_main_process:
        val_dataset = GroupedContrastiveDataset(
            manifest_path=args.manifest,
            split=args.val_split,
            max_negatives=args.max_negatives,
            min_negative_offset_seconds=args.min_negative_offset_seconds,
            positive_variant_policy=args.positive_variant_policy,
            candidate_window_policy=args.candidate_window_policy,
            randomize_candidate_windows=False,
            seed=args.seed,
        )
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=grouped_contrastive_collate,
            )

    initially_freeze_hubert = args.freeze_hubert or args.hubert_freeze_epochs > 0
    model = MelodyAudioContrastiveModel(
        hubert_model_name=args.hubert_model_name,
        audio_pooling=args.audio_pooling,
        projection_dim=args.projection_dim,
        freeze_hubert=initially_freeze_hubert,
        melody_d_model=args.melody_d_model,
        melody_num_layers=args.melody_num_layers,
        melody_num_heads=args.melody_num_heads,
        melody_dim_feedforward=args.melody_dim_feedforward,
        dropout=args.dropout,
    ).to(device)
    if args.melody_pretrained_checkpoint is not None:
        load_pretrained_melody_encoder(
            model=model,
            checkpoint_path=args.melody_pretrained_checkpoint,
            strict=args.melody_pretrained_strict,
        )

    num_hubert_layers = model.audio_encoder.num_hubert_transformer_layers
    if args.hubert_trainable_layers > num_hubert_layers:
        raise SystemExit(
            f"--hubert-trainable-layers cannot exceed this model's "
            f"{num_hubert_layers} transformer blocks"
        )
    eventual_hubert_trainable_layers = (
        0 if args.freeze_hubert else args.hubert_trainable_layers
    )
    initial_hubert_trainable_layers = (
        0
        if args.freeze_hubert or args.hubert_freeze_epochs > 0
        else eventual_hubert_trainable_layers
    )
    model.audio_encoder.configure_hubert_regularization(
        apply_spec_augment=args.hubert_spec_augment,
        layerdrop=args.hubert_layerdrop,
        frozen_modules_eval=args.hubert_frozen_modules_eval,
    )
    model.audio_encoder.set_hubert_trainable_layers(initial_hubert_trainable_layers)

    parameter_groups = build_adamw_parameter_groups(
        model,
        lr=args.lr,
        hubert_lr=args.hubert_lr,
        weight_decay=args.weight_decay,
        hubert_trainable_layers=eventual_hubert_trainable_layers,
    )
    optimizer = AdamW(
        parameter_groups,
        lr=args.lr,
        weight_decay=0.0,
    )
    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise SystemExit(
            "No complete training batches are feasible; reduce --batch-size or "
            "disable --drop-incomplete-batches."
        )
    total_training_steps = args.epochs * steps_per_epoch
    hubert_unfreeze_step = (
        None
        if eventual_hubert_trainable_layers == 0
        else args.hubert_freeze_epochs * steps_per_epoch
    )
    post_unfreeze_steps = (
        0
        if hubert_unfreeze_step is None
        else total_training_steps - hubert_unfreeze_step
    )
    scheduler = StagedWarmupCosineScheduler(
        optimizer,
        total_steps=total_training_steps,
        head_warmup_steps=round(total_training_steps * args.warmup_ratio),
        hubert_unfreeze_step=hubert_unfreeze_step,
        hubert_warmup_steps=round(post_unfreeze_steps * args.hubert_warmup_ratio),
        min_lr_ratio=args.min_lr_ratio,
        enabled=args.lr_scheduler == "warmup-cosine",
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp and amp_dtype == torch.float16,
    )

    args.steps_per_epoch = steps_per_epoch
    args.total_training_steps = total_training_steps
    args.hubert_unfreeze_step = hubert_unfreeze_step
    args.head_warmup_steps = scheduler.head_warmup_steps
    args.hubert_warmup_steps = scheduler.hubert_warmup_steps
    args.effective_hubert_trainable_layers = eventual_hubert_trainable_layers
    args.world_size = world_size
    args.effective_global_batch_size = args.batch_size * world_size

    if is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(vars(args), handle, indent=2, sort_keys=True, default=str)
    if dist.is_initialized():
        dist.barrier()

    model = wrap_distributed_model(model, device)

    best_metric_name = args.best_checkpoint_metric
    best_metric_direction = checkpoint_metric_direction(best_metric_name)
    best_metric_value = float("inf") if best_metric_direction == "min" else -float("inf")
    for epoch in range(1, args.epochs + 1):
        epoch_index = epoch - 1
        train_dataset.set_epoch(epoch_index)
        if train_batch_sampler is not None:
            train_batch_sampler.set_epoch(epoch_index)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch_index)
        desired_hubert_trainable_layers = (
            eventual_hubert_trainable_layers
            if epoch > args.hubert_freeze_epochs
            else 0
        )
        if args.freeze_hubert:
            desired_hubert_trainable_layers = 0
        base_model = unwrap_model(model)
        if (
            base_model.audio_encoder.hubert_trainable_layers
            != desired_hubert_trainable_layers
        ):
            # DDP records the trainable parameter set when it is constructed, so
            # rebuild the lightweight wrapper when the staged HuBERT freeze ends.
            model = base_model
            base_model.audio_encoder.set_hubert_trainable_layers(
                desired_hubert_trainable_layers
            )
            model = wrap_distributed_model(base_model, device)
            if is_main_process:
                print(
                    f"epoch={epoch} unfreezing top "
                    f"{desired_hubert_trainable_layers}/{num_hubert_layers} "
                    "HuBERT transformer blocks"
                )

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            temperature=args.temperature,
            global_loss_weight=args.global_loss_weight,
            symmetric_global_loss=args.symmetric_global_loss,
            audio_noise_snr_db=args.audio_noise_snr_db,
            audio_background_mix_prob=args.audio_background_mix_prob,
            audio_background_mix_snr_db=args.audio_background_mix_snr_db,
            grad_clip_norm=args.grad_clip_norm,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            scaler=scaler,
            scheduler=scheduler,
            progress=progress,
            desc=f"train epoch {epoch}/{args.epochs}",
            log_every_steps=args.log_every_steps,
        )

        learning_rates = scheduler.learning_rates()
        base_model = unwrap_model(model)
        metrics: dict[str, Any] = {
            "train": train_metrics,
            "optimization": {
                "global_step": scheduler.update_step,
                "head_lr": learning_rates.get("head", 0.0),
                "hubert_lr": learning_rates.get("hubert", 0.0),
                "hubert_trainable_layers": (
                    base_model.audio_encoder.hubert_trainable_layers
                ),
            },
        }
        message = (
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_hard_loss={train_metrics['hard_loss']:.4f} "
            f"train_global_loss={train_metrics['global_loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"head_lr={learning_rates.get('head', 0.0):.2e} "
            f"hubert_lr={learning_rates.get('hubert', 0.0):.2e} "
            f"hubert_layers={base_model.audio_encoder.hubert_trainable_layers}"
        )
        if args.global_loss_weight > 0.0:
            message += f" train_global_acc={train_metrics['global_accuracy']:.4f}"
            if args.symmetric_global_loss:
                message += (
                    f" train_global_audio_acc={train_metrics['global_audio_accuracy']:.4f}"
                )

        if val_loader is not None:
            val_metrics = evaluate(
                model=base_model,
                loader=val_loader,
                device=device,
                temperature=args.temperature,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                progress=progress,
                desc=f"val epoch {epoch}/{args.epochs}",
                log_every_steps=args.log_every_steps,
            )
            metrics["val"] = val_metrics
            message += (
                f" val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} "
                f"val_mrr={val_metrics['mrr']:.4f} "
                f"val_mean_rank={val_metrics['mean_rank']:.2f} "
                f"val_r@2={val_metrics['recall_at_2']:.4f}"
            )
            selected_metric_value = val_metrics[best_metric_name]
            if is_better_checkpoint(
                metric_name=best_metric_name,
                value=selected_metric_value,
                best_value=best_metric_value,
            ):
                best_metric_value = selected_metric_value
                metrics["best_checkpoint"] = {
                    "metric": best_metric_name,
                    "direction": best_metric_direction,
                    "value": best_metric_value,
                }
                save_checkpoint(
                    args.output_dir,
                    "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    args,
                    metrics,
                )

        if is_main_process:
            print(message)
            save_checkpoint(
                args.output_dir,
                "last.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                args,
                metrics,
            )
        if dist.is_initialized():
            dist.barrier()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
