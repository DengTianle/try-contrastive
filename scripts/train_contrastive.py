from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Sampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prosodia.datasets import GroupedContrastiveDataset, grouped_contrastive_collate
from prosodia.training import (
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


class RandomMelodyTransposition:
    """Randomly shift voiced melody pitch by a uniform number of semitones."""

    def __init__(self, max_semitones: float) -> None:
        if max_semitones < 0:
            raise ValueError("max_semitones must be non-negative")
        self.max_semitones = max_semitones

    def __call__(self, item: dict[str, Any]) -> dict[str, Any]:
        if self.max_semitones <= 0.0:
            return item

        semitones = float(torch.empty(()).uniform_(-self.max_semitones, self.max_semitones))
        log2_shift = semitones / 12.0
        voiced = item["melody_voiced"].to(dtype=torch.bool)

        melody_features = item["melody_features"].clone()
        melody_features[voiced, 0] += log2_shift
        item["melody_features"] = melody_features

        f0_hz = item["melody_f0_hz"].clone()
        f0_hz[voiced] *= 2.0 ** log2_shift
        item["melody_f0_hz"] = f0_hz
        item["melody_transposition_semitones"] = torch.tensor(semitones, dtype=torch.float32)
        return item


def maybe_progress(iterable: Any, enabled: bool, **kwargs: Any) -> Any:
    if enabled and tqdm is not None:
        return tqdm(iterable, **kwargs)
    return iterable


def masked_rms(audio: torch.Tensor, mask: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
    mask_float = mask.to(device=audio.device, dtype=audio.dtype)
    summed = (audio.square() * mask_float).sum(dim=-1, keepdim=keepdim)
    count = mask_float.sum(dim=-1, keepdim=keepdim).clamp_min(1.0)
    return (summed / count).clamp_min(1e-12).sqrt()


def apply_random_audio_gain(
    audio: torch.Tensor,
    valid_mask: torch.Tensor,
    max_abs_gain_db: float,
) -> torch.Tensor:
    if max_abs_gain_db <= 0.0:
        return audio

    gain_db = torch.empty(
        *audio.shape[:2],
        1,
        device=audio.device,
        dtype=audio.dtype,
    ).uniform_(-max_abs_gain_db, max_abs_gain_db)
    gain = torch.pow(audio.new_tensor(10.0), gain_db / 20.0)
    augmented = audio * torch.where(valid_mask.unsqueeze(-1), gain, torch.ones_like(gain))
    return augmented.clamp(-1.0, 1.0)


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
    gain_db: float,
    noise_snr_db: float | None,
    background_mix_prob: float,
    background_mix_snr_db: float,
) -> torch.Tensor:
    sample_mask = candidate_audio_attention_mask.to(dtype=torch.bool)
    valid_mask = candidate_mask.to(dtype=torch.bool)
    augmented = apply_random_audio_gain(
        audio=candidate_input_values,
        valid_mask=valid_mask,
        max_abs_gain_db=gain_db,
    )
    augmented = add_soft_audio_noise(
        audio=augmented,
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


class DifferentSongBatchSampler(Sampler[list[int]]):
    """Yield batches with at most one segment per song while possible."""

    def __init__(
        self,
        dataset: GroupedContrastiveDataset,
        batch_size: int,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        indices_by_song: dict[str, list[int]] = defaultdict(list)
        for index, group in enumerate(self.dataset.groups):
            indices_by_song[group["positive"]["dali_id"]].append(index)
        for indices in indices_by_song.values():
            rng.shuffle(indices)

        active_songs = [song for song, indices in indices_by_song.items() if indices]
        while active_songs:
            rng.shuffle(active_songs)
            chosen_songs = active_songs[: self.batch_size]
            if self.drop_last and len(chosen_songs) < self.batch_size:
                break

            batch = [indices_by_song[song].pop() for song in chosen_songs]
            active_songs = [song for song in active_songs if indices_by_song[song]]
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


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


def train_one_epoch(
    model: MelodyAudioContrastiveModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float,
    global_loss_weight: float,
    symmetric_global_loss: bool,
    audio_gain_db: float,
    audio_noise_snr_db: float | None,
    audio_background_mix_prob: float,
    audio_background_mix_snr_db: float,
    grad_clip_norm: float | None,
    use_amp: bool,
    scaler: torch.amp.GradScaler,
    scheduler: StagedWarmupCosineScheduler,
    progress: bool,
    desc: str,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_hard_loss = 0.0
    total_global_loss = 0.0
    total_correct = 0
    total_global_correct = 0
    total_global_audio_correct = 0
    total_examples = 0
    total_global_examples = 0
    progress_bar = maybe_progress(loader, enabled=progress, desc=desc, leave=False)
    for batch in progress_bar:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            candidate_input_values = augment_candidate_audio(
                candidate_input_values=batch["candidate_input_values"],
                candidate_audio_attention_mask=batch["candidate_audio_attention_mask"],
                candidate_mask=batch["candidate_mask"],
                gain_db=audio_gain_db,
                noise_snr_db=audio_noise_snr_db,
                background_mix_prob=audio_background_mix_prob,
                background_mix_snr_db=audio_background_mix_snr_db,
            )
            melody_embeddings, audio_embeddings = model(
                melody_features=batch["melody_features"],
                melody_attention_mask=batch["melody_attention_mask"],
                candidate_input_values=candidate_input_values,
                candidate_audio_attention_mask=batch["candidate_audio_attention_mask"],
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
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        targets = batch["target"].to(dtype=torch.long)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_hard_loss += float(hard_loss.detach().cpu()) * batch_size
        total_correct += int((hard_logits.argmax(dim=-1) == targets).sum().detach().cpu())
        if global_logits is not None:
            total_global_loss += float(global_loss.detach().cpu()) * batch_size
            global_targets = torch.arange(batch_size, device=global_logits.device)
            total_global_correct += int(
                (global_logits.argmax(dim=-1) == global_targets).sum().detach().cpu()
            )
            if global_audio_logits is not None:
                total_global_audio_correct += int(
                    (global_audio_logits.argmax(dim=-1) == global_targets).sum().detach().cpu()
                )
            total_global_examples += batch_size
        total_examples += batch_size
        if tqdm is not None and hasattr(progress_bar, "set_postfix"):
            progress_bar.set_postfix(
                loss=total_loss / max(total_examples, 1),
                hard=total_hard_loss / max(total_examples, 1),
                acc=total_correct / max(total_examples, 1),
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
    model: MelodyAudioContrastiveModel,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
    use_amp: bool,
    progress: bool,
    desc: str,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_rank = 0.0
    total_reciprocal_rank = 0.0
    total_recall_at_2 = 0
    total_recall_at_3 = 0
    total_recall_at_5 = 0
    total_examples = 0

    progress_bar = maybe_progress(loader, enabled=progress, desc=desc, leave=False)
    for batch in progress_bar:
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            melody_embeddings, audio_embeddings = model(
                melody_features=batch["melody_features"],
                melody_attention_mask=batch["melody_attention_mask"],
                candidate_input_values=batch["candidate_input_values"],
                candidate_audio_attention_mask=batch["candidate_audio_attention_mask"],
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
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int((logits.argmax(dim=-1) == targets).sum().detach().cpu())
        total_rank += float(ranks.sum().detach().cpu())
        total_reciprocal_rank += float((1.0 / ranks.to(dtype=torch.float32)).sum().detach().cpu())
        total_recall_at_2 += int((ranks <= 2).sum().detach().cpu())
        total_recall_at_3 += int((ranks <= 3).sum().detach().cpu())
        total_recall_at_5 += int((ranks <= 5).sum().detach().cpu())
        total_examples += batch_size
        if tqdm is not None and hasattr(progress_bar, "set_postfix"):
            progress_bar.set_postfix(
                loss=total_loss / max(total_examples, 1),
                acc=total_correct / max(total_examples, 1),
            )

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
    model: MelodyAudioContrastiveModel,
    optimizer: torch.optim.Optimizer,
    scheduler: StagedWarmupCosineScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "global_step": scheduler.update_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "args": vars(args),
        "metrics": metrics,
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
    load_projection: bool,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = melody_encoder_state_from_checkpoint(checkpoint)
    if not load_projection:
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
        strict=strict and load_projection,
    )
    if incompatible.missing_keys:
        print(f"Missing melody pretrain keys: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        print(f"Unexpected melody pretrain keys: {incompatible.unexpected_keys}")
    print(f"Loaded melody encoder pretrain from {checkpoint_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train melody/audio contrastive encoders.")
    parser.add_argument("--manifest", type=Path, default=Path("data/prepared/dali/segments_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/contrastive"))
    parser.add_argument("--hubert-model-name", default="facebook/hubert-base-ls960")
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
        "--disable-song-balanced-batches",
        action="store_true",
        help="Use ordinary shuffled batches instead of enforcing different songs per training batch.",
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
            "Minimum same-song negative offset when --manifest points to a segment manifest. "
            "Defaults to the anchor segment length."
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
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision.")
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
        help="Require an exact key match when loading --melody-pretrained-checkpoint.",
    )
    parser.add_argument(
        "--load-melody-pretrained-projection",
        action="store_true",
        help=(
            "Also load the melody encoder projection head from pretraining. "
            "By default it is trained from scratch for the contrastive space."
        ),
    )
    parser.add_argument(
        "--melody-transpose-semitones",
        type=float,
        default=0.0,
        help=(
            "Training-only melody augmentation. Randomly shift voiced melody log-F0 "
            "within +/- this many semitones. 0 disables it."
        ),
    )
    parser.add_argument(
        "--audio-gain-db",
        type=float,
        default=0.0,
        help=(
            "Training-only audio augmentation. Randomly scale each candidate waveform "
            "within +/- this many dB. 0 disables it; try 3."
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
    if args.melody_transpose_semitones < 0.0:
        raise SystemExit("--melody-transpose-semitones must be non-negative")
    if args.audio_gain_db < 0.0:
        raise SystemExit("--audio-gain-db must be non-negative")
    if args.audio_noise_snr_db is not None and args.audio_noise_snr_db <= 0.0:
        raise SystemExit("--audio-noise-snr-db must be positive")
    if not 0.0 <= args.audio_background_mix_prob <= 1.0:
        raise SystemExit("--audio-background-mix-prob must be in [0, 1]")
    if args.audio_background_mix_snr_db <= 0.0:
        raise SystemExit("--audio-background-mix-snr-db must be positive")

    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    use_amp = args.amp and device.type == "cuda"
    progress = not args.no_progress
    if progress and tqdm is None:
        print("tqdm is not installed; continuing without progress bars.")

    train_transform = None
    if args.melody_transpose_semitones > 0.0:
        train_transform = RandomMelodyTransposition(
            max_semitones=args.melody_transpose_semitones,
        )

    train_dataset = GroupedContrastiveDataset(
        manifest_path=args.manifest,
        split=args.train_split,
        max_negatives=args.max_negatives,
        min_negative_offset_seconds=args.min_negative_offset_seconds,
        seed=args.seed,
        transform=train_transform,
    )
    if len(train_dataset) == 0:
        raise SystemExit(f"No grouped training examples found for split={args.train_split}")

    if args.global_loss_weight > 0.0 and not args.disable_song_balanced_batches:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=DifferentSongBatchSampler(
                dataset=train_dataset,
                batch_size=args.batch_size,
                seed=args.seed,
            ),
            num_workers=args.num_workers,
            collate_fn=grouped_contrastive_collate,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=grouped_contrastive_collate,
        )

    val_loader = None
    if not args.no_val:
        val_dataset = GroupedContrastiveDataset(
            manifest_path=args.manifest,
            split=args.val_split,
            max_negatives=args.max_negatives,
            min_negative_offset_seconds=args.min_negative_offset_seconds,
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
            load_projection=args.load_melody_pretrained_projection,
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
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    args.steps_per_epoch = steps_per_epoch
    args.total_training_steps = total_training_steps
    args.hubert_unfreeze_step = hubert_unfreeze_step
    args.head_warmup_steps = scheduler.head_warmup_steps
    args.hubert_warmup_steps = scheduler.hubert_warmup_steps
    args.effective_hubert_trainable_layers = eventual_hubert_trainable_layers

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True, default=str)

    best_metric_name = args.best_checkpoint_metric
    best_metric_direction = checkpoint_metric_direction(best_metric_name)
    best_metric_value = float("inf") if best_metric_direction == "min" else -float("inf")
    for epoch in range(1, args.epochs + 1):
        desired_hubert_trainable_layers = (
            eventual_hubert_trainable_layers
            if epoch > args.hubert_freeze_epochs
            else 0
        )
        if args.freeze_hubert:
            desired_hubert_trainable_layers = 0
        if (
            model.audio_encoder.hubert_trainable_layers
            != desired_hubert_trainable_layers
        ):
            model.audio_encoder.set_hubert_trainable_layers(
                desired_hubert_trainable_layers
            )
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
            audio_gain_db=args.audio_gain_db,
            audio_noise_snr_db=args.audio_noise_snr_db,
            audio_background_mix_prob=args.audio_background_mix_prob,
            audio_background_mix_snr_db=args.audio_background_mix_snr_db,
            grad_clip_norm=args.grad_clip_norm,
            use_amp=use_amp,
            scaler=scaler,
            scheduler=scheduler,
            progress=progress,
            desc=f"train epoch {epoch}/{args.epochs}",
        )

        learning_rates = scheduler.learning_rates()
        metrics: dict[str, Any] = {
            "train": train_metrics,
            "optimization": {
                "global_step": scheduler.update_step,
                "head_lr": learning_rates.get("head", 0.0),
                "hubert_lr": learning_rates.get("hubert", 0.0),
                "hubert_trainable_layers": model.audio_encoder.hubert_trainable_layers,
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
            f"hubert_layers={model.audio_encoder.hubert_trainable_layers}"
        )
        if args.global_loss_weight > 0.0:
            message += f" train_global_acc={train_metrics['global_accuracy']:.4f}"
            if args.symmetric_global_loss:
                message += (
                    f" train_global_audio_acc={train_metrics['global_audio_accuracy']:.4f}"
                )

        if val_loader is not None:
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                temperature=args.temperature,
                use_amp=use_amp,
                progress=progress,
                desc=f"val epoch {epoch}/{args.epochs}",
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


if __name__ == "__main__":
    main()
