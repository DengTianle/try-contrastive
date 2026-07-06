from __future__ import annotations

import argparse
import json
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

from prosodia.datasets import MelodyConfig, GroupedContrastiveDataset, grouped_contrastive_collate
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
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

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
                melody_token_ids=batch["melody_token_ids"],
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
            batch_size = batch["melody_token_ids"].shape[0]
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
                melody_token_ids=batch["melody_token_ids"],
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

        batch_size = batch["melody_token_ids"].shape[0]
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
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
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
    parser.add_argument("--manifest", type=Path, default=Path("data/prepared_good/segments_manifest.csv"))
    parser.add_argument(
        "--quantization-dir",
        type=Path,
        default=None,
        help="Directory containing events.jsonl and ratio_vocabulary.json. Defaults to <manifest-dir>/quantization.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/contrastive"))
    parser.add_argument("--hubert-model-name", default="facebook/hubert-base-ls960")
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--freeze-hubert", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
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
    parser.add_argument("--melody-max-length", type=int, default=4096)
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
    if args.quantization_dir is not None:
        args.quantization_dir = resolve_user_path(args.quantization_dir)
    args.output_dir = resolve_user_path(args.output_dir)
    if args.melody_pretrained_checkpoint is not None:
        args.melody_pretrained_checkpoint = resolve_user_path(args.melody_pretrained_checkpoint)
    if args.global_loss_weight < 0.0:
        raise SystemExit("--global-loss-weight must be non-negative")
    if args.no_in_batch_negatives:
        args.global_loss_weight = 0.0
    if args.melody_max_length <= 0:
        raise SystemExit("--melody-max-length must be positive")
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

    train_dataset = GroupedContrastiveDataset(
        manifest_path=args.manifest,
        split=args.train_split,
        melody_config=MelodyConfig(quantization_dir=args.quantization_dir),
        max_negatives=args.max_negatives,
        min_negative_offset_seconds=args.min_negative_offset_seconds,
        seed=args.seed,
    )
    if len(train_dataset) == 0:
        raise SystemExit(f"No grouped training examples found for split={args.train_split}")
    args.melody_vocab_size = train_dataset.melody_vocab_size

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
            melody_config=MelodyConfig(quantization_dir=args.quantization_dir),
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

    model = MelodyAudioContrastiveModel(
        hubert_model_name=args.hubert_model_name,
        projection_dim=args.projection_dim,
        freeze_hubert=args.freeze_hubert,
        melody_vocab_size=args.melody_vocab_size,
        melody_d_model=args.melody_d_model,
        melody_num_layers=args.melody_num_layers,
        melody_num_heads=args.melody_num_heads,
        melody_dim_feedforward=args.melody_dim_feedforward,
        melody_max_length=args.melody_max_length,
        dropout=args.dropout,
    ).to(device)
    if args.melody_pretrained_checkpoint is not None:
        load_pretrained_melody_encoder(
            model=model,
            checkpoint_path=args.melody_pretrained_checkpoint,
            strict=args.melody_pretrained_strict,
            load_projection=args.load_melody_pretrained_projection,
        )

    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True, default=str)

    best_metric_name = args.best_checkpoint_metric
    best_metric_direction = checkpoint_metric_direction(best_metric_name)
    best_metric_value = float("inf") if best_metric_direction == "min" else -float("inf")
    for epoch in range(1, args.epochs + 1):
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
            progress=progress,
            desc=f"train epoch {epoch}/{args.epochs}",
        )

        metrics: dict[str, Any] = {"train": train_metrics}
        message = (
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_hard_loss={train_metrics['hard_loss']:.4f} "
            f"train_global_loss={train_metrics['global_loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f}"
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
                save_checkpoint(args.output_dir, "best.pt", model, optimizer, epoch, args, metrics)

        print(message)
        save_checkpoint(args.output_dir, "last.pt", model, optimizer, epoch, args, metrics)


if __name__ == "__main__":
    main()
