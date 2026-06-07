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

from prosodia.datasets import GroupedContrastiveDataset, grouped_contrastive_collate
from prosodia.training import (
    MelodyAudioContrastiveModel,
    global_in_batch_info_nce_loss,
    grouped_info_nce_loss,
    positive_audio_embeddings,
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
    total_examples = 0
    total_global_examples = 0
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    progress_bar = maybe_progress(loader, enabled=progress, desc=desc, leave=False)
    for batch in progress_bar:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            melody_embeddings, audio_embeddings = model(
                melody_features=batch["melody_features"],
                melody_attention_mask=batch["melody_attention_mask"],
                candidate_input_values=batch["candidate_input_values"],
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
            batch_size = batch["melody_features"].shape[0]
            if global_loss_weight > 0.0 and batch_size > 1:
                batch_positive_audio_embeddings = positive_audio_embeddings(
                    candidate_audio_embeddings=audio_embeddings,
                    targets=batch["target"],
                )
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
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int((logits.argmax(dim=-1) == targets).sum().detach().cpu())
        total_examples += batch_size
        if tqdm is not None and hasattr(progress_bar, "set_postfix"):
            progress_bar.set_postfix(
                loss=total_loss / max(total_examples, 1),
                acc=total_correct / max(total_examples, 1),
            )

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train melody/audio contrastive encoders.")
    parser.add_argument("--manifest", type=Path, default=Path("data/prepared/dali/segments_manifest.csv"))
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
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision.")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--melody-d-model", type=int, default=256)
    parser.add_argument("--melody-num-layers", type=int, default=4)
    parser.add_argument("--melody-num-heads", type=int, default=4)
    parser.add_argument("--melody-dim-feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--melody-transpose-semitones",
        type=float,
        default=1.0,
        help=(
            "Training-only melody augmentation. Randomly shift voiced melody log-F0 "
            "within +/- this many semitones. 0 disables it."
        ),
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.manifest = resolve_user_path(args.manifest)
    args.output_dir = resolve_user_path(args.output_dir)
    if args.global_loss_weight < 0.0:
        raise SystemExit("--global-loss-weight must be non-negative")
    if args.no_in_batch_negatives:
        args.global_loss_weight = 0.0
    if args.melody_transpose_semitones < 0.0:
        raise SystemExit("--melody-transpose-semitones must be non-negative")

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

    model = MelodyAudioContrastiveModel(
        hubert_model_name=args.hubert_model_name,
        projection_dim=args.projection_dim,
        freeze_hubert=args.freeze_hubert,
        melody_d_model=args.melody_d_model,
        melody_num_layers=args.melody_num_layers,
        melody_num_heads=args.melody_num_heads,
        melody_dim_feedforward=args.melody_dim_feedforward,
        dropout=args.dropout,
    ).to(device)

    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True, default=str)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            temperature=args.temperature,
            global_loss_weight=args.global_loss_weight,
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
            message += f" val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}"
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(args.output_dir, "best.pt", model, optimizer, epoch, args, metrics)

        print(message)
        save_checkpoint(args.output_dir, "last.pt", model, optimizer, epoch, args, metrics)


if __name__ == "__main__":
    main()
