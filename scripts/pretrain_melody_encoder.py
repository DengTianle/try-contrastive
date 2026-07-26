from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prosodia.datasets import MelodyOnlyDataset, melody_only_collate
from prosodia.melody_encoder import MelodyMaskedProsodyModel


try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - convenience fallback for minimal environments.
    tqdm = None


def resolve_user_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return expanded.resolve(strict=False)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def maybe_progress(iterable: Any, enabled: bool, **kwargs: Any) -> Any:
    if enabled and tqdm is not None:
        return tqdm(iterable, **kwargs)
    return iterable


def random_span_mask(
    attention_mask: torch.Tensor,
    mask_prob: float,
    mask_span_frames: int,
) -> torch.Tensor:
    if not 0.0 < mask_prob < 1.0:
        raise ValueError("mask_prob must be in (0, 1)")
    if mask_span_frames <= 0:
        raise ValueError("mask_span_frames must be positive")

    frame_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
    for batch_index, valid_length_tensor in enumerate(attention_mask.sum(dim=1)):
        valid_length = int(valid_length_tensor.detach().cpu())
        if valid_length <= 0:
            continue
        target_frames = max(1, int(round(valid_length * mask_prob)))
        num_spans = max(1, math.ceil(target_frames / mask_span_frames))
        starts = torch.randint(
            low=0,
            high=valid_length,
            size=(num_spans,),
            device=attention_mask.device,
        )
        for start_tensor in starts:
            start = int(start_tensor.detach().cpu())
            end = min(start + mask_span_frames, valid_length)
            frame_mask[batch_index, start:end] = True
    return frame_mask & attention_mask.to(dtype=torch.bool)


def augment_melody_inputs(
    melody_features: torch.Tensor,
    melody_voiced: torch.Tensor,
    melody_attention_mask: torch.Tensor,
    transpose_semitones: float,
    pitch_noise_std: float,
    pitch_dropout_prob: float,
) -> torch.Tensor:
    augmented = melody_features.clone()
    voiced = melody_voiced.to(dtype=torch.bool) & melody_attention_mask.to(dtype=torch.bool)

    if transpose_semitones > 0.0:
        shifts = torch.empty(
            melody_features.shape[0],
            1,
            device=melody_features.device,
            dtype=melody_features.dtype,
        ).uniform_(-transpose_semitones / 12.0, transpose_semitones / 12.0)
        augmented[..., 0] = torch.where(voiced, augmented[..., 0] + shifts, augmented[..., 0])

    if pitch_noise_std > 0.0:
        noise = torch.randn_like(augmented[..., 0]) * pitch_noise_std
        augmented[..., 0] = torch.where(voiced, augmented[..., 0] + noise, augmented[..., 0])

    if pitch_dropout_prob > 0.0:
        pitch_dropout = torch.rand_like(augmented[..., 0]) < pitch_dropout_prob
        augmented[..., 0] = torch.where(voiced & pitch_dropout, torch.zeros_like(augmented[..., 0]), augmented[..., 0])

    return augmented


def build_prosody_targets(
    melody_features: torch.Tensor,
    melody_voiced: torch.Tensor,
    melody_attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_f0 = melody_features[..., 0]
    voiced = melody_voiced.to(dtype=torch.bool) & melody_attention_mask.to(dtype=torch.bool)

    delta = torch.zeros_like(log_f0)
    delta_valid = torch.zeros_like(voiced, dtype=torch.bool)
    delta[:, 1:] = log_f0[:, 1:] - log_f0[:, :-1]
    delta_valid[:, 1:] = voiced[:, 1:] & voiced[:, :-1]
    return delta, delta_valid


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.any():
        return values[mask].mean()
    return values.new_zeros(())


def pretrain_step(
    model: MelodyMaskedProsodyModel,
    batch: dict[str, Any],
    mask_prob: float,
    mask_span_frames: int,
    transpose_semitones: float,
    pitch_noise_std: float,
    pitch_dropout_prob: float,
    delta_loss_weight: float,
    voiced_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    attention_mask = batch["melody_attention_mask"].to(dtype=torch.bool)
    frame_mask = random_span_mask(
        attention_mask=attention_mask,
        mask_prob=mask_prob,
        mask_span_frames=mask_span_frames,
    )
    melody_inputs = augment_melody_inputs(
        melody_features=batch["melody_features"],
        melody_voiced=batch["melody_voiced"],
        melody_attention_mask=attention_mask,
        transpose_semitones=transpose_semitones,
        pitch_noise_std=pitch_noise_std,
        pitch_dropout_prob=pitch_dropout_prob,
    )
    delta_targets, delta_valid = build_prosody_targets(
        melody_features=batch["melody_features"],
        melody_voiced=batch["melody_voiced"],
        melody_attention_mask=attention_mask,
    )

    outputs = model(
        melody_features=melody_inputs,
        melody_attention_mask=attention_mask,
        frame_mask=frame_mask,
    )

    voiced_loss_mask = frame_mask & attention_mask
    delta_loss_mask = frame_mask & delta_valid
    delta_loss = masked_mean(
        F.smooth_l1_loss(outputs["delta_log_f0"], delta_targets, reduction="none"),
        delta_loss_mask,
    )
    voiced_loss = masked_mean(
        F.binary_cross_entropy_with_logits(
            outputs["voiced_logits"],
            batch["melody_voiced"].to(dtype=outputs["voiced_logits"].dtype),
            reduction="none",
        ),
        voiced_loss_mask,
    )
    loss = delta_loss_weight * delta_loss + voiced_loss_weight * voiced_loss

    with torch.no_grad():
        voiced_predictions = outputs["voiced_logits"].sigmoid() >= 0.5
        voiced_accuracy = masked_mean(
            (voiced_predictions == batch["melody_voiced"].to(dtype=torch.bool)).to(dtype=torch.float32),
            voiced_loss_mask,
        )
        delta_mae = masked_mean(
            (outputs["delta_log_f0"] - delta_targets).abs(),
            delta_loss_mask,
        )
        metrics = {
            "loss": float(loss.detach().cpu()),
            "delta_loss": float(delta_loss.detach().cpu()),
            "voiced_loss": float(voiced_loss.detach().cpu()),
            "delta_mae": float(delta_mae.detach().cpu()),
            "voiced_accuracy": float(voiced_accuracy.detach().cpu()),
            "masked_frames": float(voiced_loss_mask.sum().detach().cpu()),
            "delta_frames": float(delta_loss_mask.sum().detach().cpu()),
        }
    return loss, metrics


def train_one_epoch(
    model: MelodyMaskedProsodyModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    use_amp: bool,
    progress: bool,
    desc: str,
) -> dict[str, float]:
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    totals = {
        "loss": 0.0,
        "delta_loss": 0.0,
        "voiced_loss": 0.0,
        "delta_mae": 0.0,
        "voiced_accuracy": 0.0,
    }
    total_examples = 0

    progress_bar = maybe_progress(loader, enabled=progress, desc=desc, leave=False)
    for batch in progress_bar:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            loss, metrics = pretrain_step(
                model=model,
                batch=batch,
                mask_prob=args.mask_prob,
                mask_span_frames=args.mask_span_frames,
                transpose_semitones=args.transpose_semitones,
                pitch_noise_std=args.pitch_noise_std,
                pitch_dropout_prob=args.pitch_dropout_prob,
                delta_loss_weight=args.delta_loss_weight,
                voiced_loss_weight=args.voiced_loss_weight,
            )

        scaler.scale(loss).backward()
        if args.grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = batch["melody_features"].shape[0]
        for key in totals:
            totals[key] += metrics[key] * batch_size
        total_examples += batch_size
        if tqdm is not None and hasattr(progress_bar, "set_postfix"):
            progress_bar.set_postfix(
                loss=totals["loss"] / max(total_examples, 1),
                v_acc=totals["voiced_accuracy"] / max(total_examples, 1),
                d_mae=totals["delta_mae"] / max(total_examples, 1),
            )

    return {key: value / max(total_examples, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate(
    model: MelodyMaskedProsodyModel,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    use_amp: bool,
    progress: bool,
    desc: str,
) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "delta_loss": 0.0,
        "voiced_loss": 0.0,
        "delta_mae": 0.0,
        "voiced_accuracy": 0.0,
    }
    total_examples = 0

    progress_bar = maybe_progress(loader, enabled=progress, desc=desc, leave=False)
    for batch in progress_bar:
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            _, metrics = pretrain_step(
                model=model,
                batch=batch,
                mask_prob=args.mask_prob,
                mask_span_frames=args.mask_span_frames,
                transpose_semitones=args.transpose_semitones,
                pitch_noise_std=args.pitch_noise_std,
                pitch_dropout_prob=args.pitch_dropout_prob,
                delta_loss_weight=args.delta_loss_weight,
                voiced_loss_weight=args.voiced_loss_weight,
            )

        batch_size = batch["melody_features"].shape[0]
        for key in totals:
            totals[key] += metrics[key] * batch_size
        total_examples += batch_size
        if tqdm is not None and hasattr(progress_bar, "set_postfix"):
            progress_bar.set_postfix(
                loss=totals["loss"] / max(total_examples, 1),
                v_acc=totals["voiced_accuracy"] / max(total_examples, 1),
                d_mae=totals["delta_mae"] / max(total_examples, 1),
            )

    return {key: value / max(total_examples, 1) for key, value in totals.items()}


def save_checkpoint(
    output_dir: Path,
    name: str,
    model: MelodyMaskedProsodyModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "melody_encoder_state_dict": model.encoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "metrics": metrics,
        "objective": "masked_voicing_and_delta_log_f0",
    }
    torch.save(checkpoint, output_dir / name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain the melody encoder on masked prosody.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/prepared/dali/segments_manifest.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/melody_pretrain"))
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=0)
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
    parser.add_argument("--mask-prob", type=float, default=0.35)
    parser.add_argument("--mask-span-frames", type=int, default=5)
    parser.add_argument("--transpose-semitones", type=float, default=12.0)
    parser.add_argument("--pitch-noise-std", type=float, default=0.01)
    parser.add_argument("--pitch-dropout-prob", type=float, default=0.1)
    parser.add_argument("--delta-loss-weight", type=float, default=1.0)
    parser.add_argument("--voiced-loss-weight", type=float, default=0.5)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.mask_prob < 1.0:
        raise SystemExit("--mask-prob must be in (0, 1)")
    if args.mask_span_frames <= 0:
        raise SystemExit("--mask-span-frames must be positive")
    if args.transpose_semitones < 0.0:
        raise SystemExit("--transpose-semitones must be non-negative")
    if args.pitch_noise_std < 0.0:
        raise SystemExit("--pitch-noise-std must be non-negative")
    if not 0.0 <= args.pitch_dropout_prob < 1.0:
        raise SystemExit("--pitch-dropout-prob must be in [0, 1)")
    if args.delta_loss_weight < 0.0 or args.voiced_loss_weight < 0.0:
        raise SystemExit("Loss weights must be non-negative")


def main() -> None:
    args = parse_args()
    args.manifest = resolve_user_path(args.manifest)
    args.output_dir = resolve_user_path(args.output_dir)
    validate_args(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    use_amp = args.amp and device.type == "cuda"
    progress = not args.no_progress
    if progress and tqdm is None:
        print("tqdm is not installed; continuing without progress bars.")

    train_dataset = MelodyOnlyDataset(
        manifest_path=args.manifest,
        split=args.train_split,
    )
    if len(train_dataset) == 0:
        raise SystemExit(f"No melody training examples found for split={args.train_split}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=melody_only_collate,
    )

    val_loader = None
    if not args.no_val:
        val_dataset = MelodyOnlyDataset(
            manifest_path=args.manifest,
            split=args.val_split,
        )
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=melody_only_collate,
            )

    model = MelodyMaskedProsodyModel(
        projection_dim=args.projection_dim,
        d_model=args.melody_d_model,
        num_layers=args.melody_num_layers,
        num_heads=args.melody_num_heads,
        dim_feedforward=args.melody_dim_feedforward,
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
            args=args,
            use_amp=use_amp,
            progress=progress,
            desc=f"pretrain epoch {epoch}/{args.epochs}",
        )

        metrics: dict[str, Any] = {"train": train_metrics}
        message = (
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_delta_mae={train_metrics['delta_mae']:.4f} "
            f"train_voiced_acc={train_metrics['voiced_accuracy']:.4f}"
        )

        if val_loader is not None:
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                args=args,
                use_amp=use_amp,
                progress=progress,
                desc=f"val epoch {epoch}/{args.epochs}",
            )
            metrics["val"] = val_metrics
            message += (
                f" val_loss={val_metrics['loss']:.4f} "
                f"val_delta_mae={val_metrics['delta_mae']:.4f} "
                f"val_voiced_acc={val_metrics['voiced_accuracy']:.4f}"
            )
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(args.output_dir, "best.pt", model, optimizer, epoch, args, metrics)

        print(message)
        save_checkpoint(args.output_dir, "last.pt", model, optimizer, epoch, args, metrics)


if __name__ == "__main__":
    main()
