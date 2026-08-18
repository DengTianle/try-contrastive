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
from prosodia.melody_encoder import (
    DURATION_SLICE,
    MELODY_REPRESENTATION,
    ONSET_SHIFT_SLICE,
    PITCH_CHANGE_SLICE,
    PITCH_SIGN_INDEX,
    MelodyMaskedProsodyModel,
)


try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - convenience fallback for minimal environments.
    tqdm = None


PRETRAIN_METRICS = (
    "loss",
    "pitch_change_loss",
    "pitch_sign_loss",
    "duration_loss",
    "onset_shift_loss",
    "pitch_change_accuracy",
    "pitch_sign_accuracy",
    "duration_accuracy",
    "onset_shift_accuracy",
)


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
    mask_span_notes: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not 0.0 < mask_prob < 1.0:
        raise ValueError("mask_prob must be in (0, 1)")
    if mask_span_notes <= 0:
        raise ValueError("mask_span_notes must be positive")

    note_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
    for batch_index, valid_length_tensor in enumerate(attention_mask.sum(dim=1)):
        valid_length = int(valid_length_tensor.detach().cpu())
        if valid_length <= 0:
            continue
        target_notes = max(1, int(round(valid_length * mask_prob)))
        num_spans = max(1, math.ceil(target_notes / mask_span_notes))
        starts = torch.randint(
            low=0,
            high=valid_length,
            size=(num_spans,),
            generator=generator,
        )
        for start_tensor in starts:
            start = int(start_tensor.detach().cpu())
            end = min(start + mask_span_notes, valid_length)
            note_mask[batch_index, start:end] = True
    return note_mask & attention_mask.to(dtype=torch.bool)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.any():
        return values[mask].mean()
    return values.new_zeros(())


def pretrain_step(
    model: MelodyMaskedProsodyModel,
    batch: dict[str, Any],
    mask_prob: float,
    mask_span_notes: int,
    mask_generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    attention_mask = batch["melody_attention_mask"].to(dtype=torch.bool)
    note_mask = random_span_mask(
        attention_mask=attention_mask,
        mask_prob=mask_prob,
        mask_span_notes=mask_span_notes,
        generator=mask_generator,
    )
    features = batch["melody_features"]
    pitch_change_targets = features[..., PITCH_CHANGE_SLICE].argmax(dim=-1)
    pitch_sign_targets = (features[..., PITCH_SIGN_INDEX] >= 0.5).long()
    duration_targets = features[..., DURATION_SLICE].argmax(dim=-1)
    onset_shift_targets = features[..., ONSET_SHIFT_SLICE].argmax(dim=-1)

    outputs = model(
        melody_features=features,
        melody_attention_mask=attention_mask,
        note_mask=note_mask,
    )

    pitch_change_loss = masked_mean(
        F.cross_entropy(
            outputs["pitch_change_logits"].transpose(1, 2),
            pitch_change_targets,
            reduction="none",
        ),
        note_mask,
    )
    pitch_sign_loss = masked_mean(
        F.cross_entropy(
            outputs["pitch_sign_logits"].transpose(1, 2),
            pitch_sign_targets,
            reduction="none",
        ),
        note_mask,
    )
    duration_loss = masked_mean(
        F.cross_entropy(
            outputs["duration_logits"].transpose(1, 2),
            duration_targets,
            reduction="none",
        ),
        note_mask,
    )
    onset_shift_loss = masked_mean(
        F.cross_entropy(
            outputs["onset_shift_logits"].transpose(1, 2),
            onset_shift_targets,
            reduction="none",
        ),
        note_mask,
    )
    loss = pitch_change_loss + pitch_sign_loss + duration_loss + onset_shift_loss

    with torch.no_grad():
        pitch_change_accuracy = masked_mean(
            (outputs["pitch_change_logits"].argmax(-1) == pitch_change_targets).float(),
            note_mask,
        )
        pitch_sign_accuracy = masked_mean(
            (outputs["pitch_sign_logits"].argmax(-1) == pitch_sign_targets).float(),
            note_mask,
        )
        duration_accuracy = masked_mean(
            (outputs["duration_logits"].argmax(-1) == duration_targets).float(),
            note_mask,
        )
        onset_shift_accuracy = masked_mean(
            (outputs["onset_shift_logits"].argmax(-1) == onset_shift_targets).float(),
            note_mask,
        )
        metrics = {
            "loss": float(loss.detach().cpu()),
            "pitch_change_loss": float(pitch_change_loss.detach().cpu()),
            "pitch_sign_loss": float(pitch_sign_loss.detach().cpu()),
            "duration_loss": float(duration_loss.detach().cpu()),
            "onset_shift_loss": float(onset_shift_loss.detach().cpu()),
            "pitch_change_accuracy": float(pitch_change_accuracy.detach().cpu()),
            "pitch_sign_accuracy": float(pitch_sign_accuracy.detach().cpu()),
            "duration_accuracy": float(duration_accuracy.detach().cpu()),
            "onset_shift_accuracy": float(onset_shift_accuracy.detach().cpu()),
            "masked_notes": float(note_mask.sum().detach().cpu()),
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
    totals = {key: 0.0 for key in PRETRAIN_METRICS}
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
                mask_span_notes=args.mask_span_notes,
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
                pitch_acc=totals["pitch_change_accuracy"] / max(total_examples, 1),
                duration_acc=totals["duration_accuracy"] / max(total_examples, 1),
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
    totals = {key: 0.0 for key in PRETRAIN_METRICS}
    total_examples = 0
    mask_generator = torch.Generator().manual_seed(args.seed)

    progress_bar = maybe_progress(loader, enabled=progress, desc=desc, leave=False)
    for batch in progress_bar:
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            _, metrics = pretrain_step(
                model=model,
                batch=batch,
                mask_prob=args.mask_prob,
                mask_span_notes=args.mask_span_notes,
                mask_generator=mask_generator,
            )

        batch_size = batch["melody_features"].shape[0]
        for key in totals:
            totals[key] += metrics[key] * batch_size
        total_examples += batch_size
        if tqdm is not None and hasattr(progress_bar, "set_postfix"):
            progress_bar.set_postfix(
                loss=totals["loss"] / max(total_examples, 1),
                pitch_acc=totals["pitch_change_accuracy"] / max(total_examples, 1),
                duration_acc=totals["duration_accuracy"] / max(total_examples, 1),
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
        "melody_encoder_state_dict": {
            key: value
            for key, value in model.encoder.state_dict().items()
            if not key.startswith("projection.")
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "metrics": metrics,
        "melody_representation": MELODY_REPRESENTATION,
        "objective": "masked_note_attribute_classification",
    }
    torch.save(checkpoint, output_dir / name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain the melody encoder on masked notes.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/prepared/dali/segments_manifest.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/melody_pretrain"))
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
    parser.add_argument("--mask-span-notes", type=int, default=2)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.mask_prob < 1.0:
        raise SystemExit("--mask-prob must be in (0, 1)")
    if args.mask_span_notes <= 0:
        raise SystemExit("--mask-span-notes must be positive")


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
            f"train_pitch_acc={train_metrics['pitch_change_accuracy']:.4f} "
            f"train_duration_acc={train_metrics['duration_accuracy']:.4f}"
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
                f"val_pitch_acc={val_metrics['pitch_change_accuracy']:.4f} "
                f"val_duration_acc={val_metrics['duration_accuracy']:.4f}"
            )
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(args.output_dir, "best.pt", model, optimizer, epoch, args, metrics)

        print(message)
        save_checkpoint(args.output_dir, "last.pt", model, optimizer, epoch, args, metrics)


if __name__ == "__main__":
    main()
