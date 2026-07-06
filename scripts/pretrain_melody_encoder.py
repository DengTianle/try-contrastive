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

from prosodia.datasets import MelodyConfig, MelodyOnlyDataset, melody_only_collate
from prosodia.melody_encoder import MelodyMaskedTokenModel


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
    mask_span_tokens: int,
) -> torch.Tensor:
    if not 0.0 < mask_prob < 1.0:
        raise ValueError("mask_prob must be in (0, 1)")
    if mask_span_tokens <= 0:
        raise ValueError("mask_span_tokens must be positive")

    token_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
    for batch_index, valid_length_tensor in enumerate(attention_mask.sum(dim=1)):
        valid_length = int(valid_length_tensor.detach().cpu())
        if valid_length <= 0:
            continue
        target_tokens = max(1, int(round(valid_length * mask_prob)))
        num_spans = max(1, math.ceil(target_tokens / mask_span_tokens))
        starts = torch.randint(
            low=0,
            high=valid_length,
            size=(num_spans,),
            device=attention_mask.device,
        )
        for start_tensor in starts:
            start = int(start_tensor.detach().cpu())
            end = min(start + mask_span_tokens, valid_length)
            token_mask[batch_index, start:end] = True
    return token_mask & attention_mask.to(dtype=torch.bool)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.any():
        return values[mask].mean()
    return values.new_zeros(())


def pretrain_step(
    model: MelodyMaskedTokenModel,
    batch: dict[str, Any],
    mask_prob: float,
    mask_span_tokens: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    attention_mask = batch["melody_attention_mask"].to(dtype=torch.bool)
    token_mask = random_span_mask(
        attention_mask=attention_mask,
        mask_prob=mask_prob,
        mask_span_tokens=mask_span_tokens,
    )
    token_targets = batch["melody_token_ids"]
    melody_inputs = token_targets.masked_fill(token_mask, model.mask_token_id)

    outputs = model(
        melody_token_ids=melody_inputs,
        melody_attention_mask=attention_mask,
    )

    loss_mask = token_mask & attention_mask
    if not bool(loss_mask.any()):
        loss = outputs["token_logits"].new_zeros(())
    else:
        loss = F.cross_entropy(outputs["token_logits"][loss_mask], token_targets[loss_mask])

    with torch.no_grad():
        predictions = outputs["token_logits"].argmax(dim=-1)
        token_accuracy = masked_mean(
            (predictions == token_targets).to(dtype=torch.float32),
            loss_mask,
        )
        metrics = {
            "loss": float(loss.detach().cpu()),
            "token_accuracy": float(token_accuracy.detach().cpu()),
            "masked_tokens": float(loss_mask.sum().detach().cpu()),
        }
    return loss, metrics


def train_one_epoch(
    model: MelodyMaskedTokenModel,
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
        "token_accuracy": 0.0,
        "masked_tokens": 0.0,
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
                mask_span_tokens=args.mask_span_tokens,
            )

        scaler.scale(loss).backward()
        if args.grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = batch["melody_token_ids"].shape[0]
        for key in totals:
            totals[key] += metrics[key] * batch_size
        total_examples += batch_size
        if tqdm is not None and hasattr(progress_bar, "set_postfix"):
            progress_bar.set_postfix(
                loss=totals["loss"] / max(total_examples, 1),
                acc=totals["token_accuracy"] / max(total_examples, 1),
            )

    return {key: value / max(total_examples, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate(
    model: MelodyMaskedTokenModel,
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
        "token_accuracy": 0.0,
        "masked_tokens": 0.0,
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
                mask_span_tokens=args.mask_span_tokens,
            )

        batch_size = batch["melody_token_ids"].shape[0]
        for key in totals:
            totals[key] += metrics[key] * batch_size
        total_examples += batch_size
        if tqdm is not None and hasattr(progress_bar, "set_postfix"):
            progress_bar.set_postfix(
                loss=totals["loss"] / max(total_examples, 1),
                acc=totals["token_accuracy"] / max(total_examples, 1),
            )

    return {key: value / max(total_examples, 1) for key, value in totals.items()}


def save_checkpoint(
    output_dir: Path,
    name: str,
    model: MelodyMaskedTokenModel,
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
        "objective": "masked_note_token_modeling",
    }
    torch.save(checkpoint, output_dir / name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain the melody encoder on masked note tokens.")
    parser.add_argument("--manifest", type=Path, default=Path("data/prepared_good/segments_manifest.csv"))
    parser.add_argument(
        "--quantization-dir",
        type=Path,
        default=None,
        help="Directory containing events.jsonl and ratio_vocabulary.json. Defaults to <manifest-dir>/quantization.",
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
    parser.add_argument("--melody-max-length", type=int, default=4096)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask-prob", type=float, default=0.35)
    parser.add_argument("--mask-span-tokens", type=int, default=3)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.mask_prob < 1.0:
        raise SystemExit("--mask-prob must be in (0, 1)")
    if args.mask_span_tokens <= 0:
        raise SystemExit("--mask-span-tokens must be positive")
    if args.melody_max_length <= 0:
        raise SystemExit("--melody-max-length must be positive")


def main() -> None:
    args = parse_args()
    args.manifest = resolve_user_path(args.manifest)
    if args.quantization_dir is not None:
        args.quantization_dir = resolve_user_path(args.quantization_dir)
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
        melody_config=MelodyConfig(quantization_dir=args.quantization_dir),
    )
    if len(train_dataset) == 0:
        raise SystemExit(f"No melody training examples found for split={args.train_split}")
    args.melody_vocab_size = train_dataset.melody_vocab_size
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
            melody_config=MelodyConfig(quantization_dir=args.quantization_dir),
        )
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=melody_only_collate,
            )

    model = MelodyMaskedTokenModel(
        vocab_size=args.melody_vocab_size,
        projection_dim=args.projection_dim,
        d_model=args.melody_d_model,
        num_layers=args.melody_num_layers,
        num_heads=args.melody_num_heads,
        dim_feedforward=args.melody_dim_feedforward,
        dropout=args.dropout,
        max_length=args.melody_max_length,
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
            f"train_token_acc={train_metrics['token_accuracy']:.4f} "
            f"train_masked_tokens={train_metrics['masked_tokens']:.1f}"
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
                f"val_token_acc={val_metrics['token_accuracy']:.4f}"
            )
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(args.output_dir, "best.pt", model, optimizer, epoch, args, metrics)

        print(message)
        save_checkpoint(args.output_dir, "last.pt", model, optimizer, epoch, args, metrics)


if __name__ == "__main__":
    main()
