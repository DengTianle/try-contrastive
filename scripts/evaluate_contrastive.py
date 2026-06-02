from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prosodia.datasets import GroupedContrastiveDataset, grouped_contrastive_collate
from prosodia.training import MelodyAudioContrastiveModel, grouped_info_nce_loss


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
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def parse_recall_k(value: str) -> list[int]:
    recall_k = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not recall_k or any(k <= 0 for k in recall_k):
        raise argparse.ArgumentTypeError("--recall-k must contain positive integers")
    return recall_k


def checkpoint_arg(checkpoint_args: dict[str, Any], name: str, default: Any) -> Any:
    value = checkpoint_args.get(name, default)
    return default if value is None else value


def build_model(checkpoint_args: dict[str, Any]) -> MelodyAudioContrastiveModel:
    return MelodyAudioContrastiveModel(
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


def positive_ranks(logits: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
    valid_logits = logits.masked_fill(~candidate_mask.to(dtype=torch.bool), torch.finfo(logits.dtype).min)
    positive_scores = valid_logits[:, 0]
    return (valid_logits > positive_scores[:, None]).sum(dim=-1) + 1


@torch.no_grad()
def evaluate_retrieval(
    model: MelodyAudioContrastiveModel,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
    recall_k: list[int],
    use_amp: bool,
    collect_predictions: bool,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    total_rank = 0.0
    total_reciprocal_rank = 0.0
    total_candidates = 0.0
    recall_hits = {k: 0 for k in recall_k}
    all_ranks: list[int] = []
    predictions: list[dict[str, Any]] = []

    for batch in loader:
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
                temperature=temperature,
            )

        ranks = positive_ranks(logits, batch["candidate_mask"])
        candidate_counts = batch["candidate_mask"].sum(dim=-1)
        batch_size = ranks.shape[0]

        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size
        total_rank += float(ranks.sum().detach().cpu())
        total_reciprocal_rank += float((1.0 / ranks.float()).sum().detach().cpu())
        total_candidates += float(candidate_counts.sum().detach().cpu())
        all_ranks.extend(int(rank) for rank in ranks.detach().cpu().tolist())
        for k in recall_k:
            recall_hits[k] += int((ranks <= k).sum().detach().cpu())

        if collect_predictions:
            predictions.extend(
                prediction_rows(
                    batch=batch,
                    logits=logits,
                    ranks=ranks,
                    candidate_counts=candidate_counts,
                )
            )

    if total_examples == 0:
        raise SystemExit("No evaluation examples found.")

    sorted_ranks = sorted(all_ranks)
    midpoint = len(sorted_ranks) // 2
    if len(sorted_ranks) % 2:
        median_rank = float(sorted_ranks[midpoint])
    else:
        median_rank = (sorted_ranks[midpoint - 1] + sorted_ranks[midpoint]) / 2.0

    metrics: dict[str, float] = {
        "loss": total_loss / total_examples,
        "top1_accuracy": recall_hits.get(1, 0) / total_examples,
        "mrr": total_reciprocal_rank / total_examples,
        "mean_rank": total_rank / total_examples,
        "median_rank": median_rank,
        "mean_candidates": total_candidates / total_examples,
        "num_examples": float(total_examples),
    }
    for k in recall_k:
        metrics[f"recall@{k}"] = recall_hits[k] / total_examples
    return metrics, predictions


def prediction_rows(
    batch: dict[str, Any],
    logits: torch.Tensor,
    ranks: torch.Tensor,
    candidate_counts: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    logits_cpu = logits.detach().cpu()
    ranks_cpu = ranks.detach().cpu().tolist()
    candidate_counts_cpu = candidate_counts.detach().cpu().tolist()
    candidate_mask_cpu = batch["candidate_mask"].detach().cpu()

    for batch_index, melody_sample_id in enumerate(batch["melody_sample_id"]):
        valid_count = int(candidate_counts_cpu[batch_index])
        valid_logits = logits_cpu[batch_index, :valid_count]
        order = torch.argsort(valid_logits, descending=True).tolist()
        top_index = int(order[0]) if order else -1
        rows.append(
            {
                "melody_sample_id": melody_sample_id,
                "rank": int(ranks_cpu[batch_index]),
                "num_candidates": valid_count,
                "positive_score": float(logits_cpu[batch_index, 0]),
                "top_score": float(logits_cpu[batch_index, top_index]) if top_index >= 0 else "",
                "top_candidate_index": top_index,
                "top_candidate_id": batch["candidate_ids"][batch_index][top_index]
                if top_index >= 0
                else "",
                "top_candidate_type": batch["candidate_types"][batch_index][top_index]
                if top_index >= 0
                else "",
                "positive_candidate_id": batch["candidate_ids"][batch_index][0],
                "correct_top1": bool(int(ranks_cpu[batch_index]) == 1),
                "valid_mask": " ".join(
                    "1" if bool(value) else "0" for value in candidate_mask_cpu[batch_index].tolist()
                ),
            }
        )
    return rows


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "melody_sample_id",
        "rank",
        "num_candidates",
        "positive_score",
        "top_score",
        "top_candidate_index",
        "top_candidate_id",
        "top_candidate_type",
        "positive_candidate_id",
        "correct_top1",
        "valid_mask",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate melody/audio contrastive retrieval.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/prepared/dali/segments_manifest.csv"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-negatives", type=int, default=4)
    parser.add_argument(
        "--min-negative-offset-seconds",
        type=float,
        default=None,
        help="Minimum same-song negative offset for segment manifests. Defaults to the segment length.",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--recall-k", type=parse_recall_k, default=parse_recall_k("1,2,3,5"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--predictions-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.checkpoint = resolve_user_path(args.checkpoint)
    args.manifest = resolve_user_path(args.manifest)
    if args.output_json is not None:
        args.output_json = resolve_user_path(args.output_json)
    if args.predictions_csv is not None:
        args.predictions_csv = resolve_user_path(args.predictions_csv)

    device = choose_device(args.device)
    use_amp = args.amp and device.type == "cuda"
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    temperature = args.temperature
    if temperature is None:
        temperature = checkpoint_arg(checkpoint_args, "temperature", 0.07)

    dataset = GroupedContrastiveDataset(
        manifest_path=args.manifest,
        split=args.split,
        max_negatives=args.max_negatives,
        min_negative_offset_seconds=args.min_negative_offset_seconds,
        seed=checkpoint_arg(checkpoint_args, "seed", 13),
    )
    if len(dataset) == 0:
        raise SystemExit(f"No evaluation examples found for split={args.split}: {args.manifest}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=grouped_contrastive_collate,
    )

    model = build_model(checkpoint_args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics, predictions = evaluate_retrieval(
        model=model,
        loader=loader,
        device=device,
        temperature=float(temperature),
        recall_k=args.recall_k,
        use_amp=use_amp,
        collect_predictions=args.predictions_csv is not None,
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "split": args.split,
        "max_negatives": args.max_negatives,
        "temperature": float(temperature),
        "metrics": metrics,
    }

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
    if args.predictions_csv is not None:
        write_predictions(args.predictions_csv, predictions)


if __name__ == "__main__":
    main()
