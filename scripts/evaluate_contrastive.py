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
from prosodia.training import (
    MelodyAudioContrastiveModel,
    build_contrastive_model_from_checkpoint_args,
    checkpoint_arg,
    grouped_info_nce_loss,
)


SEGMENT_LENGTH_TOLERANCE_SECONDS = 1e-3


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


def positive_ranks(
    logits: torch.Tensor,
    candidate_mask: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    valid_logits = logits.masked_fill(~candidate_mask.to(dtype=torch.bool), torch.finfo(logits.dtype).min)
    positive_scores = valid_logits.gather(1, targets.to(logits.device, dtype=torch.long).unsqueeze(1)).squeeze(1)
    return (valid_logits >= positive_scores[:, None]).sum(dim=-1)


def row_segment_seconds(row: dict[str, Any]) -> float:
    if row.get("segment_seconds") not in (None, ""):
        return float(row["segment_seconds"])
    return float(row["end_seconds"]) - float(row["start_seconds"])


def segment_length_probe_for_example(
    candidate_rows: list[dict[str, Any]],
    target_index: int,
    scores: torch.Tensor,
    tolerance_seconds: float = SEGMENT_LENGTH_TOLERANCE_SECONDS,
) -> dict[str, float]:
    """Measure how well duration alone identifies the paired audio candidate."""
    if not candidate_rows:
        raise ValueError("Segment-length probe requires at least one candidate")
    if not 0 <= target_index < len(candidate_rows):
        raise ValueError("Segment-length probe target is outside the candidate list")

    lengths = torch.tensor(
        [row_segment_seconds(row) for row in candidate_rows],
        dtype=torch.float64,
    )
    scores = scores.detach().cpu().to(dtype=torch.float64)
    if scores.ndim != 1 or scores.shape[0] != lengths.shape[0]:
        raise ValueError("Segment-length probe scores must match the candidate rows")

    target_length = lengths[target_index]
    length_errors = (lengths - target_length).abs()
    minimum_error = length_errors.min()
    nearest_mask = length_errors <= minimum_error + tolerance_seconds
    matching_candidates = int(nearest_mask.sum().item())
    target_is_nearest = bool(nearest_mask[target_index])
    expected_top1 = (
        1.0 / matching_candidates if target_is_nearest and matching_candidates else 0.0
    )

    top_index = int(scores.argmax().item())
    model_top1_length_match = float(
        length_errors[top_index] <= tolerance_seconds
    )

    correlation = float("nan")
    if len(candidate_rows) >= 2:
        proximity = -length_errors
        centered_scores = scores - scores.mean()
        centered_proximity = proximity - proximity.mean()
        denominator = torch.sqrt(
            centered_scores.square().sum() * centered_proximity.square().sum()
        )
        if float(denominator) > 0.0:
            correlation = float(
                (centered_scores * centered_proximity).sum() / denominator
            )

    return {
        "expected_top1_accuracy": expected_top1,
        "chance_top1_accuracy": 1.0 / len(candidate_rows),
        "unique_match": float(matching_candidates == 1),
        "matching_candidates": float(matching_candidates),
        "model_top1_length_match": model_top1_length_match,
        "score_proximity_correlation": correlation,
    }


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
    total_top1 = 0
    recall_hits = {k: 0 for k in recall_k}
    all_ranks: list[int] = []
    predictions: list[dict[str, Any]] = []
    length_probe_expected_top1 = 0.0
    length_probe_chance_top1 = 0.0
    length_probe_unique_matches = 0.0
    length_probe_matching_candidates = 0.0
    length_probe_model_top1_matches = 0.0
    length_probe_correlations: list[float] = []

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
                targets=batch["target"],
                temperature=temperature,
            )

        ranks = positive_ranks(logits, batch["candidate_mask"], batch["target"])
        candidate_counts = batch["candidate_mask"].sum(dim=-1)
        batch_size = ranks.shape[0]
        top_indices = logits.argmax(dim=-1)
        targets = batch["target"].to(device=top_indices.device, dtype=torch.long)

        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size
        total_top1 += int((top_indices == targets).sum().detach().cpu())
        total_rank += float(ranks.sum().detach().cpu())
        total_reciprocal_rank += float((1.0 / ranks.float()).sum().detach().cpu())
        total_candidates += float(candidate_counts.sum().detach().cpu())
        all_ranks.extend(int(rank) for rank in ranks.detach().cpu().tolist())
        for k in recall_k:
            recall_hits[k] += int((ranks <= k).sum().detach().cpu())

        logits_cpu = logits.detach().cpu()
        targets_cpu = batch["target"].detach().cpu().tolist()
        candidate_counts_cpu = candidate_counts.detach().cpu().tolist()
        for batch_index, target_index in enumerate(targets_cpu):
            valid_count = int(candidate_counts_cpu[batch_index])
            probe = segment_length_probe_for_example(
                candidate_rows=batch["metadata"][batch_index][:valid_count],
                target_index=int(target_index),
                scores=logits_cpu[batch_index, :valid_count],
            )
            length_probe_expected_top1 += probe["expected_top1_accuracy"]
            length_probe_chance_top1 += probe["chance_top1_accuracy"]
            length_probe_unique_matches += probe["unique_match"]
            length_probe_matching_candidates += probe["matching_candidates"]
            length_probe_model_top1_matches += probe["model_top1_length_match"]
            correlation = probe["score_proximity_correlation"]
            if correlation == correlation:
                length_probe_correlations.append(correlation)

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
        "top1_accuracy": total_top1 / total_examples,
        "mrr": total_reciprocal_rank / total_examples,
        "mean_rank": total_rank / total_examples,
        "median_rank": median_rank,
        "mean_candidates": total_candidates / total_examples,
        "num_examples": float(total_examples),
        "segment_length_probe_expected_top1_accuracy": (
            length_probe_expected_top1 / total_examples
        ),
        "segment_length_probe_chance_top1_accuracy": (
            length_probe_chance_top1 / total_examples
        ),
        "segment_length_probe_unique_match_rate": (
            length_probe_unique_matches / total_examples
        ),
        "segment_length_probe_mean_matching_candidates": (
            length_probe_matching_candidates / total_examples
        ),
        "segment_length_probe_model_top1_match_rate": (
            length_probe_model_top1_matches / total_examples
        ),
        "segment_length_probe_score_proximity_correlation": (
            sum(length_probe_correlations) / len(length_probe_correlations)
            if length_probe_correlations
            else float("nan")
        ),
        "segment_length_probe_correlation_examples": float(
            len(length_probe_correlations)
        ),
    }
    for k in recall_k:
        metrics[f"recall@{k}"] = (
            total_top1 / total_examples if k == 1 else recall_hits[k] / total_examples
        )
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
    targets_cpu = batch["target"].detach().cpu().tolist()
    candidate_counts_cpu = candidate_counts.detach().cpu().tolist()
    candidate_mask_cpu = batch["candidate_mask"].detach().cpu()

    for batch_index, melody_sample_id in enumerate(batch["melody_sample_id"]):
        valid_count = int(candidate_counts_cpu[batch_index])
        valid_logits = logits_cpu[batch_index, :valid_count]
        target_index = int(targets_cpu[batch_index])
        order = torch.argsort(valid_logits, descending=True).tolist()
        top_index = int(order[0]) if order else -1
        top_row = batch["metadata"][batch_index][top_index] if top_index >= 0 else {}
        positive_row = batch["metadata"][batch_index][target_index]
        top_start = float(top_row["start_seconds"]) if top_row else ""
        positive_start = float(positive_row["start_seconds"])
        top_segment_seconds = row_segment_seconds(top_row) if top_row else ""
        positive_segment_seconds = row_segment_seconds(positive_row)
        rows.append(
            {
                "melody_sample_id": melody_sample_id,
                "melody_start_seconds": positive_start,
                "melody_end_seconds": float(positive_row["end_seconds"]),
                "rank": int(ranks_cpu[batch_index]),
                "num_candidates": valid_count,
                "positive_score": float(logits_cpu[batch_index, target_index]),
                "top_score": float(logits_cpu[batch_index, top_index]) if top_index >= 0 else "",
                "top_score_margin": float(
                    logits_cpu[batch_index, top_index] - logits_cpu[batch_index, target_index]
                )
                if top_index >= 0
                else "",
                "top_candidate_index": top_index,
                "top_candidate_id": batch["candidate_ids"][batch_index][top_index]
                if top_index >= 0
                else "",
                "top_candidate_sample_id": top_row.get("sample_id", ""),
                "top_candidate_start_seconds": top_start,
                "top_candidate_end_seconds": float(top_row["end_seconds"]) if top_row else "",
                "top_candidate_offset_seconds": top_start - positive_start if top_row else "",
                "top_candidate_segment_seconds": top_segment_seconds,
                "top_candidate_length_error_seconds": (
                    abs(float(top_segment_seconds) - positive_segment_seconds)
                    if top_row
                    else ""
                ),
                "top_candidate_type": batch["candidate_types"][batch_index][top_index]
                if top_index >= 0
                else "",
                "positive_candidate_id": batch["candidate_ids"][batch_index][target_index],
                "positive_candidate_sample_id": positive_row["sample_id"],
                "positive_candidate_start_seconds": positive_start,
                "positive_candidate_end_seconds": float(positive_row["end_seconds"]),
                "positive_candidate_segment_seconds": positive_segment_seconds,
                "correct_top1": bool(top_index == target_index),
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
        "melody_start_seconds",
        "melody_end_seconds",
        "rank",
        "num_candidates",
        "positive_score",
        "top_score",
        "top_score_margin",
        "top_candidate_index",
        "top_candidate_id",
        "top_candidate_sample_id",
        "top_candidate_start_seconds",
        "top_candidate_end_seconds",
        "top_candidate_offset_seconds",
        "top_candidate_segment_seconds",
        "top_candidate_length_error_seconds",
        "top_candidate_type",
        "positive_candidate_id",
        "positive_candidate_sample_id",
        "positive_candidate_start_seconds",
        "positive_candidate_end_seconds",
        "positive_candidate_segment_seconds",
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
        help=(
            "Optional minimum difference between same-song segment start times. "
            "By default, all non-overlapping DALI line segments are eligible negatives."
        ),
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

    model = build_contrastive_model_from_checkpoint_args(checkpoint_args).to(device)
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
        "segment_length_probe_tolerance_seconds": SEGMENT_LENGTH_TOLERANCE_SECONDS,
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
