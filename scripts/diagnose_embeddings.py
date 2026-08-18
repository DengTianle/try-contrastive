from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prosodia.datasets import GroupedContrastiveDataset, grouped_contrastive_collate
from prosodia.training import (
    build_contrastive_model_from_checkpoint_args,
    checkpoint_arg,
    positive_audio_embeddings,
    sanitize_json_value,
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
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def finite_mean_and_std(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(finite)), float(np.std(finite))


def finite_mean_midi_semitones(
    midi_pitches: torch.Tensor,
    attention_mask: torch.Tensor,
) -> list[float]:
    values: list[float] = []
    for sample_pitches, sample_mask in zip(
        midi_pitches.detach().cpu(),
        attention_mask.detach().cpu(),
    ):
        valid = sample_mask.to(dtype=torch.bool)
        if not bool(valid.any()):
            values.append(float("nan"))
            continue
        values.append(float(sample_pitches[valid].float().mean() - 69.0))
    return values


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> dict[str, Any]:
    model.eval()
    melody_embeddings: list[torch.Tensor] = []
    audio_embeddings: list[torch.Tensor] = []
    sample_ids: list[str] = []
    song_ids: list[str] = []
    starts_seconds: list[float] = []
    voiced_ratios: list[float] = []
    pitch_semitones: list[float] = []
    note_counts: list[float] = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            batch_melody_embeddings, candidate_audio_embeddings = model(
                melody_features=batch["melody_features"],
                melody_attention_mask=batch["melody_attention_mask"],
                candidate_input_values=batch["candidate_input_values"],
                candidate_audio_attention_mask=batch["candidate_audio_attention_mask"],
            )
            batch_audio_embeddings = positive_audio_embeddings(
                candidate_audio_embeddings=candidate_audio_embeddings,
                targets=batch["target"],
            )

        melody_embeddings.append(batch_melody_embeddings.detach().cpu())
        audio_embeddings.append(batch_audio_embeddings.detach().cpu())
        pitch_semitones.extend(
            finite_mean_midi_semitones(
                batch["melody_midi_pitches"],
                batch["melody_attention_mask"],
            )
        )

        for anchor_row in batch["anchor_metadata"]:
            sample_ids.append(str(anchor_row["sample_id"]))
            song_ids.append(str(anchor_row["dali_id"]))
            starts_seconds.append(safe_float(anchor_row.get("start_seconds")))
            voiced_ratios.append(safe_float(anchor_row.get("voiced_ratio")))
            note_counts.append(safe_float(anchor_row.get("note_count")))

    if not melody_embeddings:
        raise SystemExit("No examples found for diagnostics.")

    return {
        "melody_embeddings": torch.cat(melody_embeddings, dim=0).numpy(),
        "audio_embeddings": torch.cat(audio_embeddings, dim=0).numpy(),
        "sample_ids": sample_ids,
        "song_ids": song_ids,
        "start_seconds": starts_seconds,
        "voiced_ratio": voiced_ratios,
        "pitch_semitones": pitch_semitones,
        "note_count": note_counts,
    }


def pairwise_cosine_summary(embeddings: np.ndarray) -> dict[str, float]:
    if embeddings.shape[0] < 2:
        return {
            "mean_pairwise_cosine": float("nan"),
            "std_pairwise_cosine": float("nan"),
        }
    if embeddings.shape[0] > 2048:
        sample_indices = np.linspace(0, embeddings.shape[0] - 1, num=2048, dtype=np.int64)
        embeddings = embeddings[sample_indices]
    norms = np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    normalized = embeddings / norms
    similarities = normalized @ normalized.T
    off_diagonal = similarities[~np.eye(similarities.shape[0], dtype=bool)]
    return {
        "mean_pairwise_cosine": float(np.mean(off_diagonal)),
        "std_pairwise_cosine": float(np.std(off_diagonal)),
    }


def embedding_health(prefix: str, embeddings: np.ndarray) -> dict[str, float]:
    metrics: dict[str, float] = {}
    n, dim = embeddings.shape
    norms = np.linalg.norm(embeddings, axis=1)
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    dim_std = centered.std(axis=0)

    metrics[f"{prefix}_num_examples"] = float(n)
    metrics[f"{prefix}_dim"] = float(dim)
    metrics[f"{prefix}_norm_mean"] = float(np.mean(norms))
    metrics[f"{prefix}_norm_std"] = float(np.std(norms))
    metrics[f"{prefix}_feature_std_mean"] = float(np.mean(dim_std))
    metrics[f"{prefix}_feature_std_min"] = float(np.min(dim_std))

    if n >= 2:
        singular_values = np.linalg.svd(centered, compute_uv=False)
        variances = (singular_values**2) / max(n - 1, 1)
        total_variance = float(np.sum(variances))
        if total_variance > 1e-12:
            probabilities = variances / total_variance
            entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
            metrics[f"{prefix}_effective_rank"] = float(math.exp(entropy))
            metrics[f"{prefix}_participation_ratio"] = float(
                total_variance**2 / np.sum(variances**2)
            )
            metrics[f"{prefix}_top_singular_variance_fraction"] = float(
                variances[0] / total_variance
            )
        else:
            metrics[f"{prefix}_effective_rank"] = 0.0
            metrics[f"{prefix}_participation_ratio"] = 0.0
            metrics[f"{prefix}_top_singular_variance_fraction"] = float("nan")
    metrics.update(
        {
            f"{prefix}_{key}": value
            for key, value in pairwise_cosine_summary(embeddings).items()
        }
    )
    return metrics


def different_song_negative_indices(song_ids: list[str], seed: int) -> np.ndarray:
    rng = random.Random(seed)
    indices = list(range(len(song_ids)))
    chosen: list[int] = []
    for index, song_id in enumerate(song_ids):
        candidates = [candidate for candidate in indices if song_ids[candidate] != song_id]
        if not candidates:
            candidates = [candidate for candidate in indices if candidate != index]
        chosen.append(rng.choice(candidates) if candidates else index)
    return np.asarray(chosen, dtype=np.int64)


def modality_alignment_metrics(
    melody_embeddings: np.ndarray,
    audio_embeddings: np.ndarray,
    song_ids: list[str],
    seed: int,
) -> dict[str, float]:
    positive_cosines = np.sum(melody_embeddings * audio_embeddings, axis=1)
    negative_indices = different_song_negative_indices(song_ids, seed=seed)
    negative_cosines = np.sum(melody_embeddings * audio_embeddings[negative_indices], axis=1)

    labels = np.concatenate(
        [np.ones_like(positive_cosines, dtype=np.int64), np.zeros_like(negative_cosines, dtype=np.int64)]
    )
    scores = np.concatenate([positive_cosines, negative_cosines])
    auc = float("nan")
    if len(np.unique(labels)) == 2 and len(np.unique(scores)) > 1:
        auc = float(roc_auc_score(labels, scores))

    melody_centroid = melody_embeddings.mean(axis=0)
    audio_centroid = audio_embeddings.mean(axis=0)
    centroid_cosine = float(
        np.dot(melody_centroid, audio_centroid)
        / (
            max(np.linalg.norm(melody_centroid), 1e-12)
            * max(np.linalg.norm(audio_centroid), 1e-12)
        )
    )

    return {
        "positive_cosine_mean": float(np.mean(positive_cosines)),
        "positive_cosine_std": float(np.std(positive_cosines)),
        "cross_song_negative_cosine_mean": float(np.mean(negative_cosines)),
        "alignment_margin_mean": float(np.mean(positive_cosines - negative_cosines)),
        "positive_vs_cross_song_auc": auc,
        "modality_centroid_cosine": centroid_cosine,
        "modality_centroid_l2_distance": float(np.linalg.norm(melody_centroid - audio_centroid)),
    }


def modality_classifier_metrics(
    melody_embeddings: np.ndarray,
    audio_embeddings: np.ndarray,
    seed: int,
    test_size: float,
) -> dict[str, float]:
    features = np.vstack([melody_embeddings, audio_embeddings])
    labels = np.concatenate(
        [np.zeros(melody_embeddings.shape[0], dtype=np.int64), np.ones(audio_embeddings.shape[0], dtype=np.int64)]
    )
    if len(labels) < 8:
        return {"modality_probe_accuracy": float("nan")}

    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
    )
    probe.fit(train_x, train_y)
    predictions = probe.predict(test_x)
    return {
        "modality_probe_accuracy": float(accuracy_score(test_y, predictions)),
        "modality_probe_chance": 0.5,
    }


def grouped_probe_split(
    labels: np.ndarray,
    group_ids: list[str],
    seed: int,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    valid_indices = np.flatnonzero(np.isfinite(labels))
    valid_groups = [group_ids[index] for index in valid_indices]
    unique_groups = sorted(set(valid_groups))
    if len(unique_groups) >= 3:
        rng = random.Random(seed)
        shuffled_groups = list(unique_groups)
        rng.shuffle(shuffled_groups)
        num_test_groups = max(1, int(round(len(shuffled_groups) * test_size)))
        test_groups = set(shuffled_groups[:num_test_groups])
        test_mask = np.asarray([group in test_groups for group in valid_groups], dtype=bool)
        train_indices = valid_indices[~test_mask]
        test_indices = valid_indices[test_mask]
        if len(train_indices) >= 4 and len(test_indices) >= 2:
            return train_indices, test_indices, "grouped_by_song"

    train_indices, test_indices = train_test_split(
        valid_indices,
        test_size=test_size,
        random_state=seed,
    )
    return np.asarray(train_indices), np.asarray(test_indices), "random_examples"


def pitch_probe_for_embeddings(
    prefix: str,
    embeddings: np.ndarray,
    pitch_semitones: np.ndarray,
    song_ids: list[str],
    seed: int,
    test_size: float,
) -> dict[str, float | str]:
    if np.isfinite(pitch_semitones).sum() < 12:
        return {
            f"{prefix}_pitch_probe_r2": float("nan"),
            f"{prefix}_pitch_probe_mae_semitones": float("nan"),
            f"{prefix}_pitch_probe_baseline_mae_semitones": float("nan"),
            f"{prefix}_pitch_probe_split": "too_few_examples",
        }

    train_indices, test_indices, split_name = grouped_probe_split(
        labels=pitch_semitones,
        group_ids=song_ids,
        seed=seed,
        test_size=test_size,
    )
    probe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    probe.fit(embeddings[train_indices], pitch_semitones[train_indices])
    predictions = probe.predict(embeddings[test_indices])
    baseline = np.full_like(predictions, np.median(pitch_semitones[train_indices]), dtype=np.float64)

    return {
        f"{prefix}_pitch_probe_r2": float(r2_score(pitch_semitones[test_indices], predictions)),
        f"{prefix}_pitch_probe_mae_semitones": float(
            mean_absolute_error(pitch_semitones[test_indices], predictions)
        ),
        f"{prefix}_pitch_probe_baseline_mae_semitones": float(
            mean_absolute_error(pitch_semitones[test_indices], baseline)
        ),
        f"{prefix}_pitch_probe_train_examples": float(len(train_indices)),
        f"{prefix}_pitch_probe_test_examples": float(len(test_indices)),
        f"{prefix}_pitch_probe_split": split_name,
    }


def note_count_probe_for_embeddings(
    embeddings: np.ndarray,
    note_counts: np.ndarray,
    song_ids: list[str],
    seed: int,
    test_size: float,
) -> dict[str, float | str]:
    metric_prefix = "melody_note_count_probe"
    finite_count = int(np.isfinite(note_counts).sum())
    if finite_count < 12:
        return {
            f"{metric_prefix}_r2": float("nan"),
            f"{metric_prefix}_mae_notes": float("nan"),
            f"{metric_prefix}_baseline_mae_notes": float("nan"),
            f"{metric_prefix}_split": (
                "missing_labels" if finite_count == 0 else "too_few_examples"
            ),
        }

    train_indices, test_indices, split_name = grouped_probe_split(
        labels=note_counts,
        group_ids=song_ids,
        seed=seed,
        test_size=test_size,
    )
    probe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    probe.fit(embeddings[train_indices], note_counts[train_indices])
    predictions = probe.predict(embeddings[test_indices])
    baseline = np.full_like(
        predictions,
        np.median(note_counts[train_indices]),
        dtype=np.float64,
    )

    return {
        f"{metric_prefix}_r2": float(r2_score(note_counts[test_indices], predictions)),
        f"{metric_prefix}_mae_notes": float(
            mean_absolute_error(note_counts[test_indices], predictions)
        ),
        f"{metric_prefix}_baseline_mae_notes": float(
            mean_absolute_error(note_counts[test_indices], baseline)
        ),
        f"{metric_prefix}_train_examples": float(len(train_indices)),
        f"{metric_prefix}_test_examples": float(len(test_indices)),
        f"{metric_prefix}_split": split_name,
    }


def run_diagnostics(
    extracted: dict[str, Any],
    seed: int,
    probe_test_size: float,
) -> dict[str, Any]:
    melody_embeddings = extracted["melody_embeddings"]
    audio_embeddings = extracted["audio_embeddings"]
    song_ids = extracted["song_ids"]
    pitch_semitones = np.asarray(extracted["pitch_semitones"], dtype=np.float64)
    note_counts = np.asarray(extracted["note_count"], dtype=np.float64)
    pitch_mean, pitch_std = finite_mean_and_std(pitch_semitones)
    note_count_mean, note_count_std = finite_mean_and_std(note_counts)

    metrics: dict[str, Any] = {
        "num_examples": float(melody_embeddings.shape[0]),
        "num_songs": float(len(set(song_ids))),
        "pitch_label_mean_semitones_from_a4": pitch_mean,
        "pitch_label_std_semitones": pitch_std,
        "note_count_label_mean": note_count_mean,
        "note_count_label_std": note_count_std,
        "note_count_label_num_examples": float(np.isfinite(note_counts).sum()),
    }
    metrics.update(embedding_health("melody", melody_embeddings))
    metrics.update(embedding_health("audio", audio_embeddings))
    metrics.update(
        modality_alignment_metrics(
            melody_embeddings=melody_embeddings,
            audio_embeddings=audio_embeddings,
            song_ids=song_ids,
            seed=seed,
        )
    )
    metrics.update(
        modality_classifier_metrics(
            melody_embeddings=melody_embeddings,
            audio_embeddings=audio_embeddings,
            seed=seed,
            test_size=probe_test_size,
        )
    )
    metrics.update(
        pitch_probe_for_embeddings(
            prefix="melody",
            embeddings=melody_embeddings,
            pitch_semitones=pitch_semitones,
            song_ids=song_ids,
            seed=seed,
            test_size=probe_test_size,
        )
    )
    metrics.update(
        pitch_probe_for_embeddings(
            prefix="audio",
            embeddings=audio_embeddings,
            pitch_semitones=pitch_semitones,
            song_ids=song_ids,
            seed=seed,
            test_size=probe_test_size,
        )
    )
    metrics.update(
        note_count_probe_for_embeddings(
            embeddings=melody_embeddings,
            note_counts=note_counts,
            song_ids=song_ids,
            seed=seed,
            test_size=probe_test_size,
        )
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose melody/audio embedding collapse, modality gap, absolute-pitch leakage, "
            "and note-count information."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/prepared/dali/segments_manifest.csv"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=0, help="0 means use the full split.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--probe-test-size", type=float, default=0.25)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.checkpoint = resolve_user_path(args.checkpoint)
    args.manifest = resolve_user_path(args.manifest)
    if args.output_json is not None:
        args.output_json = resolve_user_path(args.output_json)
    if not 0.05 <= args.probe_test_size <= 0.8:
        raise SystemExit("--probe-test-size must be between 0.05 and 0.8")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    seed = int(args.seed if args.seed is not None else checkpoint_arg(checkpoint_args, "seed", 13))

    dataset = GroupedContrastiveDataset(
        manifest_path=args.manifest,
        split=args.split,
        max_negatives=0,
        positive_variant_policy=str(
            checkpoint_arg(checkpoint_args, "positive_variant_policy", "self")
        ),
        candidate_window_policy=str(
            checkpoint_arg(checkpoint_args, "candidate_window_policy", "line")
        ),
        randomize_candidate_windows=False,
        seed=seed,
    )
    if len(dataset) == 0:
        raise SystemExit(f"No diagnostic examples found for split={args.split}: {args.manifest}")
    if args.max_examples < 0:
        raise SystemExit("--max-examples must be non-negative")
    if args.max_examples > 0:
        dataset = Subset(dataset, range(min(args.max_examples, len(dataset))))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=grouped_contrastive_collate,
    )

    device = choose_device(args.device)
    use_amp = args.amp and device.type == "cuda"
    model = build_contrastive_model_from_checkpoint_args(checkpoint_args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    extracted = extract_embeddings(
        model=model,
        loader=loader,
        device=device,
        use_amp=use_amp,
    )
    metrics = run_diagnostics(
        extracted=extracted,
        seed=seed,
        probe_test_size=args.probe_test_size,
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "split": args.split,
        "metrics": metrics,
    }

    report = sanitize_json_value(report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)


if __name__ == "__main__":
    main()
