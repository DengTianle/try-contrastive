from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prosodia.training import DifferentSongBatchSampler


class ManifestPlanningDataset:
    """Lightweight grouped dataset containing only fields needed by the sampler."""

    def __init__(self, songs: Sequence[str]) -> None:
        self.groups = [{"anchor": {"dali_id": song}} for song in songs]

    def __len__(self) -> int:
        return len(self.groups)


def load_song_ids(manifest_path: Path, split: str) -> list[str]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"dali_id", "split"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Manifest is missing required columns: {missing}")
        return [row["dali_id"] for row in reader if row["split"] == split]


def plan_summary(
    *,
    manifest_path: Path,
    split: str,
    batch_size: int,
    seed: int,
    epoch: int,
    drop_incomplete_batches: bool,
) -> dict[str, Any]:
    song_ids = load_song_ids(manifest_path, split)
    if not song_ids:
        raise ValueError(f"No examples found for split={split!r}")

    dataset = ManifestPlanningDataset(song_ids)
    sampler = DifferentSongBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        seed=seed,
        drop_last=drop_incomplete_batches,
    )
    sampler.set_epoch(epoch)
    batches = list(sampler)

    batch_size_counts = Counter(map(len, batches))
    song_size_counts = Counter(song_ids)
    scheduled_examples = sum(map(len, batches))
    total_capacity = len(batches) * batch_size
    return {
        "manifest": str(manifest_path),
        "split": split,
        "seed": seed,
        "epoch": epoch,
        "requested_batch_size": batch_size,
        "drop_incomplete_batches": drop_incomplete_batches,
        "input_examples": len(song_ids),
        "songs": len(song_size_counts),
        "smallest_song_segments": min(song_size_counts.values()),
        "median_song_segments": statistics.median(song_size_counts.values()),
        "mean_song_segments": statistics.mean(song_size_counts.values()),
        "largest_song_segments": max(song_size_counts.values()),
        "planned_batches": len(batches),
        "scheduled_examples": scheduled_examples,
        "dropped_examples": len(song_ids) - scheduled_examples,
        "capacity_utilization": scheduled_examples / max(total_capacity, 1),
        "batch_size_distribution": {
            str(size): count for size, count in sorted(batch_size_counts.items())
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report the exact DifferentSongBatchSampler plan for a manifest split."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/prepared/dali/segments_manifest.csv"),
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--drop-incomplete-batches", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve(strict=False)
    try:
        summary = plan_summary(
            manifest_path=manifest_path,
            split=args.train_split,
            batch_size=args.batch_size,
            seed=args.seed,
            epoch=args.epoch,
            drop_incomplete_batches=args.drop_incomplete_batches,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    print(f"manifest: {summary['manifest']}")
    print(f"split: {summary['split']}")
    print(
        "settings: "
        f"batch_size={summary['requested_batch_size']}, "
        f"seed={summary['seed']}, epoch={summary['epoch']}, "
        f"drop_incomplete={summary['drop_incomplete_batches']}"
    )
    print(
        "input: "
        f"{summary['input_examples']} examples across {summary['songs']} songs "
        f"(segments/song min={summary['smallest_song_segments']}, "
        f"median={summary['median_song_segments']}, "
        f"mean={summary['mean_song_segments']:.2f}, "
        f"max={summary['largest_song_segments']})"
    )
    print(
        "plan: "
        f"{summary['planned_batches']} batches, "
        f"{summary['scheduled_examples']} scheduled, "
        f"{summary['dropped_examples']} dropped, "
        f"{summary['capacity_utilization']:.2%} capacity utilization"
    )
    print("batch-size distribution:")
    distribution = summary["batch_size_distribution"]
    if not distribution:
        print("  (no batches)")
    for size, count in distribution.items():
        print(f"  size {size}: {count} batches")


if __name__ == "__main__":
    main()
