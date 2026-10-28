"""Export reproducible original/edited pairs and quality-gate diagnostics."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prosodia.datasets import GroupedContrastiveDataset, StressNegativeConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--epoch", type=int, default=5, help="One-based epoch; must be >= 5")
    parser.add_argument("--num-examples", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=200)
    parser.add_argument("--max-gain-db", type=float, default=6)
    parser.add_argument("--min-changed-fraction", type=float, default=0.6)
    parser.add_argument("--max-changed-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    if args.epoch < 5 or min(args.num_examples, args.max_attempts) < 1:
        parser.error("epoch must be >= 5 and example/attempt counts must be positive")
    config = replace(
        StressNegativeConfig(), probability=1, max_gain_db=args.max_gain_db,
        min_changed_fraction=args.min_changed_fraction, max_changed_fraction=args.max_changed_fraction,
    )
    dataset = GroupedContrastiveDataset(
        args.manifest, split=args.split, max_negatives=0,
        positive_variant_policy="retexted-first", candidate_window_policy="segment",
        stress_negative_config=config, seed=args.seed,
    )
    dataset.set_epoch(args.epoch - 1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reasons = Counter()
    examples = []
    for index in range(min(len(dataset), args.max_attempts)):
        item = dataset[index]
        diagnostics = item["stress_negative_diagnostics"]
        reasons[diagnostics["reason"]] += 1
        synthetic = item["candidate_stress_negative_mask"].nonzero().flatten()
        if not len(synthetic):
            continue
        paths = {}
        for name, candidate in (("original", int(item["target"])), ("stress_negative", int(synthetic[0]))):
            path = args.output_dir / f"{len(examples) + 1:02d}_{name}.wav"
            mask = item["candidate_audio_attention_mask"][candidate]
            sf.write(path, item["candidate_input_values"][candidate][mask].numpy(), dataset.audio_config.sample_rate, subtype="FLOAT")
            paths[name] = str(path.resolve())
        examples.append({"anchor": item["melody_sample_id"], "positive": item["metadata"][int(item["target"])]["sample_id"], **paths, **diagnostics})
        if len(examples) >= args.num_examples:
            break
    report = {"config": asdict(config), "epoch": args.epoch, "reasons": dict(reasons), "examples": examples}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
