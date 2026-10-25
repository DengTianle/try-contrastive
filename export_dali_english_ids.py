#!/usr/bin/env python3
"""Export all English DALI v2 song IDs, one per line, without availability checks.

Uses cluster_dl's metadata.language filter (case-insensitive, whitespace-trimmed).
No filtering on note annotations, audio URLs, or the historical working flag.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cluster_dl import is_english, load_dali, resolve_user_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dali-data-dir",
        type=Path,
        default=Path("data/DALI_v2/annot_tismir"),
        help="Directory containing DALI v2 annotation .gz files (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dali_v2_english_ids.txt"),
        help="Output text file of DALI IDs (default: %(default)s).",
    )
    args = parser.parse_args()
    data_dir = resolve_user_path(args.dali_data_dir)
    output = resolve_user_path(args.output)
    if not data_dir.is_dir() or not any(data_dir.glob("*.gz")):
        raise SystemExit(f"No DALI annotation .gz files found in: {data_dir}")

    dataset = load_dali(data_dir, gt_file=None)
    ids = sorted({entry.info["id"] for entry in dataset.values() if is_english(entry)})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(f"{dali_id}\n" for dali_id in ids), encoding="utf-8")
    print(f"Wrote {len(ids)} English DALI IDs to: {output}")


if __name__ == "__main__":
    main()
