#!/usr/bin/env python3
"""Match YouTube IDs or URLs to DALI IDs using local annotation metadata.

Outputs unique DALI IDs, a CSV mapping (including unmatched videos), and a list
of unmatched YouTube IDs. No downloads or availability checks are performed.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from cluster_dl import load_dali, normalize_youtube_id, read_ids_file, resolve_user_path


def read_youtube_ids(path: Path) -> list[str]:
    """Read the first CSV column, ignoring comments, blanks, and an optional header."""
    ids = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if not row or not row[0].strip() or row[0].lstrip().startswith("#"):
                continue
            value = row[0].strip()
            if value.casefold() in {"youtube_id", "youtube", "youtube_url"}:
                continue
            youtube_id = normalize_youtube_id(value)
            if youtube_id:
                ids.append(youtube_id)
    return list(dict.fromkeys(ids))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--youtube-ids-file", type=Path, required=True,
                        help="Text/CSV file with one YouTube ID or URL per row (first column).")
    parser.add_argument("--dali-data-dir", type=Path, default=Path("data/DALI_v2/annot_tismir"),
                        help="DALI annotation directory (default: %(default)s).")
    parser.add_argument("--ids-file", type=Path,
                        help="Optional text/CSV list of DALI IDs to search (first column).")
    parser.add_argument("--output", type=Path, default=Path("matched_dali_ids.txt"),
                        help="Unique matched DALI IDs, one per line (default: %(default)s).")
    args = parser.parse_args()
    youtube_ids = read_youtube_ids(resolve_user_path(args.youtube_ids_file))
    if not youtube_ids:
        parser.error("The YouTube ID list is empty.")
    data_dir = resolve_user_path(args.dali_data_dir)
    if not data_dir.is_dir() or not any(data_dir.glob("*.gz")):
        parser.error(f"No DALI annotation .gz files found in: {data_dir}")
    keep_ids = read_ids_file(resolve_user_path(args.ids_file)) if args.ids_file else None
    # An empty DALI filter must match nothing; the DALI loader treats [] as all.
    dataset = {} if keep_ids == set() else load_dali(data_dir, gt_file=None, keep_ids=keep_ids)
    requested = set(youtube_ids)
    matches = defaultdict(list)
    for entry in dataset.values():
        dali_id = entry.info["id"]
        if keep_ids is not None and dali_id not in keep_ids:
            continue
        youtube_id = normalize_youtube_id(entry.info.get("audio", {}).get("url"))
        if youtube_id in requested:
            matches[youtube_id].append(dali_id)

    output = resolve_user_path(args.output)
    mapping_path = output.with_suffix(".csv")
    unmatched_path = output.with_name(f"{output.stem}_unmatched_youtube_ids.txt")
    if len({output, mapping_path, unmatched_path}) != 3:
        parser.error("--output must use a suffix other than .csv.")
    inputs = {resolve_user_path(args.youtube_ids_file)}
    if args.ids_file:
        inputs.add(resolve_user_path(args.ids_file))
    if inputs & {output, mapping_path, unmatched_path}:
        parser.error("Output paths must differ from input lists.")
    output.parent.mkdir(parents=True, exist_ok=True)
    dali_ids = sorted({dali_id for ids in matches.values() for dali_id in ids})
    unmatched = [youtube_id for youtube_id in youtube_ids if youtube_id not in matches]
    output.write_text("".join(f"{dali_id}\n" for dali_id in dali_ids), encoding="utf-8")
    unmatched_path.write_text("".join(f"{youtube_id}\n" for youtube_id in unmatched), encoding="utf-8")
    with mapping_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["youtube_id", "dali_id"])
        for youtube_id in youtube_ids:
            for dali_id in sorted(matches.get(youtube_id, [])) or [""]:
                writer.writerow([youtube_id, dali_id])
    print(f"Matched {len(youtube_ids) - len(unmatched)}/{len(youtube_ids)} YouTube IDs "
          f"to {len(dali_ids)} DALI entries (searched {len(dataset)} entries).")
    print(f"DALI IDs: {output}\nMapping: {mapping_path}\nUnmatched YouTube IDs: {unmatched_path}")


if __name__ == "__main__":
    main()
