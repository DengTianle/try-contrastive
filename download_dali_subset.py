#!/usr/bin/env python
"""Download a small DALI audio subset for local dataset preparation.

This script loads DALI annotations, chooses a small set of usable entries, and
downloads the corresponding YouTube audio into --audio-dir.
"""

from __future__ import annotations

import argparse
import gzip
import pickle
import random
import re
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac"}
YOUTUBE_WATCH_URL = "https://www.youtube.com/watch?v={youtube_id}"


def resolve_user_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return expanded.resolve(strict=False)


def audio_files_by_stem(audio_dir: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in audio_dir.rglob("*")
        if path.suffix.lower() in AUDIO_EXTENSIONS
    }


def existing_audio_path(audio_dir: Path, dali_id: str) -> Path | None:
    return audio_files_by_stem(audio_dir).get(dali_id)


def parse_ids(value: str | None) -> set[str] | None:
    if value is None:
        return None
    ids = {item.strip() for item in value.split(",") if item.strip()}
    return ids or None


def read_ids_file(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    ids = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.add(line.split(",")[0].strip())
    return ids


def merge_id_filters(*filters: set[str] | None) -> set[str] | None:
    merged = None
    for id_filter in filters:
        if id_filter is None:
            continue
        merged = set(id_filter) if merged is None else merged & id_filter
    return merged


def normalize_youtube_id(value: str | None) -> str:
    if value is None:
        return ""
    value = str(value).strip()
    if not value or value.lower() == "none":
        return ""

    parsed = urlparse(value)
    if parsed.netloc:
        if parsed.netloc.endswith("youtu.be"):
            return parsed.path.strip("/")
        query_id = parse_qs(parsed.query).get("v", [""])[0]
        if query_id:
            return query_id
        match = re.search(r"/(?:embed|shorts)/([^/?#]+)", parsed.path)
        if match:
            return match.group(1)
    return value


def load_dali(
    dali_data_dir: Path,
    gt_file: Path | None,
    keep_ids: set[str] | None = None,
) -> dict[str, Any]:
    try:
        import DALI as dali_code
    except ImportError as exc:
        raise SystemExit(
            "Could not import DALI. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    keep = sorted(keep_ids) if keep_ids else []
    if gt_file is None:
        return dali_code.get_the_DALI_dataset(str(dali_data_dir), skip=[], keep=keep)
    return dali_code.get_the_DALI_dataset(str(dali_data_dir), gt_file=str(gt_file), skip=[], keep=keep)


def read_ground_truth_ids(gt_file: Path) -> set[str]:
    with gzip.open(gt_file, "rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise SystemExit(f"Expected ground-truth dict, got {type(data).__name__}: {gt_file}")
    return set(data.keys())


def has_notes(entry: Any) -> bool:
    if entry.annotations.get("type") == "vertical":
        entry.vertical2horizontal()
    return bool(entry.annotations["annot"].get("notes", []))


def entry_to_info_row(entry: Any) -> list[Any] | None:
    dali_id = entry.info["id"]
    artist = entry.info.get("artist", "")
    title = entry.info.get("title", "")
    name = f"{artist} - {title}".strip(" -") or dali_id
    audio_info = entry.info.get("audio", {})
    youtube_id = normalize_youtube_id(audio_info.get("url"))
    if not youtube_id:
        return None
    working = bool(audio_info.get("working", True))
    if not working:
        return None
    return [dali_id, name, youtube_id, working]


def info_subset_from_annotations(
    dali_data: dict[str, Any],
    seed: int,
    keep_ids: set[str] | None,
) -> list[list[Any]]:
    rows = []
    entries = list(dali_data.values())
    entries.sort(key=lambda entry: entry.info["id"])
    random.Random(seed).shuffle(entries)

    for entry in entries:
        if keep_ids is not None and entry.info["id"] not in keep_ids:
            continue
        if not has_notes(entry):
            continue
        row = entry_to_info_row(entry)
        if row is not None:
            rows.append(row)
    return [["DALI_ID", "NAME", "YOUTUBE", "WORKING"], *rows]


def load_or_build_info(
    dali_data_dir: Path,
    gt_file: Path | None,
    dali_info_file: Path | None,
    seed: int,
    keep_ids: set[str] | None,
) -> list[list[Any]]:
    import DALI as dali_code

    if dali_info_file is not None:
        info = dali_code.get_info(str(dali_info_file))
        if info is None:
            raise SystemExit(f"Could not read DALI info file: {dali_info_file}")
        rows = list(info)
        header, body = rows[0], rows[1:]
        if keep_ids is not None:
            body = [row for row in body if row[0] in keep_ids]
        random.Random(seed).shuffle(body)
        return [header, *body]

    dali_data = load_dali(dali_data_dir, gt_file, keep_ids=keep_ids)
    return info_subset_from_annotations(dali_data, seed, keep_ids)


def count_available(rows: list[list[Any]], audio_dir: Path) -> int:
    existing = audio_files_by_stem(audio_dir)
    return sum(1 for row in rows if row[0] in existing)


def download_one_with_ytdlp(
    row: list[Any],
    audio_dir: Path,
    audio_format: str,
    ffmpeg_location: Path | None,
) -> Exception | None:
    try:
        import yt_dlp
    except ImportError as exc:
        raise SystemExit(
            "yt-dlp is not installed. Run: pip install -r requirements.txt\n"
            "The DALI package uses youtube-dl, which often fails on current YouTube pages."
        ) from exc

    dali_id = row[0]
    youtube_id = row[-2]
    ydl_opts: dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": str(audio_dir / f"{dali_id}.%(ext)s"),
        "noplaylist": True,
        "nocheckcertificate": True,
        "ignoreerrors": False,
    }
    if audio_format != "none":
        ydl_opts["postprocessors"] = [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": audio_format,
                "preferredquality": "192",
            }
        ]
    if ffmpeg_location is not None:
        ydl_opts["ffmpeg_location"] = str(ffmpeg_location)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([YOUTUBE_WATCH_URL.format(youtube_id=youtube_id)])
    except Exception as exc:
        return exc
    return None


def download_rows_with_dali(rows: list[list[Any]], all_info: list[list[Any]], audio_dir: Path) -> list[list[Any]]:
    import DALI as dali_code

    batch_ids = [row[0] for row in rows]
    return dali_code.get_audio(all_info, str(audio_dir), keep=batch_ids)


def download_rows_with_ytdlp(
    rows: list[list[Any]],
    audio_dir: Path,
    audio_format: str,
    ffmpeg_location: Path | None,
) -> list[list[Any]]:
    errors = []
    for row in rows:
        dali_id = row[0]
        if existing_audio_path(audio_dir, dali_id) is not None:
            print(f"Skipping existing audio: {dali_id}")
            continue
        print(f"Downloading {dali_id} from YouTube id {row[-2]}")
        error = download_one_with_ytdlp(row, audio_dir, audio_format, ffmpeg_location)
        if error is not None:
            print(error)
            errors.append([dali_id, row[-2], error])
    return errors


def converter_available(ffmpeg_location: Path | None) -> bool:
    if ffmpeg_location is not None:
        if ffmpeg_location.is_dir():
            return (ffmpeg_location / "ffmpeg").exists() and (ffmpeg_location / "ffprobe").exists()
        return ffmpeg_location.exists()
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def require_converter_if_needed(args: argparse.Namespace) -> None:
    if args.downloader != "yt-dlp" or args.audio_format == "none" or args.dry_run:
        return
    if converter_available(args.ffmpeg_location):
        return
    raise SystemExit(
        "ffmpeg and ffprobe are required to convert downloaded audio to "
        f"{args.audio_format!r}, but they were not found.\n"
        "Install them with one of:\n"
        "  conda install -c conda-forge ffmpeg\n"
        "  brew install ffmpeg\n"
        "or pass --ffmpeg-location /path/to/ffmpeg/bin"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dali-data-dir",
        type=Path,
        required=True,
        help="Directory containing DALI .gz annotations. Relative paths are resolved from the current directory.",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="Directory where downloaded audio will be stored.",
    )
    parser.add_argument("--gt-file", type=Path, default=None, help="Optional DALI ground-truth gzip file.")
    parser.add_argument(
        "--ground-truth-only",
        action="store_true",
        help="Use only DALI ids present in --gt-file. Useful for small aligned experiments.",
    )
    parser.add_argument(
        "--dali-info-file",
        type=Path,
        default=None,
        help="Optional DALI info .gz file. If omitted, the script builds the info table from annotations.",
    )
    parser.add_argument("--ids", type=str, default=None, help="Comma-separated DALI ids to download.")
    parser.add_argument("--ids-file", type=Path, default=None, help="Text/CSV file of DALI ids; first column is used.")
    parser.add_argument("--target-tracks", type=int, default=20)
    parser.add_argument("--candidate-multiplier", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--downloader", choices=["yt-dlp", "dali"], default="yt-dlp")
    parser.add_argument(
        "--audio-format",
        type=str,
        default="mp3",
        help="yt-dlp postprocessed audio format. Use 'none' to keep the downloaded container.",
    )
    parser.add_argument(
        "--ffmpeg-location",
        type=Path,
        default=None,
        help="Optional path to ffmpeg/ffprobe or the directory containing them.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print selected ids without downloading.")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dali_data_dir = resolve_user_path(args.dali_data_dir)
    args.audio_dir = resolve_user_path(args.audio_dir)
    if args.gt_file is not None:
        args.gt_file = resolve_user_path(args.gt_file)
    if args.dali_info_file is not None:
        args.dali_info_file = resolve_user_path(args.dali_info_file)
    if args.ids_file is not None:
        args.ids_file = resolve_user_path(args.ids_file)
    if args.ffmpeg_location is not None:
        args.ffmpeg_location = resolve_user_path(args.ffmpeg_location)

    args.audio_dir.mkdir(parents=True, exist_ok=True)
    if not args.dali_data_dir.is_dir():
        args.dali_data_dir.mkdir(parents=True, exist_ok=True)
        raise SystemExit(
            f"Created DALI annotation directory: {args.dali_data_dir}\n"
            "Put DALI annotation .gz files there, then rerun this downloader."
        )
    if not list(args.dali_data_dir.rglob("*.gz")):
        raise SystemExit(
            f"No DALI annotation .gz files found in: {args.dali_data_dir}\n"
            "Put DALI annotation files there, then rerun this downloader."
        )

    if args.ground_truth_only and args.gt_file is None:
        raise SystemExit("--ground-truth-only requires --gt-file")
    keep_ids = merge_id_filters(
        read_ground_truth_ids(args.gt_file) if args.ground_truth_only else None,
        parse_ids(args.ids),
        read_ids_file(args.ids_file),
    )

    dali_info = load_or_build_info(
        args.dali_data_dir,
        gt_file=args.gt_file,
        dali_info_file=args.dali_info_file,
        seed=args.seed,
        keep_ids=keep_ids,
    )
    if keep_ids is not None:
        candidate_rows = dali_info[1:]
    else:
        candidate_rows = dali_info[1 : 1 + args.target_tracks * args.candidate_multiplier]
    if not candidate_rows:
        raise SystemExit("No downloadable DALI candidates found. Check annotations and audio URLs.")

    target_tracks = min(args.target_tracks, len(candidate_rows)) if keep_ids is None else len(candidate_rows)
    available = count_available(candidate_rows, args.audio_dir)
    print(f"Audio directory: {args.audio_dir}")
    print(f"Candidate tracks: {len(candidate_rows)}")
    print(f"Already available: {available}")
    require_converter_if_needed(args)
    if args.dry_run:
        for row in candidate_rows:
            status = "exists" if existing_audio_path(args.audio_dir, row[0]) else "missing"
            print(f"{row[0]},{row[-2]},{status}")
        return

    errors = []
    for start in range(0, len(candidate_rows), args.batch_size):
        if available >= target_tracks:
            break

        batch = candidate_rows[start : start + args.batch_size]
        missing_rows = [row for row in batch if existing_audio_path(args.audio_dir, row[0]) is None]
        if not missing_rows:
            available = count_available(candidate_rows, args.audio_dir)
            continue

        print(f"Downloading batch: {', '.join(row[0] for row in missing_rows)}")
        if args.downloader == "yt-dlp":
            batch_errors = download_rows_with_ytdlp(
                missing_rows,
                args.audio_dir,
                args.audio_format,
                args.ffmpeg_location,
            )
        else:
            batch_errors = download_rows_with_dali(missing_rows, dali_info, args.audio_dir)
        errors.extend(batch_errors)
        available = count_available(candidate_rows, args.audio_dir)
        print(f"Available after batch: {available}/{target_tracks}")

    if errors:
        error_path = args.audio_dir / "download_errors.txt"
        with error_path.open("w", encoding="utf-8") as handle:
            for item in errors:
                handle.write(repr(item) + "\n")
        print(f"Wrote download errors to: {error_path}")

    if available < target_tracks:
        raise SystemExit(
            f"Only {available}/{target_tracks} requested audio files are available. "
            "Try increasing --candidate-multiplier or rerunning later."
        )

    print(f"Ready: found at least {target_tracks} usable audio files in {args.audio_dir}")


if __name__ == "__main__":
    main()
