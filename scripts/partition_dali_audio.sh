#!/usr/bin/env bash
# Usage: bash scripts/partition_dali_audio.sh AUDIO_DIR BATCH_DIR [SIZE]
# Creates batch_0000, batch_0001, ... using symlinks, without copying audio.
set -euo pipefail
export LC_ALL=C
source_dir=$(cd "${1:?Pass audio directory}" && pwd)
destination=${2:?Pass a new or empty batch directory}
size=${3:-500}
[[ "$size" =~ ^[1-9][0-9]*$ ]] || { echo "Size must be a positive integer" >&2; exit 1; }
mkdir -p -- "$destination"
[[ -z "$(ls -A "$destination")" ]] || { echo "Batch directory is not empty: $destination" >&2; exit 1; }
n=0
for f in "$source_dir"/*; do
  [[ -f "$f" ]] || continue
  case "$f" in
    *.mp3|*.flac|*.wav|*.m4a|*.ogg|*.opus|*.aac) ;;
    *) continue ;;
  esac
  batch=$(printf '%s/batch_%04d' "$destination" "$((n / size))")
  mkdir -p "$batch"
  ln -s -- "$f" "$batch/${f##*/}"
  n=$((n + 1))
done
(( n > 0 )) || { echo "No audio files found" >&2; exit 1; }
echo "$n files in $(((n + size - 1) / size)) batches; originals unchanged."
