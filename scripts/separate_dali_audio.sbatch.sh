#!/usr/bin/env bash
# Add your site's working H100 resource selection to the sbatch command.
#SBATCH --job-name=dali-vocals
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vocal-extraction

# Arguments: batch-folder root, final audio folder, temporary stem folder root.
batches=$(cd "${1:?Pass batch folder root}" && pwd)
mkdir -p "${2:?Pass final audio folder}" "${3:?Pass temporary stem folder root}"
output=$(cd "$2" && pwd)
scratch=$(cd "$3" && pwd)
batch=$(printf 'batch_%04d' "${SLURM_ARRAY_TASK_ID:-0}")
stems="$scratch/$batch"
mkdir -p "$stems"
model="${MODEL:-UVR-MDX-NET-Voc_FT.onnx}"

audio-separator "$batches/$batch" \
  --output_dir "$stems" --output_format FLAC \
  -m "$model" --single_stem Vocals \
  --model_file_dir "${MODEL_DIR:-$HOME/.cache/audio-separator-models}"

suffix="_(Vocals)_${model%.*}.flac"
shopt -s nullglob
files=("$stems/"*"$suffix")
(( ${#files[@]} > 0 )) || { echo "No matching vocal stems in $stems" >&2; exit 1; }
for f in "${files[@]}"; do
  name=${f##*/}
  id=${name%"$suffix"}
  ffmpeg -nostdin -v error -xerror -y -i "$f" \
    -ar 16000 -ac 1 -c:a flac -sample_fmt s16 "$output/$id.partial.flac"
  mv -- "$output/$id.partial.flac" "$output/$id.flac"
  rm -- "$f"
done
