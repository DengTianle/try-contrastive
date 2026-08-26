Basic contrastive training framework, with transformer-based melody encoder and HuBERT (frozen CNN layer) as audio/waveform encoder.
Training uses one variable-duration segment per DALI lyric-line occurrence. Repeat
groupings turn congruent occurrences into positive variants; without grouping metadata,
only the first occurrence of each normalized lyric line is retained.
I have used a variant based on the InfoNCE loss: L_{hard negatives} + L_{in-batch}, where hard negatives are other segments from the same song, and in-batch compares with other positive samples in the batch. 
Evaluation is against hard negatives (only). 

## Training batch planning

Training with the global in-batch loss plans each epoch so that a batch contains at
most one segment from a song. Inspect the exact batch-size distribution before
training with the same manifest, split, and batch size you intend to use:

```bash
conda run -n try-contrastive python scripts/plan_song_batches.py \
  --manifest data/prepared/dali/segments_manifest.csv \
  --train-split train \
  --batch-size 8
```

Pass `--drop-incomplete-batches` to both the planning script and
`scripts/train_contrastive.py` to retain only complete training batches. The planner
selects the maximum feasible number of full batches while keeping songs distinct.

## Multi-GPU training

Launch one process per GPU with `torchrun`; no additional training flag is needed:

```bash
conda run -n try-contrastive torchrun --standalone --nproc-per-node=4 \
  scripts/train_contrastive.py --batch-size 8
```

`--batch-size` is per GPU, so this example has an effective global batch size of 32.
Gradients are synchronized with DistributedDataParallel, while the global in-batch
loss and its negatives remain local to each GPU. Validation, logging, and checkpoint
writes run only on rank 0.

## Dataset preparation

Prepare line-aligned melody and audio segments:

```bash
conda run -n try-contrastive python scripts/find_dali_repeats.py build \
  --data-dir data/DALI_v1 \
  --output-dir data/repeat_groupings \
  --levels line

conda run -n try-contrastive python scripts/prepare_dali_dataset.py \
  --repeat-groupings data/repeat_groupings
```

To prepare only a specific set of songs, put one DALI id per line in a text
file and pass `--keep-file path/to/song_ids.txt`. Blank lines and `#` comments
are ignored.

Preparation discards line segments longer than 10 seconds or containing fewer
than three overlapping notes by default. Override these gates with
`--max-segment-seconds` and `--min-segment-notes`; a maximum duration of 0
disables the duration limit.

For songs covered by the grouping results, the manifest retains every line occurrence
and records its lyric and melody classes. Training never uses another member of the
anchor's melody class as a negative. By default it chooses a positive occurrence with
different lyrics when available, then an ordinary repeated occurrence, and finally
the aligned audio. Songs absent from a partial grouping file retain the older behavior
of keeping the first normalized lyric occurrence.

Each melody segment is represented as a note sequence using the 177-dimensional scheme
from [Wang et al.](https://arxiv.org/html/2508.00123): 129 dimensions for MIDI pitch
change relative to the first note (128-way magnitude plus sign), 24 for quantized
log-duration, and 24 for quantized log inter-onset interval. Duration and onset shift
are min-max normalized within each segment before quantization, matching the authors'
[reference implementation](https://github.com/changhongw/mlm). The manifest retains
`voiced_ratio` as note-coverage metadata, but it is not used to filter segments. Melody
notes and audio waveforms are padded per batch and masked during training, pretraining,
diagnostics, and evaluation. Prepared frame-level `.npz` files are incompatible; rerun
the preparation command after upgrading.

Contrastive training defaults to `--candidate-window-policy match-positive`. Within
each retrieval set, every audio candidate is presented at the selected positive's
duration. A shorter negative is extended with real surrounding song context, while a
longer negative is cropped inside its annotated line. Training randomizes the context
placement; validation and evaluation use the center deterministically. Windows that
would have to cross a known congruent/repeated positive occurrence are excluded. Use
`--candidate-window-policy line` for the original variable-line-duration behavior.
Training also selects a reproducible, epoch-specific positive variant and capped
same-song negative subset; validation and evaluation retain the epoch-zero selection.

## Evaluation

```bash
conda run -n try-contrastive python scripts/evaluate_contrastive.py \
  --checkpoint checkpoints/contrastive/best.pt
```

Evaluation includes a duration-only retrieval baseline. It ranks candidate audio by
the duration actually presented to the model, relative to the melody anchor, and
reports tie-aware top-1, MRR, mean rank, and recall. With `match-positive`, all
candidates in a set tie and this baseline should equal chance. The model top-1
duration-nearest rate and score/proximity correlation remain useful when evaluating
the older `line` policy. Override `--positive-variant-policy` to compare aligned and
congruent-repeat retrieval explicitly.

## Melody encoder pretraining

Pretrain the melody encoder by masking notes and classifying their pitch-change,
pitch-sign, duration, and onset-shift attributes before contrastive training:

```bash
conda run -n try-contrastive python scripts/pretrain_melody_encoder.py
```

Then initialize the contrastive melody tower from the pretraining checkpoint:

```bash
conda run -n try-contrastive python scripts/train_contrastive.py \
  --melody-pretrained-checkpoint checkpoints/melody_pretrain/best.pt
```

The pretraining checkpoint contains only the transferable melody encoder trunk. The
contrastive melody projection head is always initialized from scratch.

## Melody encoder diagnostics

The embedding diagnostic reports collapse and alignment metrics plus linear probes for
absolute pitch and the total number of annotated notes overlapping each segment. The
note-count probe requires a manifest produced by the current dataset preparation script.

```bash
conda run -n try-contrastive python scripts/diagnose_embeddings.py \
  --checkpoint checkpoints/contrastive/best.pt
```
