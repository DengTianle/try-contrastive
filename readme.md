Basic contrastive training framework, with transformer-based melody encoder and HuBERT (frozen CNN layer) as audio/waveform encoder.
Training uses variable-duration segments made from a configurable number of consecutive
DALI lyric-line occurrences (one line by default). Repeat groupings turn congruent
line or multi-line occurrences into positive variants.
I have used a variant based on the InfoNCE loss: L_{hard negatives} + L_{in-batch}, where hard negatives are other segments from the same song, and in-batch compares with other positive samples in the batch. 
Evaluation is against hard negatives (only). 

Training and validation accumulate metrics on the device and transfer them together
for progress updates every 50 batches. Set `--log-every-steps N` to change that
interval, or `--log-every-steps 0` for epoch results only. `--no-progress` also
avoids intermediate metric transfers. Epoch metrics always include every batch.

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

Use `--lines-per-segment` to prepare fixed-size consecutive line windows. The default
stride equals the number of lines, producing non-overlapping windows. For example:

```bash
# Two-line segments, with every constituent line capped at 5 seconds.
conda run -n try-contrastive python scripts/prepare_dali_dataset.py \
  --repeat-groupings data/repeat_groupings \
  --output-dir data/prepared/dali_seg2 \
  --lines-per-segment 2 \
  --max-line-seconds 5

# Four-line segments. The optional whole-window cap rejects the complete window;
# it never truncates a line.
conda run -n try-contrastive python scripts/prepare_dali_dataset.py \
  --repeat-groupings data/repeat_groupings \
  --output-dir data/prepared/dali_seg4 \
  --lines-per-segment 4 \
  --max-line-seconds 5 \
  --max-window-seconds 25
```

Set `--segment-stride-lines 1` for sliding overlapping windows. Multi-line preparation
always retains repeated lyric occurrences so windows remain temporally consecutive;
an ineligible constituent line breaks a run and is never bridged. Segment-level lyric
and melody classes are derived from the complete sequence of constituent line classes,
so repeat-aware positives must match the full window. Without repeat-grouping metadata,
identical full lyric sequences receive a shared lyric class and are protected from use
as negatives, but they are not promoted to positives without melody evidence.

To prepare only a specific set of songs, put one DALI id per line in a text
file and pass `--keep-file path/to/song_ids.txt`. Blank lines and `#` comments
are ignored.

Preparation discards an individual line longer than 10 seconds or containing fewer
than three overlapping notes by default. Override these gates with
`--max-line-seconds` and `--min-line-notes`; the older names
`--max-segment-seconds` and `--min-segment-notes` remain aliases. A maximum duration
of 0 disables that limit. `--max-window-seconds` independently caps the total span
of a multi-line window.

For songs covered by the grouping results, the manifest records segment-level lyric
and melody classes. Training never uses another member of the anchor's complete
melody class as a negative. By default it chooses a positive occurrence with different
lyrics when available, then an ordinary repeated occurrence, and finally the aligned
audio. In one-line mode, songs absent from a partial grouping file retain the older
behavior of keeping the first normalized lyric occurrence.

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

The audio tower uses the note boundaries from each audio candidate, including repeated
positives and same-song negatives. HuBERT frames whose centers fall inside a note are
mean-pooled into one projected note embedding; those note embeddings are then equally
mean-pooled into the normalized vector used by contrastive training. Notes are clipped
and shifted when a candidate window is cropped or extended. The audio encoder can also
return the projected note sequence and its mask with `return_note_embeddings=True` for
downstream note-level tasks.

Use `--audio-pooling mean` when training to restore the original audio encoding:
mean-pool valid HuBERT frames, apply the projection head once, then normalize.
This includes all valid frames, including frames outside annotated notes. It ignores
candidate note timings; the shared dataset still prepares them. The default,
`--audio-pooling note`, retains the two-stage note pooling described above.
`return_note_embeddings=True` is supported only in note mode.

New checkpoints save `audio_pooling` in their training arguments. Evaluation and
embedding diagnostics restore that setting automatically and include it in their
JSON reports. An explicit `--audio-pooling` that conflicts with the saved mode is
rejected. Older checkpoints without this metadata require an explicit choice on
`scripts/evaluate_contrastive.py` or `scripts/diagnose_embeddings.py`: use
`--audio-pooling mean` for checkpoints trained before two-stage pooling, or
`--audio-pooling note` for checkpoints trained with two-stage pooling. Both modes
have identical parameter names and shapes, so their mode cannot be inferred from
the weights. State dictionaries are still loaded strictly to catch unrelated
architecture mismatches. Direct `load_state_dict` calls alone cannot check pooling;
custom callers should use `build_contrastive_model_from_checkpoint_args`, passing
`audio_pooling` explicitly for legacy checkpoints.

Contrastive training defaults to `--candidate-window-policy match-positive`. Within
each retrieval set, every audio candidate is presented at the selected positive's
duration. A shorter negative is extended with real surrounding song context, while a
longer negative is cropped inside its prepared interval. Training randomizes the context
placement; validation and evaluation use the center deterministically. Windows that
would have to cross a known congruent/repeated positive occurrence are excluded. Use
`--candidate-window-policy segment` to preserve every complete prepared one- or
multi-line interval (`line` is retained as a legacy alias).
Training also selects a reproducible, epoch-specific positive variant and capped
same-song negative subset; validation and evaluation retain the epoch-zero selection.

Train a prepared multi-line manifest with the same training script:

```bash
conda run -n try-contrastive python scripts/train_contrastive.py \
  --manifest data/prepared/dali_seg2/segments_manifest.csv \
  --output-dir checkpoints/contrastive_seg2 \
  --candidate-window-policy segment \
  --batch-size 4
```

The model and collate path already pad variable note and waveform lengths. Reduce
`--batch-size` and/or `--max-negatives` for longer windows as needed; keep at least two
examples per GPU when relying on the local in-batch contrastive loss.

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
