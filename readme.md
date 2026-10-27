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

CUDA mixed precision is optional: `--amp` uses FP16, while
`--amp --amp-dtype bf16` uses BF16 for training and validation on GPUs with native
BF16 support, including the H100. BF16 runs without gradient scaling; FP16 retains
gradient scaling. Without `--amp`, training uses FP32. For example:

```bash
conda run -n try-contrastive python scripts/train_contrastive.py \
  --amp --amp-dtype bf16
```

## Map YouTube IDs to DALI IDs

Put one YouTube video ID or URL per line in `youtube_ids.txt`, then run:

```bash
conda run -n try-contrastive python youtube_to_dali_ids.py \
  --youtube-ids-file youtube_ids.txt \
  --output matched_dali_ids.txt
```

This searches all local DALI v2 annotations. Add `--ids-file dali_v2_english_ids.txt`
to search only that subset, or `--dali-data-dir data/DALI_v1` to use DALI v1.
The script writes a unique DALI ID list, a CSV mapping, and an unmatched YouTube
ID list beside the output. It preserves every matching DALI entry when a video
has multiple matches. Matching uses the recorded video ID and does not check
current YouTube availability or identify other uploads of the same song.

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

To prevent downstream validation/evaluation songs from entering contrastive training,
export their **DALI IDs** (one per line) and pass each list with `--train-exclude-file`:

```bash
conda run -n try-contrastive python scripts/prepare_dali_dataset.py \
  --dali-data-dir data/DALI_v2/annot_tismir \
  --audio-dir data/audio \
  --output-dir data/prepared/dali_downstream_safe \
  --train-exclude-file downstream_val_dali_ids.txt \
  --train-exclude-file downstream_eval_dali_ids.txt
```

Listed songs are prepared normally and assigned only to contrastive **val/test**.
The splitter selects train songs from the remaining eligible IDs, keeping the usual
split sizes when possible. If there are too few train-eligible songs, train shrinks
and val/test grow in the requested val:test proportions. Every usable song is retained,
subject to the existing quality gates and optional track/segment limits. If every
song is train-excluded, preparation succeeds with val/test only.
`--exclude-file` is an alias for `--train-exclude-file` with the same train-only behavior.

`--keep-file` still selects the overall prepared pool, rather than assigning every
listed song to train. `metadata.json` records the train-exclusion files, their combined
ID list, the number of prepared songs barred from train, and per-split track counts.
Use a new output directory to keep this manifest separate from earlier preparations,
and point training at its `segments_manifest.csv`.

If the downstream export contains YouTube IDs, first convert it with
`youtube_to_dali_ids.py` as described above. Train restrictions only apply to matching
DALI entries; include every known DALI ID for a held-out song when multiple entries
or uploads represent that song.

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
fall entirely in an annotation gap move to the nearest note-centered placement,
keeping the same duration and staying inside the prepared candidate. Windows that
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

## Stress hard negatives

Enable `--stress-negatives` to add one synthetic audio candidate to a subset of
training retrieval sets. The candidate is made from the **selected positive
occurrence**, including repeated/retexted positives, with identical duration and
note timings. It reverses strong/weak word-level loudness ranks within lyric
lines. One gain is applied across each whole word, preserving relative levels
between its syllables and notes. This targets conventional sung prominence while
limiting changes to pronunciation; it does not explicitly change lexical stress
inside words.

Defaults leave **epochs 1–4 completely free of these negatives**. Generation starts
in epoch 5, and its loss weight increases linearly over epochs 5–6 to 0.25. It then
stays at 0.25 for **every remaining epoch**. Use
`--stress-negative-start-epoch 7` to leave six complete warm-up epochs instead.
`--stress-negative-ramp-epochs 0` applies the final weight immediately at the start.
There is no automatic end epoch. The feature is opt-in; ordinary training remains
available without word metadata.

New runs of `scripts/prepare_dali_dataset.py` automatically write small word
metadata files and a `stress_metadata_path` manifest column. For an existing
**note-level** manifest, add that metadata without reprocessing audio:

```bash
conda run -n try-contrastive python scripts/prepare_stress_metadata.py \
  --manifest data/prepared/dali_seg2/segments_manifest.csv \
  --dali-data-dir data/DALI_v1
```

This writes `stress_segments_manifest.csv` beside the source. The source manifest,
audio, and melody features are preserved. Use the same DALI version and optional
`--gt-file` as the original preparation: the backfill checks pitches and timings
before associating words with notes. Obsolete frame-level manifests still need
fresh note-level preparation. Missing or inconsistent word links are not guessed.

```bash
conda run -n try-contrastive python scripts/train_contrastive.py \
  --manifest data/prepared/dali_seg2/stress_segments_manifest.csv \
  --output-dir checkpoints/contrastive_stress \
  --stress-negatives \
  --stress-negative-start-epoch 5 \
  --stress-negative-ramp-epochs 2 \
  --stress-negative-weight 0.25 \
  --audio-noise-snr-db 35
```

`--stress-negative-prob` defaults to 0.5: half of active training anchors attempt an
edit. Generation runs in CPU data-loader workers with reproducible sample/epoch
randomness. Each edit uses robust 20 ms active-frame RMS statistics and a bounded
search for rank reversals. Default gain limits are ±6 dB with 20 ms smooth boundary
ramps. An accepted edit must change 60–80% of eligible notes by at least 1.5 dB,
cover at least 50% of eligible note duration, and reverse the mean loudness rank
correlation of edited lines to at most −0.25. Short/silent notes, overlapping or
truncated words, and lines without three usable words and 3 dB of contrast are
excluded. At least 80% of notes must have trustworthy word links. A bounded
three-word permutation fallback handles short lines where pair swaps cannot meet
the requested coverage. Segments still failing the gates produce no negative.

The edited clip is matched to the positive's RMS. Edits requiring peaks above
0.99 are rejected rather than clipped. Existing quiet-noise/background augmentation
is applied afterwards with the same settings for all candidate types. Adjust
`--stress-negative-max-gain-db`, `--stress-negative-min-changed-fraction`, and
`--stress-negative-max-changed-fraction` to study other strengths/coverage levels.

Only synthetic-negative denominator terms are multiplied by their scheduled
weight: the positive and ordinary negatives retain weight 1. The in-batch loss
still uses the original selected positives. Accuracy uses **raw, unweighted**
similarities. Training logs/checkpoints include acceptance rate/count, positive
versus synthetic accuracy, cosine score gap, synthetic selection rate, and
ordinary-candidate accuracy. The complete effective augmentation configuration is
saved in checkpoint arguments. Normal validation/evaluation uses natural candidates
only, so those metrics remain comparable across experiments.

Listen to accepted pairs and inspect rejection counts before committing a run:

```bash
conda run -n try-contrastive python scripts/preview_stress_negatives.py \
  --manifest data/prepared/dali_seg2/stress_segments_manifest.csv \
  --output-dir data/stress_preview --num-examples 8
```

The preview writes floating-point WAV pairs and `report.json` with measured
coverage/rank changes and rejection reasons. Synthetic discrimination alone does
not demonstrate better melody note embeddings; evaluate the downstream stress task
or a frozen note-level probe as well as ordinary retrieval.

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
