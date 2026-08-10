Basic contrastive training framework, with transformer-based melody encoder and HuBERT (frozen CNN layer) as audio/waveform encoder.
Training uses one variable-duration segment per DALI lyric-line occurrence. Repeat
groupings turn congruent occurrences into positive variants; without grouping metadata,
only the first occurrence of each normalized lyric line is retained.
I have used a variant based on the InfoNCE loss: L_{hard negatives} + L_{in-batch}, where hard negatives are other segments from the same song, and in-batch compares with other positive samples in the batch. 
Evaluation is against hard negatives (only). 

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

For songs covered by the grouping results, the manifest retains every line occurrence
and records its lyric and melody classes. Training never uses another member of the
anchor's melody class as a negative. By default it chooses a positive occurrence with
different lyrics when available, then an ordinary repeated occurrence, and finally
the aligned audio. Songs absent from a partial grouping file retain the older behavior
of keeping the first normalized lyric occurrence.

The manifest retains `voiced_ratio` for diagnostics, but vocal ratio is no longer used
to filter segments. Melody frames and audio waveforms are padded per batch and masked
during training, pretraining, diagnostics, and evaluation.

Contrastive training defaults to `--candidate-window-policy match-positive`. Within
each retrieval set, every audio candidate is presented at the selected positive's
duration. A shorter negative is extended with real surrounding song context, while a
longer negative is cropped inside its annotated line. Training randomizes the context
placement; validation and evaluation use the center deterministically. Windows that
would have to cross a known congruent/repeated positive occurrence are excluded. Use
`--candidate-window-policy line` for the original variable-line-duration behavior.

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

Pretrain the melody encoder with a masked prosody objective before contrastive training:

```bash
conda run -n try-contrastive python scripts/pretrain_melody_encoder.py
```

Then initialize the contrastive melody tower from the pretraining checkpoint:

```bash
conda run -n try-contrastive python scripts/train_contrastive.py \
  --melody-pretrained-checkpoint checkpoints/melody_pretrain/best.pt
```

The contrastive melody projection head is initialized from scratch by default.

## Melody encoder diagnostics

The embedding diagnostic reports collapse and alignment metrics plus linear probes for
absolute pitch and the total number of annotated notes overlapping each segment. The
note-count probe requires a manifest produced by the current dataset preparation script.

```bash
conda run -n try-contrastive python scripts/diagnose_embeddings.py \
  --checkpoint checkpoints/contrastive/best.pt
```
