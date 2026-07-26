Basic contrastive training framework, with transformer-based melody encoder and HuBERT (froze CNN layer) as audio/waveform encoder. 
Training uses one variable-duration segment per DALI lyric line. Repeated occurrences
of the same normalized line within a song are removed, keeping the first occurrence.
I have used a variant based on the InfoNCE loss: L_{hard negatives} + L_{in-batch}, where hard negatives are other segments from the same song, and in-batch compares with other positive samples in the batch. 
Evaluation is against hard negatives (only). 

## Dataset preparation

Prepare line-aligned melody and audio segments:

```bash
conda run -n try-contrastive python scripts/prepare_dali_dataset.py
```

The manifest retains `voiced_ratio` for diagnostics, but vocal ratio is no longer
used to filter segments. Melody frames and audio waveforms are padded per batch and
masked during training, pretraining, diagnostics, and evaluation.

## Evaluation

```bash
conda run -n try-contrastive python scripts/evaluate_contrastive.py \
  --checkpoint checkpoints/contrastive/best.pt
```

Evaluation includes a segment-length probe. Its expected top-1 metric reports how
accurately duration alone can identify the positive candidate (with ties split
uniformly); the model top-1 length-match rate and score/proximity correlation report
whether model decisions actually favor duration-matched candidates.

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
