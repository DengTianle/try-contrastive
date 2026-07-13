Basic contrastive training framework, with transformer-based melody encoder and HuBERT (froze CNN layer) as audio/waveform encoder. 
Training on fixed-duration segments from DALI songs. 
I have used a variant based on the InfoNCE loss: L_{hard negatives} + L_{in-batch}, where hard negatives are other segments from the same song, and in-batch compares with other positive samples in the batch. 
Evaluation is against hard negatives (only). 

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
