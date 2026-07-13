Basic contrastive training framework, with transformer-based melody encoder and HuBERT (froze CNN layer) as audio/waveform encoder. 
Training on fixed-duration segments from DALI songs. 
I have used a variant based on the InfoNCE loss: L_{hard negatives} + L_{in-batch}, where hard negatives are other segments from the same song, and in-batch compares with other positive samples in the batch. 
Evaluation is against hard negatives (only). 

## Melody encoder pretraining

Quantized melody tokens combine a duration ratio with either a rest or a
semitone pitch. By default, note pitches use the 88-key piano range (MIDI
21--108); out-of-range annotations are clipped to the nearest boundary.

Pretrain the melody encoder with masked onset, pitch, and duration-ratio
objectives before contrastive training:

```bash
conda run -n try-contrastive python scripts/pretrain_melody_encoder.py
```

Then initialize the contrastive melody tower from the pretraining checkpoint:

```bash
conda run -n try-contrastive python scripts/train_contrastive.py \
  --melody-pretrained-checkpoint checkpoints/melody_pretrain/best.pt
```

The contrastive melody projection head is initialized from scratch by default.
Use matching `--min-pitch-midi` and `--max-pitch-midi` options for both commands
to experiment with a smaller vocal range. The inclusive range may contain at
most 88 semitone bins.
