Basic contrastive training framework, with transformer-based melody encoder and HuBERT (froze CNN layer) as audio/waveform encoder. 
Training on fixed-duration segments from DALI songs. 
I have used a variant based on the InfoNCE loss: L_{hard negatives} + L_{in-batch}, where hard negatives are other segments from the same song, and in-batch compares with other positive samples in the batch. 
Evaluation is against hard negatives (only). 
