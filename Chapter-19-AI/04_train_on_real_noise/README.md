# 04 — Retrain on real detector noise

**Chapter section 1.4.1.** Produces **Figure 1.2(b)**.

The model failed on Dataset 4 partly because it had never seen real noise.
This stage changes nothing but the noise the training examples are built from:
same architecture, same parameter count, same training budget.

## Getting the noise

The MLGWSC-1 repository supplies the download and segment selection. The data
is GWOSC's public 4 kHz O3a release over GPS **1238166018–1253977218**,
restricted to times where both detectors satisfy the challenge's quality
requirements, with hardware injections and known GWTC-2 events removed and
segments shorter than two hours discarded. About 81 days of coincident live
time survive, in a 94 GB file.

## Preparing it

Two stages, and the order matters.

```bash
# 1. Split along segment boundaries, so training and validation never
#    share a stretch of strain.
python prepare_real_noise.py split \
    real_noise_o3a.hdf noise_train.hdf noise_val.hdf \
    --duration 4147200 \
    --seed 2026 --verbose

# 2. Whiten each segment, slice into one-second samples, store the PSD
#    used for each.
python prepare_real_noise.py slice \
    noise_train.hdf --output noise_train_sliced.hdf \
    --step-size 0.5 --verbose

python prepare_real_noise.py slice \
    noise_val.hdf --output noise_val_sliced.hdf \
    --step-size 0.5 --verbose
```

**Split before slicing, never after.** Consecutive slices overlap by half their
length and are strongly correlated. Splitting a set of overlapping slices would
put near-duplicate windows on both sides of the boundary, and validation
performance would look better than it is. This is the second half of Box 1.

## Adding signals and training

Injections use `IMRPhenomXPHM` with the same source-parameter ranges as
Dataset 4. The chapter's training set is 1,020,000 examples, half containing a
signal; validation is 204,000 examples, also half signal.

```bash
python ../01_generate_gaussian_data/generate_training_data.py \
    --output-file data_o3_pregen.h5 \
    --real-noise-file noise_train_sliced.hdf \
    --training-samples   510000 510000 \
    --validation-samples 102000 102000 \
    --approximant IMRPhenomXPHM --verbose

python ../02_train_coherent_gaussian/train_coherent_pregen.py \
    --pregen-dir data_o3_pregen.h5 \
    --output-directory runs/o3_pregen \
    --epochs 250 --batch-size 256 \
    --verbose
```

Evaluate exactly as in stage 03, pointing `--checkpoint` at
`runs/o3_pregen/best_state_dict.pt`.

## What you should get

Training stops around **epoch 97** by early stopping, with the best validation
loss of **0.04938** at **epoch 67**. On Dataset 4 the sensitive distance
reaches **364 Mpc** at one false alarm per month.

The qualitative change matters more than the number: the collapse is gone. The
curve now descends smoothly across the whole false-alarm range instead of
giving out partway, which is what we would expect if the earlier failure came
from an unmodelled glitch background rather than from an inability to
recognise signals.

It is still roughly three times below PyCBC at the same false-alarm rate.
Stage 05 takes up why.

## Running time

Preparation is I/O-bound and takes a few hours for the full file. Training is
about two hours on an A100 80 GB.
