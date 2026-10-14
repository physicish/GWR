# 02 — Train the coherent transformer on Gaussian noise

**Chapter sections 1.2.2 and 1.3.** Produces **Figure 1.1**.

## The model

Three convolutional layers turn one second of two-detector strain into 64
tokens, which pass through six transformer layers of width 90, attention
pooling, and a two-layer classification head. Learned positional embeddings
supply token order, which attention does not encode on its own.

Both detectors enter together as two input channels — the *coherent*
configuration. Total: **664,522 parameters**.

Training minimises cross-entropy. The ranking statistic is the unbounded
softmax replacement, the difference of the two output scores,

```
USR = z_signal - z_noise
```

which orders candidates identically to the softmax probability but does not
saturate: score differences of 20 and 30 both round to a probability of 1 in
single precision, and a search needs to tell them apart.

## Commands

```bash
python train_coherent_pregen.py \
    --pregen-dir ../01_generate_gaussian_data/data_gaussian.h5 \
    --output-directory runs/gaussian \
    --epochs 250 \
    --batch-size 128 \
    --verbose
```

Then the ROC curves of Figure 1.1, on the held-out test set:

```bash
python plot_roc.py \
    --weights      runs/gaussian/best_state_dict.pt \
    --dataset-file ../01_generate_gaussian_data/data_gaussian_test.h5 \
    --group testing \
    --snr 6 8 10 15 20 \
    --output figures/fig1_1_roc_by_snr.png --verbose
```

## What you should get

About half an hour for 250 epochs on an A100 80 GB. Areas under the curve, by
injected network SNR:

| SNR | AUC |
|---|---|
| 6 | 0.8450 |
| 8 | 0.9583 |
| 10 | 0.9942 |
| 15 | 1.0000 |
| 20 | 1.0000 |

Perfect separation above SNR 15, and a drop at SNR 6 — below the lowest SNR
used in training, so expected.

## Loss curves

Every training run writes `loss_curve.pdf` and `loss_curve.png` into its
output directory when it finishes, alongside the raw history in `losses.txt`.
The epoch with the lowest validation loss is marked with a star, and is the
epoch saved as the best checkpoint.

The validation curve is the loss on the held-out validation examples.

To redraw a figure without retraining:

```bash
python -c "from gwsearch.plotting import plot_loss_curves; \
           plot_loss_curves('runs/gaussian/losses.txt', 'runs/gaussian')"
```

Pass `--no-loss-curve` to skip the figure, or `--loss-curve-log` for a
logarithmic loss axis. If matplotlib is unavailable the run still completes
and the history is kept — only the figure is skipped.

## Why this figure proves less than it looks

A ROC curve is computed on a balanced test set. With 100,000 noise-only
examples the smallest non-zero false-positive rate it can measure is 1×10⁻⁵,
about 250 times larger than the ~4×10⁻⁸ per-window rate that one false alarm
per month demands. The region that decides a search is simply not sampled.

The classification test also scores each window independently. A real search
sees the same signal or glitch in many overlapping windows and must cluster
them into events. Stage 03 makes the measurement that accounts for both.

## Resuming

Training writes `last_state_dict.pt` each epoch. To continue an interrupted
run, pass `--resume` with the same output directory.
