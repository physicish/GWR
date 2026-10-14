# 06 — The coincident design

**Chapter section 1.4.3.** Produces **Figure 1.2(d)**.

A practical question motivates this stage: how expensive is it to establish
that a candidate's false-alarm rate is below one per year?

MLGWSC-1 supplies one month of noise-only data, so its benchmark needs no time
slides. Lower false-alarm rates need more background than the observation
contains, and the standard way to manufacture it is to shift one detector
relative to the other by more than the allowed coincidence interval, breaking
astrophysical coincidences while creating accidental ones.

For the **coherent** model this is expensive. Both detectors enter the network
together, so changing the pairing changes its joint input, and the full 664k
network must be evaluated again for every slide. One month of data without a
shift takes about **18 minutes**; 100 slides would take about **30 hours**.

## The change

Process one detector at a time: the first convolutional layer goes from two
input channels to one, and the rest of the network is unchanged. The same
weights score Hanford and Livingston independently, giving a separate pair of
scores per detector, combined afterwards by

```
USR_HL = ln( exp(s_H) + exp(s_L) + exp(s_H + s_L) )
```

which increases when either detector provides strong evidence.

At **664,042 parameters** this is marginally *smaller* than the coherent
model's 664,522, so any performance difference can be attributed to the design
rather than to capacity.

The practical gain: per-detector outputs are computed once and cached. A time
slide then pairs outputs from different times and repeats only the combination
and the clustering, with no further network evaluation.

## Commands

```bash
python train_coincident.py \
    --pregen-dir       ../04_train_on_real_noise/data_o3_pregen.h5 \
    --noise-dir        ../05_mixed_ml4gw_training/noise_ml4gw_36h.hdf \
    --output-directory runs/coincident \
    --epochs 250 --batch-size 256 \
    --pregen-prob-start 0.1 --pregen-prob-end 0.3 \
    --pregen-ramp-epochs 100 \
    --verbose
```

Evaluate on Dataset 4 (no slides):

```bash
python apply_coincident_transformer.py \
    --inputfile  ds4_fg.hdf \
    --outputfile events_ds4_fg_coinc.hdf \
    --checkpoint runs/coincident/best_state_dict.pt \
    --step-size 0.1
```

Then score and plot as in stage 03.

## Loss curves

Every training run writes `loss_curve.pdf` and `loss_curve.png` into its
output directory when it finishes, alongside the raw history in `losses.txt`.
The epoch with the lowest validation loss is marked with a star, and is the
epoch saved as the best checkpoint.

The validation curve is the **mean** of the on-the-fly and
pre-generated validation losses, which is the quantity this script
minimises when choosing which checkpoint to keep, so the starred epoch
always matches the saved model.

To redraw a figure without retraining:

```bash
python -c "from gwsearch.plotting import plot_loss_curves; \
           plot_loss_curves('runs/coincident/losses.txt', 'runs/coincident')"
```

Pass `--no-loss-curve` to skip the figure, or `--loss-curve-log` for a
logarithmic loss axis. If matplotlib is unavailable the run still completes
and the history is kept — only the figure is skipped.

## What you should get

**1364 Mpc** at one false alarm per month — essentially identical to the
coherent model's 1303 Mpc across the plotted range.

## What this does and does not establish

Equivalent sensitivity on Dataset 4 does not mean the two designs are
equivalent in general. The coherent model can compare waveform patterns and
their relative timing directly between detectors, which could help reject
glitches that look convincing in one detector alone. Whether it actually learns
to use that depends on its training, so neither design is inherently more
sensitive.

What this comparison establishes is that, on this benchmark, adopting the
coincident design costs nothing measurable — and the reason to adopt it is the
cost of the background estimate, which stage 07 measures.
