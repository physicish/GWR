# 01 — Gaussian-noise datasets

**Chapter section 1.2.1.** Builds the simulated datasets the first model trains
and is tested on.

Noise is Gaussian, coloured by the Advanced LIGO Zero Detuning High Power
design-sensitivity PSD. Signals use `IMRPhenomXPHM` over the Dataset 4 source
parameters: component masses uniform in 7–50 M☉, dimensionless spin magnitudes
uniform in 0–0.99, isotropic spin directions.

## How the files are laid out

Waveforms and noise are stored separately rather than as finished examples.
Each signal is normalised to unit network SNR, and a signal-containing example
is assembled during training as

```
x = n + rho * h
```

with `rho` drawn afresh on every access. The SNR distribution is therefore a
training-time choice, and one stored waveform is seen at many amplitudes.

Every example needs its own noise realisation, but only half carry a signal.
`N_inj` injections and `N_noise` pure-noise examples therefore need `N_inj`
waveforms and `N_inj + N_noise` noise segments — which is why the noise array
is twice the length of the waveform array when the classes are balanced.

## Commands

Training and validation sets (the chapter's sizes):

```bash
python generate_training_data.py \
    --output-file data_gaussian.h5 \
    --training-samples   50000 50000 \
    --validation-samples 10000 20000 \
    --verbose
```

Held-out test set, written under a single `testing` group:

```bash
python generate_test_data.py \
    --output-file data_gaussian_test.h5 \
    --samples 100000 100000 \
    --verbose
```

Inspect what you made before training on it:

```bash
python plot_dataset.py data_gaussian.h5 --group training --output figures/dataset_overview.png
```

## What to check

`plot_dataset.py` reports the amplitude statistics. PyCBC-whitened data has a
standard deviation near **32** at 2048 Hz, not unity — this is the convention
discussed in Box 2 of the chapter, and confirming it here costs seconds where
discovering it after a training run costs hours.

If you intend to train with a different package's whitening, make the
convention consistent across generation, training and evaluation.

## Running time

About an hour for the training set on a typical workstation CPU; the test set
is roughly twice that. Both are dominated by waveform generation, and
`--verbose` gives a progress bar.
