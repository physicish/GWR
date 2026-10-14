# 03 — Evaluate on MLGWSC-1 Dataset 4

**Chapter section 1.3.** Produces **Figure 1.2(a)**.

This is where a classifier becomes a search. Instead of scoring independent
labelled windows, the model slides a one-second window through a month of
continuous strain at a 0.1 s stride, thresholds the ranking statistic, and
clusters nearby triggers into candidate events.

Dataset 4 provides a month of real O3a strain from Hanford and Livingston,
with and without simulated binary black hole signals. Running on both files
and scoring the pair gives sensitive distance against false-alarm rate, which
is what the challenge compares pipelines on.

## Getting the data

```bash
git clone https://github.com/gwastro/ml-mock-data-challenge-1
cd ml-mock-data-challenge-1
python generate_data.py -d 4 -i ds4_fg.hdf -b ds4_bg.hdf --duration 2592000
```

## Commands

Run the search over both files:

```bash
python apply_coherent_transformer.py \
    --inputfile  ds4_fg.hdf \
    --outputfile events_ds4_fg.hdf \
    --checkpoint ../02_train_coherent_gaussian/runs/gaussian/best_state_dict.pt \
    --step-size 0.1

python apply_coherent_transformer.py \
    --inputfile  ds4_bg.hdf \
    --outputfile events_ds4_bg.hdf \
    --checkpoint ../02_train_coherent_gaussian/runs/gaussian/best_state_dict.pt \
    --step-size 0.1
```

Score them with the challenge's own routine:

```bash
python -m gwsearch.evaluate \
    --injection-file ds4_injections.hdf \
    --foreground-events events_ds4_fg.hdf \
    --foreground-files  ds4_fg.hdf \
    --background-events events_ds4_bg.hdf \
    --output eval_ds4_gaussian.hdf
```

Plot it against the published pipelines:

```bash
python plot_sensitive_distance.py \
    --transformer eval_ds4_gaussian.hdf \
    --results-dir reference_curves/ \
    --output figures/fig1_2a_gaussian_trained.png
```

## What you should get

The curve collapses. It starts near 700 Mpc at the loosest thresholds, falls
steeply around a few hundred false alarms per month, and flattens at roughly
100 Mpc — it never reaches the one-per-month operating point the benchmark is
scored at. Every other pipeline in the figure stays above it throughout.

Nothing in Figure 1.1 predicted this.

## Why it fails

Two things changed at once. The noise is now real, with glitches and drifting
sensitivity that Gaussian training never showed the model, and the evaluation
now runs at false-alarm rates the classification test could not reach. The
model's only category for a loud, short, non-Gaussian excursion is *signal*,
so the background tail fills with transients it cannot reject, and the
threshold needed to suppress them discards most of the real signals too.

Stage 04 addresses the first of these.

## A note on the metric

Sensitive volume is estimated from recovered injections weighted by chirp mass
to the five-halves power, reflecting the challenge's sampling in chirp distance
rather than luminosity distance. The weighting favours heavier systems, so a
pipeline that misses light, long-duration signals can still post a respectable
sensitive distance. Worth remembering when comparing any two curves.

## Running time

About 18 minutes per file on an A100 80 GB for a month of data at a 0.1 s
stride.
