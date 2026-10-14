# Supplementary — the experiment behind Box 2

**Not required to reproduce any figure in the chapter.** Box 2 states the
amplitude-convention problem as a precaution; this directory holds the
experiment that produced that advice, kept for readers who want to see the
effect rather than take it on trust.

## The issue

PyCBC-whitened strain has a standard deviation near 32 at 2048 Hz, because
whitening sets the one-sided PSD to unity per hertz and the variance is the
integral of the PSD over the retained band. GWpy, `ml4gw` and Bilby apply a
further normalisation and return fluctuations of order unity. Neither is wrong,
but "whitened" does not fix the numerical scale.

`amplitude_normalisation.md` works through the derivation, the numerical
verification across sample rates, why the effect is specific to where the
positional embedding is added, and how a convolutional network fails
differently.

## The model here

`train_original_model.py` and `apply_original_model.py` contain an earlier
architecture that adds its positional embedding directly to the tokenizer
output, with no normalisation in between. It is **not** the model used anywhere
in the chapter — the model in `gwsearch/coherent_model.py` normalises the
tokens first, which makes the amplitude convention irrelevant by construction.

## What the figures show

| Figure | Input | Result |
|---|---|---|
| `sensitive_distance_unscaled.png` | PyCBC convention, σ ≈ 32 | ~350 Mpc, collapses early |
| `sensitive_distance_scaled.png` | Same data divided by 32 | ~700 Mpc |

Same weights, same training budget, same evaluation. The only difference is the
amplitude of the numbers entering the first layer.

The control that settles the interpretation is in the main repository: the
corrected architecture, given the **unscaled** data that produced the collapse,
reproduces the scaled result. The division was never supplying information — it
was compensating for where the embedding was added.

## The lesson

A normalisation layer removes a global input scale, because it divides by a
standard deviation measured from the data. But it only protects what comes
*after* it. Anything with a fixed, data-independent scale that meets the data
earlier — a positional embedding, a bias, a hard-coded threshold — stays
exposed. Placement matters more than which normalisation you choose.
