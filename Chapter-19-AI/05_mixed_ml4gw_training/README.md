# 05 — Add on-the-fly data generation

**Chapter section 1.4.2.** Produces **Figure 1.2(c)**.

Training on real noise fixed the failure mode but left a gap. The model still
saw a fixed collection of pre-generated waveforms and noise segments; varying
amplitudes and pairings creates new *examples*, but no new waveforms and no new
noise features.

This stage generates fresh signals *during* training with
[`ml4gw`](https://github.com/ML4GW/ml4gw), which does waveform generation,
detector projection, SNR calculation, PSD estimation and whitening on the GPU,
so batches are prepared alongside the network's training updates.

## The waveform mixture

There is a practical limitation. At the time of writing `ml4gw` supports
`IMRPhenomPv2`, which includes precession but **not** higher-order modes, while
Dataset 4 uses `IMRPhenomXPHM`, which includes both.

Rather than choose, the chapter combines both sources:

- most batches are generated on the fly with `IMRPhenomPv2`, using the same
  mass, spin and orientation distributions as before;
- the rest are drawn from the pre-generated `IMRPhenomXPHM` set of stage 04.

The probability of drawing a pre-generated batch rises from **10% to 30% over
the first 100 epochs**. The on-the-fly stream supplies waveform variety; the
pre-generated set keeps the higher-mode content of the benchmark represented.

## Extra noise

This stage also adds **36 hours** of real O3a noise beyond the stage 04 pool,
drawn from segments distributed across the observing run, GPS range
**1238175744–1253920768**. Hanford and Livingston are sampled at matching times
so coincident noise conditions are preserved, and each example's PSD is
estimated from neighbouring data that does not contain the injection.

## Commands

```bash
python train_coherent_mix.py \
    --pregen-dir       ../04_train_on_real_noise/data_o3_pregen.h5 \
    --noise-dir        noise_ml4gw_36h.hdf \
    --validation-noise-dir noise_ml4gw_val.hdf \
    --output-directory runs/mixed \
    --epochs 250 --batch-size 256 \
    --pregen-prob-start 0.1 \
    --pregen-prob-end   0.3 \
    --pregen-ramp-epochs 100 \
    --verbose
```

Evaluate as in stage 03, with `--checkpoint runs/mixed/best_state_dict.pt`.

## Validation

Validation uses a **fixed** set of examples from both sources. This matters:
if validation batches were also generated on the fly, successive measurements
would differ because the data differed, and the checkpoint selection would be
comparing noise. The checkpoint with the lowest average validation loss is kept.

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
           plot_loss_curves('runs/mixed/losses.txt', 'runs/mixed')"
```

Pass `--no-loss-curve` to skip the figure, or `--loss-curve-log` for a
logarithmic loss axis. If matplotlib is unavailable the run still completes
and the history is kept — only the figure is skipped.

## What you should get

**1303 Mpc** at one false alarm per month — the largest single improvement in
the chapter. This places the transformer above cWB across much of the range and
close to TPI FSU Jena, though still below PyCBC, AresGW and SAGE at low
false-alarm rates.

## Running time

About five hours on an A100 80 GB. Slower per epoch than stage 04 because
waveforms are simulated as training proceeds, but it removes the ceiling that a
fixed dataset imposes.

## If ml4gw will not install

This is the only stage that needs it. Everything else runs without it.
