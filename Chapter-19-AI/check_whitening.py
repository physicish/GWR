#!/usr/bin/env python
"""
Compare whitening settings on real data, against the training distribution.

The network only ever sees whitened samples, so what has to match between
training and application is the distribution of those samples -- not the value
of any particular whitening parameter. This script measures that directly. It
whitens the same stretch of real strain several times, once per candidate
filter length, and compares each result with the data the model was actually
trained on.

It answers the question left open in docs/whitening_filter_length.pdf: a long
filter suppresses narrow spectral lines but smears transients over seconds,
while a short one confines transients but leaves line power the model has never
seen. Which matters more is a property of the data, so it is measured here
rather than argued.

------------------------------------------------------------------------------
WHAT IS MEASURED
------------------------------------------------------------------------------
(a) Amplitude spectral density. Correct whitening gives a flat spectrum. Peaks
    that survive are residual instrumental lines. Reported as the ratio of the
    99th percentile to the median across the analysis band: 1.0 is perfectly
    flat, larger values mean lines are getting through.

(b) Sample amplitude distribution, in units of its own standard deviation, so
    the SHAPE can be compared against a Gaussian independently of scale. Heavy
    tails indicate non-Gaussian transients reaching the network; excess
    kurtosis and the rate of excursions beyond four sigma quantify them.

(c) Per-window standard deviation, relative to the median window. A tail
    towards large values counts the windows a glitch has contaminated -- the
    quantity a long filter inflates by spreading transients.

Note on scale: pycbc's whitening convention does NOT produce unit variance. It
yields a standard deviation of sqrt(f_s/2), about 32 at 2048 Hz, which is why
the original CASTOR code carried a hard-coded factor of 32. The absolute value
of the "std" column is therefore only meaningful by COMPARISON with the
training row; a setting that departs from it is presenting the network with
data of the wrong amplitude altogether.

In every panel the training set is drawn as a reference. The setting whose
curves lie closest to it is the one that presents the model with data most like
what it learned on.

------------------------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------------------------
Requires: pycbc, torch, numpy, h5py, matplotlib, tqdm, and apply_baseline.py
alongside (the whitening function is imported from it, so this tests exactly
the code the search will run).

    # The basic comparison: two filter lengths against the training set
    python check_whitening.py \
        --inputfile ds4_fg.hdf \
        --training-file data_gaussian.h5 \
        -o figures/whitening_check --verbose

    # Try more values, and use more data for better statistics
    python check_whitening.py --inputfile ds4_fg.hdf \
        --training-file data_gaussian.h5 \
        --filter-durations 8 4 1 0.25 \
        --n-segments 8 --seconds-per-segment 512 \
        -o figures/whitening_check --verbose

    # Without a training file, the comparison is against a unit Gaussian only
    python check_whitening.py --inputfile ds4_fg.hdf -o figures/check

Outputs <stem>.pdf, <stem>.png and a summary table on stdout.

Note that this reads the UNWHITENED foreground file, since it performs the
whitening itself. Run it on the background file too: the chosen setting must be
used for both, or the comparison between them is meaningless.
------------------------------------------------------------------------------
"""

from argparse import ArgumentParser
import logging
import os
import sys

import h5py
import numpy as np
from tqdm import tqdm

# The authoritative whitening implementation, so that what is measured here is
# what the search will actually do.
from apply_baseline import whiten

TRAINING_COLOUR = "#000000"
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9"]

STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9.5,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": False,
    "savefig.bbox": "tight",
}


# =============================================================================
# Measurement
# =============================================================================
def amplitude_spectrum(windows, sample_rate):
    """Mean one-sided ASD over a stack of equal-length windows."""
    spectra = np.fft.rfft(windows, axis=-1)
    n_time = windows.shape[-1]
    power = 2.0 * np.abs(spectra) ** 2 / (n_time * sample_rate)
    frequencies = np.fft.rfftfreq(n_time, d=1.0 / sample_rate)
    return frequencies, np.sqrt(power.reshape(-1, power.shape[-1]).mean(axis=0))


def to_windows(data, window_samples):
    """Cut (n_detectors, n_samples) into non-overlapping windows.

    Returns an array of shape (n_windows * n_detectors, window_samples): the
    network sees one detector pair per window, but for these one-dimensional
    statistics the detectors are simply pooled.
    """
    n_detectors, n_samples = data.shape
    usable = (n_samples // window_samples) * window_samples
    if usable == 0:
        return np.empty((0, window_samples), dtype=np.float64)
    trimmed = data[:, :usable].reshape(n_detectors, -1, window_samples)
    return trimmed.reshape(-1, window_samples).astype(np.float64)


def summarise(windows, sample_rate, band=(20.0, 1000.0)):
    """Reduce a stack of whitened windows to the diagnostics of interest."""
    flat = windows.ravel()
    mean, std = float(flat.mean()), float(flat.std())

    centred = flat - mean
    kurtosis = float((centred ** 4).mean() / max(centred.var() ** 2, 1e-30) - 3.0)
    outlier_rate = float(np.mean(np.abs(centred) > 4.0 * max(std, 1e-30)))

    frequencies, asd = amplitude_spectrum(windows, sample_rate)
    in_band = (frequencies >= band[0]) & (frequencies <= band[1])
    band_asd = asd[in_band]
    # A single narrow line occupies only a handful of the ~1000 in-band bins,
    # so a high percentile does not register it -- the maximum is what detects
    # a line, and the median gives the noise floor it is measured against.
    flatness = float(band_asd.max() / max(np.median(band_asd), 1e-30))

    # Expressed RELATIVE to the typical window, so the statistic does not
    # depend on the overall normalisation of the whitening convention. An
    # absolute threshold would be meaningless: pycbc's whitened output has a
    # standard deviation of sqrt(f_s/2), about 32 at 2048 Hz, not 1.
    window_std = windows.std(axis=-1)
    typical = max(float(np.median(window_std)), 1e-30)
    relative_window_std = window_std / typical
    loud_fraction = float(np.mean(relative_window_std > 1.5))

    return {
        "mean": mean, "std": std, "kurtosis": kurtosis,
        "outlier_rate": outlier_rate, "flatness": flatness,
        "loud_window_fraction": loud_fraction,
        "frequencies": frequencies, "asd": asd,
        "samples": flat / max(std, 1e-30),          # in units of its own sigma
        "window_std": relative_window_std,
        "n_windows": int(windows.shape[0]),
    }


# =============================================================================
# Data
# =============================================================================
def load_raw_segments(path, detectors, n_segments, seconds, rng):
    """Read a sample of UNWHITENED segments from an MLGWSC-1 strain file."""
    with h5py.File(path, "r") as handle:
        missing = [d for d in detectors if d not in handle]
        if missing:
            raise KeyError(f"Detector group(s) {missing} not found in {path}. "
                           f"Available: {list(handle.keys())}")
        keys = sorted(handle[detectors[0]].keys())
        if not keys:
            raise KeyError(f"No segments found in {path}.")
        chosen = [keys[i] for i in
                  rng.choice(len(keys), min(n_segments, len(keys)), replace=False)]

        segments = []
        for key in chosen:
            delta_t = float(handle[detectors[0]][key].attrs["delta_t"])
            limit = int(round(seconds / delta_t)) if seconds else None
            block = np.stack([handle[d][key][:limit] for d in detectors])
            segments.append((key, block.astype(np.float64), delta_t))
    return segments


def load_training_reference(path, group, max_samples):
    """Whitened noise the model was trained on, as (n, window) windows."""
    with h5py.File(path, "r") as handle:
        if group not in handle:
            for candidate in ("training", "validation", "test"):
                if candidate in handle:
                    group = candidate
                    break
            else:
                raise KeyError(f"No usable group in {path}: "
                               f"{list(handle.keys())}")
        noises = handle[group]["noises"][:max_samples]
        rate = int(handle.attrs.get("sample_rate", 2048))
    windows = noises.reshape(-1, noises.shape[-1]).astype(np.float64)
    return windows, rate, group


# =============================================================================
# Plot
# =============================================================================
def plot_comparison(results, reference, output_stem, band, dpi=300):
    """Three panels: spectrum, sample amplitudes, per-window spread."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))
        entries = list(results.items())

        # ---- (a) amplitude spectral density ------------------------------
        ax = axes[0]
        if reference is not None:
            ax.loglog(reference["frequencies"][1:], reference["asd"][1:],
                      lw=2.2, color=TRAINING_COLOUR, alpha=0.45,
                      label="training set", zorder=1)
        for (duration, data), colour in zip(entries, PALETTE):
            ax.loglog(data["frequencies"][1:], data["asd"][1:], lw=1.2,
                      color=colour, label=f"{duration:g} s filter", zorder=2)
        ax.set_xlim(band)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(r"ASD [1/$\sqrt{\rm Hz}$]")
        ax.set_title("(a) Spectral flatness", size=10, pad=6)
        ax.grid(alpha=0.25, lw=0.5, which="both")
        ax.legend(loc="upper right")

        # ---- (b) sample amplitude distribution ---------------------------
        ax = axes[1]
        bins = np.linspace(-8, 8, 241)
        grid = 0.5 * (bins[1:] + bins[:-1])
        ax.semilogy(grid, np.exp(-0.5 * grid ** 2) / np.sqrt(2 * np.pi),
                    ls=(0, (4, 3)), lw=1.2, color="0.45", label="unit Gaussian")
        if reference is not None:
            counts, _ = np.histogram(reference["samples"], bins=bins, density=True)
            ax.semilogy(grid, np.maximum(counts, 1e-9), lw=2.2,
                        color=TRAINING_COLOUR, alpha=0.45, label="training set")
        for (duration, data), colour in zip(entries, PALETTE):
            counts, _ = np.histogram(data["samples"], bins=bins, density=True)
            ax.semilogy(grid, np.maximum(counts, 1e-9), lw=1.2, color=colour,
                        label=f"{duration:g} s filter")
        ax.set_xlim(-8, 8)
        ax.set_ylim(1e-7, 1.0)
        ax.set_xlabel(r"Whitened strain / $\sigma$")
        ax.set_ylabel("Density")
        ax.set_title("(b) Amplitude distribution (tails)", size=10, pad=6)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(loc="lower center")

        # ---- (c) per-window standard deviation ---------------------------
        ax = axes[2]
        bins = np.linspace(0, 4, 161)
        if reference is not None:
            ax.hist(reference["window_std"], bins=bins, density=True,
                    histtype="stepfilled", color=TRAINING_COLOUR, alpha=0.18,
                    label="training set")
        for (duration, data), colour in zip(entries, PALETTE):
            ax.hist(data["window_std"], bins=bins, density=True,
                    histtype="step", lw=1.3, color=colour,
                    label=f"{duration:g} s filter")
        ax.axvline(1.0, color="0.45", ls=(0, (4, 3)), lw=1.2)
        ax.set_yscale("log")
        ax.set_xlim(0, 4)
        ax.set_xlabel("Window standard deviation / median")
        ax.set_ylabel("Density")
        ax.set_title("(c) Per-window spread (glitches)", size=10, pad=6)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(loc="upper right")

        fig.tight_layout()
        written = []
        for extension in ("pdf", "png"):
            path = f"{output_stem}.{extension}"
            fig.savefig(path, dpi=dpi)
            written.append(path)
        plt.close(fig)
    return written


def print_table(results, reference):
    """Print the diagnostics, closest-to-training being the thing to look for."""
    header = (f"{'setting':>16} {'mean':>8} {'std':>7} {'kurtosis':>9} "
              f"{'>4 sigma':>10} {'flatness':>9} {'loud win':>9}")
    print("\n" + header)
    print("-" * len(header))
    if reference is not None:
        print(f"{'training set':>16} {reference['mean']:>8.3f} "
              f"{reference['std']:>7.3f} {reference['kurtosis']:>9.3f} "
              f"{reference['outlier_rate']:>10.2e} "
              f"{reference['flatness']:>9.3f} "
              f"{reference['loud_window_fraction']:>9.2e}")
        print("-" * len(header))
    for duration, data in results.items():
        print(f"{f'{duration:g} s filter':>16} {data['mean']:>8.3f} "
              f"{data['std']:>7.3f} {data['kurtosis']:>9.3f} "
              f"{data['outlier_rate']:>10.2e} {data['flatness']:>9.3f} "
              f"{data['loud_window_fraction']:>9.2e}")
    print("\n  flatness  loudest in-band ASD bin / median; near 1 is flat, "
          "and a large\n            value means a residual spectral line "
          "survived whitening.")
    print("  loud win  fraction of 1 s windows whose standard deviation exceeds "
          "1.5x the\n            median window, a proxy for transient "
          "contamination.")
    if reference is not None:
        print("\n  Prefer the setting whose numbers sit closest to the "
              "training row, and\n  confirm with the sensitive distance if two "
              "settings look comparable.")


# =============================================================================
# Command-line interface
# =============================================================================
def main():
    parser = ArgumentParser(
        description="Compare whitening filter lengths on unwhitened strain, "
                    "against the training distribution.")
    parser.add_argument("--inputfile", required=True,
                        help="UNWHITENED MLGWSC-1 strain file, e.g. the "
                             "dataset-4 foreground.")
    parser.add_argument("--training-file", default=None,
                        help="Dataset from generate_dataset.py, used as the "
                             "reference distribution. Strongly recommended: "
                             "without it the only reference is a unit "
                             "Gaussian.")
    parser.add_argument("--training-group", default="training",
                        help="Group to read from the training file. "
                             "Default: training.")
    parser.add_argument("-o", "--output", default="whitening_check",
                        help="Output path stem. Default: whitening_check.")
    parser.add_argument("--filter-durations", type=float, nargs="+",
                        default=[4.0, 0.25], metavar="SECONDS",
                        help="Whitening filter lengths to compare. "
                             "Default: 4 0.25.")
    parser.add_argument("--whitening-segment-duration", type=float, default=4.0,
                        help="Welch segment length for the PSD estimate, held "
                             "fixed across settings. Default: 4.")
    parser.add_argument("--low-frequency-cutoff", type=float, default=20.0,
                        help="Cutoff for inverse spectrum truncation. "
                             "Default: 20.")
    parser.add_argument("--bandpass-lower", type=float, default=20.0,
                        help="High-pass corner after whitening. Default: 20.")
    parser.add_argument("--n-segments", type=int, default=4,
                        help="Number of segments to sample. Default: 4.")
    parser.add_argument("--seconds-per-segment", type=float, default=512.0,
                        help="Seconds read from each segment; 0 reads all. "
                             "Default: 512.")
    parser.add_argument("--window-duration", type=float, default=1.0,
                        help="Window length for the statistics, matching the "
                             "search. Default: 1.")
    parser.add_argument("--max-training-samples", type=int, default=2000,
                        help="Training examples used for the reference. "
                             "Default: 2000.")
    parser.add_argument("--band", type=float, nargs=2, default=(20.0, 1000.0),
                        metavar=("LOW", "HIGH"),
                        help="Band for the flatness statistic. "
                             "Default: 20 1000.")
    parser.add_argument("--n-detectors", type=int, default=2, choices=[1, 2])
    parser.add_argument("--seed", type=int, default=2026,
                        help="Seed for choosing segments. Default: 2026.")
    parser.add_argument("--verbose", action="store_true", help="Print progress.")

    args = parser.parse_args()
    logging.basicConfig(
        format="%(levelname)s | %(asctime)s: %(message)s",
        level=logging.INFO if args.verbose else logging.WARN,
        datefmt="%d-%m-%Y %H:%M:%S")

    detectors = ("H1", "L1")[:args.n_detectors]
    rng = np.random.default_rng(args.seed)

    try:
        segments = load_raw_segments(
            args.inputfile, detectors, args.n_segments,
            args.seconds_per_segment, rng)
    except (KeyError, OSError) as exc:
        print(f"\nError: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        sys.exit(1)

    total = sum(block.shape[1] * delta_t for _, block, delta_t in segments)
    logging.info("Read %i segment(s), %.1f s of strain per detector.",
                 len(segments), total)

    # -- reference ---------------------------------------------------------
    reference = None
    if args.training_file:
        try:
            windows, rate, group = load_training_reference(
                args.training_file, args.training_group,
                args.max_training_samples)
        except (KeyError, OSError) as exc:
            print(f"\nError reading the training file: "
                  f"{exc.args[0] if exc.args else exc}", file=sys.stderr)
            sys.exit(1)
        reference = summarise(windows, rate, band=tuple(args.band))
        logging.info("Reference: %i windows from group '%s' of %s.",
                     reference["n_windows"], group, args.training_file)
    else:
        logging.warning("No --training-file given; comparing against a unit "
                        "Gaussian only, which cannot reveal a mismatch with "
                        "the data the model actually saw.")

    # -- whiten at each setting -------------------------------------------
    results = {}
    for duration in args.filter_durations:
        stacks = []
        for key, block, delta_t in tqdm(segments, ascii=True,
                                        desc=f"{duration:g} s filter",
                                        disable=not args.verbose):
            whitened = whiten(
                block, delta_t,
                segment_duration=args.whitening_segment_duration,
                max_filter_duration=duration,
                low_frequency_cutoff=args.low_frequency_cutoff,
                bandpass_lower=args.bandpass_lower,
                remove_corrupted=True)
            sample_rate = int(round(1.0 / delta_t))
            stacks.append(to_windows(
                whitened, int(round(args.window_duration * sample_rate))))
        windows = np.concatenate([s for s in stacks if len(s)], axis=0)
        results[duration] = summarise(windows, sample_rate,
                                      band=tuple(args.band))
        logging.info("%g s filter: %i windows.", duration,
                     results[duration]["n_windows"])

    parent = os.path.dirname(args.output)
    if parent:
        os.makedirs(parent, exist_ok=True)
    written = plot_comparison(results, reference, args.output,
                              tuple(args.band))
    print_table(results, reference)
    print("\nWrote " + ", ".join(written))


if __name__ == "__main__":
    main()
