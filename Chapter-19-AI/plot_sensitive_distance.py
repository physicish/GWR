#!/usr/bin/env python
"""
Plot sensitive distance versus false-alarm rate on MLGWSC-1 dataset 4.

This is the measurement the ROC curve could not make. A ROC is computed on a
balanced test set and says nothing about behaviour at one false alarm per
month; this figure places the baseline transformer on the axes the challenge
actually scores, alongside the published pipelines.

The baseline curve is the slot to fill: pass its evaluation output with
--baseline. Everything else is context, and the script runs happily without it,
so the reference figure can be produced before the baseline result exists.

------------------------------------------------------------------------------
WHERE THE INPUT COMES FROM
------------------------------------------------------------------------------
Each curve is read from an evaluation file written by the MLGWSC-1 scoring
script, which contains a `far` dataset (in Hz, converted to events per 30-day
month here) and a `sensitive-distance` dataset in Mpc. To produce one for the
baseline, run the search over both the foreground and background files and then
score them:

    python apply_baseline.py --inputfile ds4_fg.hdf \
        --outputfile events_ds4_fg.hdf --checkpoint best_state_dict.pt ...
    python apply_baseline.py --inputfile ds4_bg.hdf \
        --outputfile events_ds4_bg.hdf --checkpoint best_state_dict.pt ...

    python evaluate.py --injection-file injections.hdf \
        --foreground-events events_ds4_fg.hdf --foreground-files ds4_fg.hdf \
        --background-events events_ds4_bg.hdf --output eval_baseline_ds4.hdf

Both searches must use identical whitening settings, or the foreground and
background are not comparable.

------------------------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------------------------
Requires: numpy, h5py, matplotlib

    # With the baseline result
    python plot_sensitive_distance.py \
        --baseline eval_baseline_ds4.hdf \
        --results-dir /path/to/results \
        -o figures/sensitive_distance_ds4

    # Reference pipelines only, before the baseline has been run
    python plot_sensitive_distance.py --results-dir /path/to/results \
        -o figures/sensitive_distance_ds4

    # Widen the vertical range if the baseline falls below the published set
    python plot_sensitive_distance.py --baseline eval_baseline_ds4.hdf \
        --results-dir /path/to/results --ylim 30 2500 -o figures/sd_ds4

Missing reference files are skipped with a warning rather than treated as an
error, so a partial results directory still produces a figure.

------------------------------------------------------------------------------
Adapted for this chapter from the CASTOR paper's dataset-3/4 comparison figure.
Copyright 2026 Chayan Chatterjee.
------------------------------------------------------------------------------
"""

from argparse import ArgumentParser
import logging
import os
import sys

import h5py
import numpy as np


SECONDS_PER_MONTH = 30.0 * 24.0 * 60.0 * 60.0

# The baseline is the subject of the figure and is drawn heavily; the published
# pipelines are context and are drawn in grey, distinguished by line style. The
# two conventional analyses keep their own colours, since the comparison
# against matched filtering and coherent WaveBurst is the one readers look for.
CLR_BASELINE = "#0072B2"
CLR_PYCBC = "#B2182B"
CLR_CWB = "#EF8A62"
CLR_GRAY = "#555555"

GRAY_LINE_STYLES = ["-", "--", "-.", ":", (0, (5, 2)), (0, (3, 1, 1, 1))]

# Learned pipelines read from MLGWSC-1 evaluation files, relative to
# --results-dir. Any that are missing are skipped.
REFERENCE_HDF = [
    ("CNN-Coinc", "CNN-Coinc/ds4/eval.hdf"),
    ("MFCNN", "MFCNN/ds4/eval.hdf"),
    ("TPI FSU Jena", "TPI_FSU_Jena/challenge_ds4/eval.hdf"),
]

# Pipelines whose published dataset-4 curves are distributed as two-column
# arrays of (FAR per month, sensitive distance in Mpc) rather than HDF5.
REFERENCE_NPY = [
    ("AresGW", "aresgw_sensitive_distance.npy"),
    ("SAGE", "sage_sensitive_distance.npy"),
]

CONVENTIONAL_HDF = [
    ("cWB", "cWB/ds4/eval.hdf", CLR_CWB, "--", 2.2),
    ("PyCBC", "PyCBC/ds4/eval.hdf", CLR_PYCBC, ":", 2.4),
]

STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 13,
    "axes.labelsize": 15,
    "axes.titlesize": 16,
    "axes.linewidth": 1.2,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3.5,
    "ytick.minor.size": 3.5,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
    "savefig.bbox": "tight",
}


# =============================================================================
# Loading
# =============================================================================
def prepare_curve(far, distance, sort_descending=True):
    """Drop points that cannot be drawn on logarithmic axes, and order them.

    Both axes are logarithmic, so non-finite and non-positive entries are
    removed rather than silently clipped. Sorting by decreasing false-alarm
    rate matches the reversed x axis.
    """
    far = np.asarray(far, dtype=float).ravel()
    distance = np.asarray(distance, dtype=float).ravel()

    valid = (np.isfinite(far) & np.isfinite(distance)
             & (far > 0.0) & (distance > 0.0))
    far, distance = far[valid], distance[valid]

    if sort_descending:
        order = np.argsort(far)[::-1]
        far, distance = far[order], distance[order]
    return far, distance


def load_evaluation(path):
    """Read one MLGWSC-1 evaluation file as (FAR per month, distance in Mpc).

    The stored false-alarm rate is in Hz; it is converted to events per 30-day
    month, the unit the challenge quotes.
    """
    with h5py.File(path, "r") as handle:
        for name in ("far", "sensitive-distance"):
            if name not in handle:
                raise KeyError(f"'{path}' has no '{name}' dataset. "
                               f"Contains: {list(handle.keys())}")
        far = handle["far"][()] * SECONDS_PER_MONTH
        distance = handle["sensitive-distance"][()]
    return prepare_curve(far, distance)


def load_numpy_curve(path):
    """Read a published curve stored as (FAR per month, distance) columns."""
    data = np.load(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"'{path}' should be a two-column array of "
                         f"(FAR per month, distance); got shape {data.shape}.")
    return prepare_curve(data[:, 0], data[:, 1])


def describe(exc):
    """A readable message for both OSError (errno) and KeyError (text)."""
    if isinstance(exc, OSError):
        return f"{exc.strerror or exc}" if exc.strerror else str(exc)
    return exc.args[0] if exc.args else str(exc)


def gather_references(results_dir):
    """Load every reference pipeline that is present, skipping the rest."""
    learned, conventional = [], []
    if results_dir is None:
        logging.warning("No --results-dir given; drawing the baseline alone.")
        return learned, conventional

    for label, relative in REFERENCE_HDF:
        path = os.path.join(results_dir, relative)
        try:
            learned.append((label,) + load_evaluation(path))
        except (OSError, KeyError) as exc:
            logging.warning("Skipping %s: %s", label, describe(exc))

    for label, relative in REFERENCE_NPY:
        path = os.path.join(results_dir, relative)
        try:
            learned.append((label,) + load_numpy_curve(path))
        except (OSError, ValueError) as exc:
            logging.warning("Skipping %s: %s", label, describe(exc))

    for label, relative, colour, style, width in CONVENTIONAL_HDF:
        path = os.path.join(results_dir, relative)
        try:
            far, distance = load_evaluation(path)
            conventional.append((label, far, distance, colour, style, width))
        except (OSError, KeyError) as exc:
            logging.warning("Skipping %s: %s", label, describe(exc))

    return learned, conventional


# =============================================================================
# Plot
# =============================================================================
def setup_axes(ax, xlim, ylim, title):
    """Log-log axes with the false-alarm rate decreasing to the right."""
    from matplotlib.ticker import LogLocator, NullFormatter

    ax.set_xscale("log")
    ax.set_yscale("log")
    # Reversed: frequent false alarms on the left, the stringent end on the
    # right, so that moving right means demanding more confidence.
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.set_xlabel(r"False-alarm rate (month$^{-1}$)")
    ax.set_ylabel("Sensitive distance (Mpc)")
    if title:
        ax.set_title(title, pad=10)

    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10.0))
        axis.set_minor_locator(LogLocator(base=10.0,
                                          subs=np.arange(2, 10) * 0.1))
        axis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="both", which="both", top=True, right=True)

    ax.grid(True, which="major", linestyle="-", linewidth=0.55,
            color="0.82", alpha=0.75)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4,
            color="0.88", alpha=0.55)


def plot_figure(baseline, learned, conventional, output_stem, xlim, ylim,
                title, baseline_label, legend_loc="below", dpi=400):
    """Draw the dataset-4 comparison and save it as PDF and PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 6.4))

        # Context first, so the baseline is drawn over it.
        for style, (label, far, distance) in zip(GRAY_LINE_STYLES, learned):
            ax.plot(far, distance, label=label, color=CLR_GRAY,
                    linestyle=style, linewidth=1.8, alpha=0.90, zorder=2)

        for label, far, distance, colour, style, width in conventional:
            ax.plot(far, distance, label=label, color=colour, linestyle=style,
                    linewidth=width, zorder=3)

        if baseline is not None:
            far, distance = baseline
            ax.plot(far, distance, label=baseline_label, color=CLR_BASELINE,
                    linestyle="-", linewidth=3.0, zorder=6)

        setup_axes(ax, xlim, ylim, title)

        # The legend goes BELOW the axes by default. With eight entries there
        # is no reliably clear region inside the frame: a baseline that
        # underperforms the published pipelines runs through the lower left,
        # which is exactly where an inside legend would otherwise sit, and
        # matplotlib's "best" placement lands on top of it.
        if legend_loc == "below":
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center",
                       bbox_to_anchor=(0.5, 0.012), ncol=3, frameon=False,
                       handlelength=3.0, columnspacing=1.6,
                       handletextpad=0.7)
            fig.subplots_adjust(left=0.13, right=0.97, top=0.93, bottom=0.26)
        else:
            ax.legend(loc=legend_loc, frameon=False, handlelength=3.0,
                      handletextpad=0.7, labelspacing=0.45)
            fig.tight_layout()
        written = []
        for extension in ("pdf", "png"):
            path = f"{output_stem}.{extension}"
            fig.savefig(path, dpi=dpi, facecolor="white", edgecolor="none")
            written.append(path)
        plt.close(fig)
    return written


def report(baseline, learned, conventional, benchmark=1.0):
    """Print the sensitive distance each pipeline reaches at the benchmark FAR."""
    print(f"\n  Sensitive distance at FAR = {benchmark:g} / month")
    print("  " + "-" * 44)

    def at_benchmark(far, distance):
        # far is descending; np.interp needs ascending x.
        if benchmark < far.min() or benchmark > far.max():
            return None
        return float(np.interp(benchmark, far[::-1], distance[::-1]))

    rows = []
    if baseline is not None:
        rows.append(("baseline", at_benchmark(*baseline)))
    rows.extend((label, at_benchmark(far, distance))
                for label, far, distance in learned)
    rows.extend((label, at_benchmark(far, distance))
                for label, far, distance, *_ in conventional)

    for label, value in rows:
        text = f"{value:8.1f} Mpc" if value is not None else "  not reached"
        print(f"  {label:<18s} {text}")
    if any(value is None for _, value in rows):
        print("\n  'not reached' means the curve does not extend to this "
              "false-alarm\n  rate: the background was too small to resolve it.")


# =============================================================================
# Command-line interface
# =============================================================================
def main():
    parser = ArgumentParser(
        description="Plot sensitive distance versus false-alarm rate on "
                    "MLGWSC-1 dataset 4.")
    parser.add_argument("--baseline", default=None,
                        help="Evaluation file for the baseline transformer. "
                             "Optional: without it the reference pipelines are "
                             "drawn alone, leaving the slot to be filled once "
                             "the result exists.")
    parser.add_argument("--baseline-label", default="Baseline transformer",
                        help="Legend entry for the baseline curve. "
                             "Default: 'Baseline transformer'.")
    parser.add_argument("--results-dir", default=None,
                        help="Directory holding the published pipeline "
                             "results, laid out as <pipeline>/ds4/eval.hdf "
                             "plus the AresGW and SAGE .npy curves.")
    parser.add_argument("-o", "--output", default="sensitive_distance_ds4",
                        help="Output path stem. Default: "
                             "sensitive_distance_ds4.")
    parser.add_argument("--title", default="Dataset 4",
                        help="Panel title. Pass an empty string for none. "
                             "Default: 'Dataset 4'.")
    parser.add_argument("--xlim", type=float, nargs=2, default=(2000.0, 1.0),
                        metavar=("LEFT", "RIGHT"),
                        help="False-alarm rate limits, reversed so the "
                             "stringent end is on the right. "
                             "Default: 2000 1.")
    parser.add_argument("--ylim", type=float, nargs=2, default=(100.0, 2500.0),
                        metavar=("LOW", "HIGH"),
                        help="Sensitive-distance limits in Mpc. Default: "
                             "100 2500. The published pipelines occupy "
                             "300-2500; the lower default leaves room for a "
                             "baseline that falls short of them.")
    parser.add_argument("--benchmark-far", type=float, default=1.0,
                        help="False-alarm rate, per month, at which to report "
                             "sensitive distances. Default: 1.")
    parser.add_argument("--legend-loc", default="below",
                        help="Legend placement: 'below' puts it under the "
                             "axes, where it cannot overlap a curve. Any "
                             "matplotlib location string places it inside "
                             "instead. Default: below.")
    parser.add_argument("--dpi", type=int, default=400,
                        help="Raster resolution. Default: 400.")
    parser.add_argument("--verbose", action="store_true", help="Print progress.")

    args = parser.parse_args()
    logging.basicConfig(
        format="%(levelname)s | %(asctime)s: %(message)s",
        level=logging.INFO if args.verbose else logging.WARN,
        datefmt="%d-%m-%Y %H:%M:%S")

    baseline = None
    if args.baseline:
        try:
            baseline = load_evaluation(args.baseline)
        except (OSError, KeyError) as exc:
            print(f"\nError reading the baseline file: {describe(exc)}",
                  file=sys.stderr)
            sys.exit(1)
        logging.info("Baseline: %i points from %s",
                     len(baseline[0]), args.baseline)
        lowest = baseline[1].min()
        if lowest < args.ylim[0]:
            logging.warning(
                "The baseline reaches %.0f Mpc, below the plotted range "
                "(%.0f Mpc). Lower it with --ylim to show the whole curve.",
                lowest, args.ylim[0])
    else:
        logging.warning("No --baseline given; the baseline slot is empty.")

    learned, conventional = gather_references(args.results_dir)
    if baseline is None and not learned and not conventional:
        print("\nError: nothing to plot. Give --baseline, --results-dir, or "
              "both.", file=sys.stderr)
        sys.exit(1)
    if len(learned) > len(GRAY_LINE_STYLES):
        logging.warning("More reference curves (%i) than distinct line styles "
                        "(%i); some will repeat.",
                        len(learned), len(GRAY_LINE_STYLES))

    parent = os.path.dirname(args.output)
    if parent:
        os.makedirs(parent, exist_ok=True)

    written = plot_figure(baseline, learned, conventional, args.output,
                          tuple(args.xlim), tuple(args.ylim), args.title,
                          args.baseline_label, legend_loc=args.legend_loc,
                          dpi=args.dpi)
    report(baseline, learned, conventional, benchmark=args.benchmark_far)
    print("\nWrote " + ", ".join(written))


if __name__ == "__main__":
    main()
