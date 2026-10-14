"""Plot ROC curves by injected SNR for Figure 1.1.

Scores a held-out test set with a trained checkpoint and draws one curve per
injected network SNR, annotated with the area under each curve.

The chapter uses this figure to make a negative point: the model separates
signals from noise almost perfectly at high SNR, and that tells us very
little about how it will behave as a search. A ROC computed on a balanced
test set cannot reach the false-positive rates a one-per-month search
operates at. Section 1.3 makes the measurement that can.
"""

from argparse import ArgumentParser
import logging
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

# Make the repository root importable so ``gwsearch`` resolves whether or not
# the package has been installed with ``pip install -e .``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gwsearch.coherent_model import MODEL_DEFAULTS, CoherentTransformer, usr

DTYPE = torch.float32

# ``usr`` is imported under its own name above; alias it so the plotting code
# below reads the same way as the rest of the repository.
usr_statistic = usr


def load_checkpoint(path, device):
    """Rebuild the coherent transformer from a training checkpoint.

    The architecture is read from the checkpoint rather than assumed, so a
    file trained with non-default settings still loads correctly.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = ckpt.get("model_config", MODEL_DEFAULTS)
    if config != MODEL_DEFAULTS:
        logging.info("Using architecture stored in checkpoint: %s", config)
    model = CoherentTransformer(**config)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    model.to(device).eval()
    return model


ACCENT = "#D55E00"
MUTED = "#5a5a5a"

STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 10,
    "axes.labelsize": 11,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": False,
    "savefig.bbox": "tight",
}


# =============================================================================
# Data
# =============================================================================
def resolve_group(path, requested):
    """Decide which group to evaluate.

    A held-out test set is normally produced by running generate_training_data.py a
    second time with a different seed, which writes its examples to a group
    called 'training' -- that is simply the default group name, and says
    nothing about how the file will be used. So 'auto' takes the first of
    'test', 'validation', 'training' that exists, rather than assuming the
    name carries meaning.
    """
    with h5py.File(path, "r") as handle:
        available = list(handle.keys())
    if requested != "auto":
        if requested not in available:
            raise KeyError(f"Group '{requested}' not found in {path}. "
                           f"Available: {available}")
        return requested
    for candidate in ("testing", "test", "validation", "training"):
        if candidate in available:
            return candidate
    raise KeyError(f"{path} contains none of 'testing', 'test', 'validation' "
                   f"or 'training'. Available: {available}")


def load_split(path, group):
    """Read the noise and unit-SNR waveform arrays of one group."""
    with h5py.File(path, "r") as handle:
        return handle[group]["noises"][()], handle[group]["waveforms"][()]


# =============================================================================
# Inference
# =============================================================================
@torch.no_grad()
def statistics_for(model, device, noises, waveforms=None, snr=None,
                   batch_size=512, verbose=False, desc="evaluating"):
    """Return the USR statistic for a set of examples.

    If ``waveforms`` is None the noise segments are evaluated as they are.
    Otherwise example ``i`` is ``noises[i] + snr[i] * waveforms[i]``, using the
    same one-to-one pairing and unit-SNR normalisation that
    ``generate_training_data.py`` wrote.

    Returns
    -------
    ndarray of shape (n_examples,)
    """
    model.eval()
    n = len(waveforms) if waveforms is not None else len(noises)
    out = np.empty(n, dtype=np.float64)

    for start in tqdm(range(0, n, batch_size), desc=desc, ascii=True,
                      leave=False, disable=not verbose):
        stop = min(start + batch_size, n)
        batch = torch.from_numpy(np.asarray(noises[start:stop])).to(
            dtype=DTYPE, device=device)
        if waveforms is not None:
            signal = torch.from_numpy(np.asarray(waveforms[start:stop])).to(
                dtype=DTYPE, device=device)
            scale = torch.from_numpy(np.asarray(snr[start:stop])).to(
                dtype=DTYPE, device=device).view(-1, 1, 1)
            batch = batch + scale * signal
        out[start:stop] = usr_statistic(model(batch)).double().cpu().numpy()
    return out


# =============================================================================
# ROC
# =============================================================================
def roc_curve(signal_statistics, noise_statistics):
    """Empirical ROC, computed directly rather than via an external library.

    For a threshold t, the true-positive rate is the fraction of signal
    examples with statistic >= t and the false-positive rate is the same
    fraction of the noise examples. Evaluating this at every observed value
    gives the full curve.

    Returns
    -------
    fpr, tpr : ndarray
        Both ascending, starting at (0, 0) and ending at (1, 1).
    """
    signal_sorted = np.sort(np.asarray(signal_statistics, dtype=np.float64))
    noise_sorted = np.sort(np.asarray(noise_statistics, dtype=np.float64))
    thresholds = np.unique(np.concatenate([signal_sorted, noise_sorted]))[::-1]

    # searchsorted(..., 'left') counts entries strictly below the threshold.
    tpr = 1.0 - np.searchsorted(signal_sorted, thresholds, side="left") / len(signal_sorted)
    fpr = 1.0 - np.searchsorted(noise_sorted, thresholds, side="left") / len(noise_sorted)

    fpr = np.concatenate([[0.0], fpr, [1.0]])
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    return fpr, tpr


def area_under_curve(fpr, tpr):
    """Trapezoidal AUC. Inputs must be sorted by increasing fpr."""
    order = np.argsort(fpr, kind="stable")
    return float(np.trapezoid(tpr[order], fpr[order])) if hasattr(np, "trapezoid") \
        else float(np.trapz(tpr[order], fpr[order]))


def tpr_at(fpr, tpr, target_fpr):
    """True-positive rate at a given false-positive rate.

    Returns NaN if the requested rate lies below the resolution of the test
    set, rather than silently extrapolating into a regime it cannot measure.
    """
    positive = fpr[fpr > 0]
    if positive.size == 0 or target_fpr < positive.min():
        return float("nan")
    return float(np.interp(target_fpr, fpr, tpr))


# =============================================================================
# Plot
# =============================================================================
def plot_roc(curves, n_noise, required_fpr, far_target, output_stem,
             linear_only=False, dpi=300):
    """Draw the ROC figure and save it as PDF and PNG.

    Parameters
    ----------
    curves : list of (label, fpr, tpr, auc)
    n_noise : int
        Number of pure-noise examples, which sets the resolution limit 1/N.
    required_fpr : float
        False-positive rate implied by the target false-alarm rate.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    resolution_limit = 1.0 / n_noise
    single = len(curves) == 1
    colours = ([TRAIN_COLOUR] if single else
               plt.get_cmap("viridis")(np.linspace(0.08, 0.86, len(curves))))

    with plt.rc_context(STYLE):
        n_panels = 1 if linear_only else 2
        fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels + 0.4, 3.9),
                                 squeeze=False)
        axes = list(axes[0])

        # ---- (a) the conventional view -----------------------------------
        ax = axes[0]
        for (label, fpr, tpr, auc), colour in zip(curves, colours):
            ax.plot(fpr, tpr, lw=1.7, color=colour,
                    label=f"{label} (AUC {auc:.4f})" if label else
                          f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], ls=(0, (4, 3)), lw=1.0, color="0.55",
                label="chance")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.005)
        ax.set_xlabel("False-positive rate")
        ax.set_ylabel("True-positive rate")
        ax.set_title("(a) Conventional axes", size=10, pad=6)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(loc="lower right")

        if linear_only:
            axes[-1].set_box_aspect(1)
        else:
            # ---- (b) the axes a search actually cares about ---------------
            ax = axes[1]
            lower = min(required_fpr, resolution_limit) / 5.0
            for (label, fpr, tpr, auc), colour in zip(curves, colours):
                mask = fpr > 0
                ax.plot(fpr[mask], tpr[mask], lw=1.7, color=colour, label=label)

            # Everything left of 1/N is an artefact of a finite test set.
            ax.axvspan(lower, resolution_limit, color="0.90", zorder=0)
            ax.axvline(resolution_limit, color=MUTED, ls=":", lw=1.2)
            ax.axvline(required_fpr, color=ACCENT, ls="--", lw=1.4)

            ax.annotate(f"test set cannot\nresolve below\n1/N = {resolution_limit:.1e}",
                        xy=(resolution_limit, 0.55), xytext=(-7, 0),
                        textcoords="offset points", size=7.5, color=MUTED,
                        ha="right", va="center")
            ax.annotate(f"FAR = {far_target:g}/month\nneeds {required_fpr:.1e}",
                        xy=(required_fpr, 0.14), xytext=(6, 0),
                        textcoords="offset points", size=7.5, color=ACCENT,
                        ha="left", va="center")

            ax.set_xscale("log")
            ax.set_xlim(lower, 1.0)
            ax.set_ylim(0, 1.005)
            ax.set_xlabel("False-positive rate")
            ax.set_title("(b) Logarithmic false-positive axis", size=10, pad=6)
            ax.grid(alpha=0.25, lw=0.5, which="both")
            if not single:
                ax.legend(loc="lower right")

        fig.tight_layout()
        written = []
        for extension in ("pdf", "png"):
            path = f"{output_stem}.{extension}"
            fig.savefig(path, dpi=dpi)
            written.append(path)
        plt.close(fig)
    return written


# =============================================================================
# Command-line interface
# =============================================================================
def main():
    parser = ArgumentParser(
        description="Plot ROC curves by injected SNR for a trained checkpoint.")
    parser.add_argument("-w", "--weights", type=str, required=True,
                        help="Checkpoint written by train_coherent_pregen.py, "
                             "typically best_state_dict.pt.")
    parser.add_argument("-d", "--dataset-file", type=str, required=True,
                        help="Dataset file from generate_training_data.py.")
    parser.add_argument("-o", "--output", type=str, default="roc",
                        help="Output path stem. Default: roc.")
    parser.add_argument("-g", "--group", type=str, default="auto",
                        help="Which group of the file to evaluate. Default: "
                             "auto, which takes the first of 'testing', "
                             "'test', 'validation', 'training' that is "
                             "present. A "
                             "held-out test file generated on its own will "
                             "contain a single group named 'training'; that is "
                             "just the default group name and is evaluated "
                             "normally. The file must of course be one the "
                             "checkpoint never saw.")
    parser.add_argument("-s", "--snr", type=float, nargs=2, default=(5.0, 15.0),
                        metavar=("LOW", "HIGH"),
                        help="SNR range for injected signals, drawn "
                             "reproducibly. Default: 5 15. Ignored when "
                             "--snr-bands is given.")
    parser.add_argument("--snr-bands", type=float, nargs="+", default=None,
                        metavar="SNR",
                        help="Draw one ROC curve per fixed SNR, e.g. "
                             "--snr-bands 6 8 10 15 20.")
    parser.add_argument("--far-target", type=float, default=1.0,
                        help="Target false-alarm rate per month, used to mark "
                             "the operating point. Default: 1.")
    parser.add_argument("--stride", type=float, default=0.1,
                        help="Analysis stride in seconds, used to convert the "
                             "target FAR into a false-positive rate. "
                             "Default: 0.1.")
    parser.add_argument("--linear-only", action="store_true",
                        help="Draw only the conventional linear-axis panel.")
    parser.add_argument("--save-statistics", action="store_true",
                        help="Also write <stem>_stats.npz with the raw "
                             "per-example statistics.")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Inference batch size. Default: 512.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for inference, e.g. 'cuda'. Default: cpu.")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Seed for the SNR draw. Default: 2026.")
    parser.add_argument("--verbose", action="store_true", help="Print progress.")

    args = parser.parse_args()
    logging.basicConfig(
        format="%(levelname)s | %(asctime)s: %(message)s",
        level=logging.INFO if args.verbose else logging.WARN,
        datefmt="%d-%m-%Y %H:%M:%S")

    try:
        group = resolve_group(args.dataset_file, args.group)
        noises, waveforms = load_split(args.dataset_file, group)
    except (KeyError, OSError) as exc:
        message = exc.args[0] if exc.args else exc
        print(f"\nError: {message}", file=sys.stderr)
        sys.exit(1)

    # The script cannot tell whether this file was used for training, so it
    # states what it is using and leaves the judgement to the caller.
    logging.info("Evaluating group '%s' of %s. This must be data the "
                 "checkpoint was never trained on.", group, args.dataset_file)

    n_injections = len(waveforms)
    n_noise = len(noises) - n_injections
    if n_noise < 1:
        print("\nError: this group contains no pure-noise examples, so a "
              "false-positive rate cannot be measured.", file=sys.stderr)
        sys.exit(1)
    logging.info("Evaluating %i injections against %i pure-noise examples.",
                 n_injections, n_noise)

    device = torch.device(args.device)
    model = load_checkpoint(args.weights, device)
    logging.info("Loaded %s", args.weights)

    # Pure-noise statistics are independent of the injected SNR, so they are
    # computed once and reused by every curve.
    noise_statistics = statistics_for(
        model, device, noises[n_injections:], batch_size=args.batch_size,
        verbose=args.verbose, desc="noise")

    rng = np.random.default_rng(args.seed)
    signal_sets = {}
    if args.snr_bands:
        for value in args.snr_bands:
            signal_sets[f"SNR = {value:g}"] = np.full(n_injections, value)
    else:
        signal_sets[""] = rng.uniform(args.snr[0], args.snr[1], n_injections)

    curves, stats_to_save = [], {"noise": noise_statistics}
    for label, snr_values in signal_sets.items():
        signal_statistics = statistics_for(
            model, device, noises[:n_injections], waveforms, snr_values,
            batch_size=args.batch_size, verbose=args.verbose,
            desc=label or "signal")
        fpr, tpr = roc_curve(signal_statistics, noise_statistics)
        auc = area_under_curve(fpr, tpr)
        curves.append((label, fpr, tpr, auc))
        stats_to_save[f"signal_{label or 'all'}"] = signal_statistics
        logging.info("%-12s AUC %.5f | TPR at FPR 1e-2: %.3f, 1e-3: %.3f",
                     label or "combined", auc,
                     tpr_at(fpr, tpr, 1e-2), tpr_at(fpr, tpr, 1e-3))

    # Trials in one month at the given stride -> the tolerable FPR.
    trials_per_month = SECONDS_PER_MONTH / args.stride
    required_fpr = args.far_target / trials_per_month
    resolution_limit = 1.0 / n_noise

    parent = os.path.dirname(args.output)
    if parent:
        os.makedirs(parent, exist_ok=True)

    written = plot_roc(curves, n_noise, required_fpr, args.far_target,
                       args.output, linear_only=args.linear_only)
    print("Wrote " + ", ".join(written))

    if args.save_statistics:
        path = f"{args.output}_stats.npz"
        np.savez_compressed(path, **stats_to_save)
        print(f"Wrote {path}")

    print(f"\n  AUC                        {curves[0][3]:.5f}"
          + ("  (first curve)" if len(curves) > 1 else ""))
    print(f"  Pure-noise examples        {n_noise:,}")
    print(f"  Smallest resolvable FPR    {resolution_limit:.2e}  (= 1/N)")
    print(f"  FPR needed for {args.far_target:g}/month     {required_fpr:.2e}"
          f"  ({trials_per_month:.2e} trials/month at {args.stride:g} s stride)")
    shortfall = resolution_limit / required_fpr
    if shortfall > 1:
        print(f"\n  This test set is short of the required regime by a factor "
              f"of {shortfall:.0f}\n  ({np.log10(shortfall):.1f} orders of "
              f"magnitude). The ROC cannot speak to it.")


if __name__ == "__main__":
    main()
