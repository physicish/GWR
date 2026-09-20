#!/usr/bin/env python
"""
Visualise and sanity-check a dataset written by ``generate_dataset.py``.

Two figures are produced.

1. ``<stem>_examples.png`` -- what the network actually sees.
   Top row:    whitened detector noise.
   Middle row: the stored whitened signal, rescaled to the chosen SNR.
   Bottom row: their sum, i.e. one training example.
   The signal is invisible by eye at realistic SNR; this is the whole point of
   the detection problem and worth showing to readers new to the field.

2. ``<stem>_diagnostics.png`` -- four checks that the generation worked.
   (a) Whitened noise amplitude distribution, against a unit Gaussian.
       Whitening leaves noise Gaussian in shape with zero mean. Note the
       amplitude is NOT unity: pycbc's convention gives sqrt(f_s/2), about 32
       at 2048 Hz, so the distribution is normalised by its own sigma here.
   (b) Amplitude spectral density of the whitened noise. It should be FLAT
       above the 20 Hz cutoff and suppressed below it. A sloped spectrum means
       the whitening PSD did not match the noise.
   (c) Norm of each stored waveform. All entries are normalised to network
       SNR = 1 before being written, so this should be tightly clustered.
   (d) Merger position within the segment, estimated from the signal envelope.
       It should be spread over the intended window, NOT concentrated at one
       sample. A network trained on signals that always peak in the same place
       learns that position instead of the signal morphology.

This script deliberately depends only on numpy, h5py and matplotlib -- NOT on
pycbc -- so it can be run on a laptop to inspect files produced on a cluster.

------------------------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------------------------
    # Default: training group, injected at SNR 15
    python plot_dataset.py data_gaussian.h5

    # Choose the output name, the SNR and which example to show
    python plot_dataset.py data_gaussian.h5 -o figures/gaussian --snr 12 --index 7

    # Inspect the validation split instead. Note this split exists only if
    # generate_dataset.py was run with --validation-samples; by default the
    # generated file contains a training split only.
    python plot_dataset.py data_gaussian.h5 --group validation

    # Print the file contents without plotting
    python plot_dataset.py data_gaussian.h5 --summary-only
------------------------------------------------------------------------------
"""

from argparse import ArgumentParser
import os
import sys

import h5py
import matplotlib
matplotlib.use("Agg")           # render to file; no interactive display needed
import matplotlib.pyplot as plt
import numpy as np


GROUP_PREFERENCE = ("testing", "test", "validation", "training")


def resolve_group(path, requested):
    """Pick which group to read, so that any generator's output just works.

    generate_test_data.py writes 'testing'; generate_dataset.py writes
    'training' and optionally 'validation'.
    """
    with h5py.File(path, "r") as handle:
        available = list(handle.keys())
    if requested != "auto":
        if requested not in available:
            raise KeyError(f"Group '{requested}' not found in {path}. "
                           f"Available: {available}")
        return requested
    for candidate in GROUP_PREFERENCE:
        if candidate in available:
            return candidate
    raise KeyError(f"{path} contains no recognised group. "
                   f"Available: {available}")


def load_split(path, group):
    """Read one split from the HDF5 file, plus the file-level metadata.

    Returns
    -------
    noises : ndarray, shape (n_samples, n_detectors, sample_length)
    waveforms : ndarray, shape (n_injections, n_detectors, sample_length)
    attrs : dict
        File-level attributes (sample rate, detector names, seed, ...).
    """
    with h5py.File(path, "r") as handle:
        if group not in handle:
            raise KeyError(
                f"Group '{group}' not found in {path}. "
                f"Available: {list(handle.keys())}")
        noises = handle[group]["noises"][()]
        waveforms = handle[group]["waveforms"][()]
        attrs = dict(handle.attrs)
    return noises, waveforms, attrs


def print_summary(path):
    """Print the structure and provenance of the file."""
    with h5py.File(path, "r") as handle:
        print(f"\nFile: {path}")
        print("-" * 70)
        print("File attributes:")
        for key in sorted(handle.attrs):
            value = handle.attrs[key]
            text = str(value)
            if len(text) > 100:
                text = text[:97] + "..."
            print(f"  {key:32s} {text}")
        for group_name in handle:
            group = handle[group_name]
            print(f"\nGroup '{group_name}':")
            for ds_name in group:
                ds = group[ds_name]
                print(f"  {ds_name:12s} shape={ds.shape} dtype={ds.dtype}")
        print("-" * 70)


def detector_names(attrs, n_detectors):
    """Detector labels from the file attributes, with a safe fallback."""
    names = attrs.get("detectors")
    if names is None:
        return [f"Detector {i + 1}" for i in range(n_detectors)]
    return [n.decode() if isinstance(n, bytes) else str(n) for n in names]


def plot_examples(noises, waveforms, attrs, index, snr, outpath):
    """Figure 1: noise, signal and their sum, for each detector."""
    sample_rate = int(attrs.get("sample_rate", 2048))
    n_detectors = noises.shape[1]
    names = detector_names(attrs, n_detectors)

    noise = noises[index]
    signal = waveforms[index] * snr          # stored at network SNR = 1
    combined = noise + signal
    times = np.arange(noise.shape[-1]) / sample_rate

    fig, axes = plt.subplots(3, n_detectors, figsize=(6 * n_detectors, 8),
                             sharex=True, squeeze=False)
    rows = [
        (noise, "Whitened noise", "tab:grey"),
        (signal, f"Signal only (network SNR = {snr:g})", "tab:orange"),
        (combined, "Noise + signal (network input)", "tab:blue"),
    ]
    for row, (data, label, colour) in enumerate(rows):
        for col in range(n_detectors):
            ax = axes[row][col]
            ax.plot(times, data[col], lw=0.7, color=colour)
            if row == 0:
                ax.set_title(names[col], fontsize=12, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{label}\n\nwhitened strain", fontsize=9)
            if row == 2:
                ax.set_xlabel("Time [s]")
            ax.grid(alpha=0.25)
    # Put the noise and the combined panels on a common scale, so the reader can
    # see directly how little the signal changes the data.
    ylim = np.max(np.abs(combined)) * 1.1
    for col in range(n_detectors):
        axes[0][col].set_ylim(-ylim, ylim)
        axes[2][col].set_ylim(-ylim, ylim)

    fig.suptitle(
        f"Dataset example #{index} - injected at network SNR {snr:g}",
        fontsize=13)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {outpath}")


def plot_diagnostics(noises, waveforms, attrs, outpath, max_samples=400):
    """Figure 2: four checks that whitening and normalisation behaved."""
    sample_rate = int(attrs.get("sample_rate", 2048))
    f_low = float(attrs.get("low_frequency_cutoff", 20.0))
    n_used = min(max_samples, noises.shape[0])
    noise_subset = noises[:n_used]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (a) Amplitude distribution of the whitened noise -----------------------
    ax = axes[0][0]
    flat = noise_subset.ravel()
    sigma = flat.std()
    # NOTE: pycbc's whitening does not return unit variance. It returns a
    # standard deviation of sqrt(f_s/2), about 32 at 2048 Hz, which is the
    # origin of the factor 32 hard-coded in the original CASTOR code. What
    # matters is that the distribution is Gaussian in SHAPE and that the same
    # convention is used at training and application time, so the amplitude is
    # normalised out here and reported in the title instead.
    ax.hist(flat / sigma, bins=200, density=True, color="tab:grey",
            alpha=0.8, label="whitened noise")
    grid = np.linspace(-5, 5, 400)
    ax.plot(grid, np.exp(-0.5 * grid ** 2) / np.sqrt(2 * np.pi),
            "r--", lw=1.5, label="Gaussian")
    ax.set_xlim(-5, 5)
    ax.set_xlabel(r"Whitened strain / $\sigma$")
    ax.set_ylabel("Density")
    ax.set_title(f"(a) Noise amplitude\nmean={flat.mean():+.3g}, "
                 f"std={sigma:.2f} "
                 f"(expect $\\sqrt{{f_s/2}}$ = {np.sqrt(sample_rate / 2):.1f})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # (b) Amplitude spectral density of the whitened noise -------------------
    # White noise of standard deviation sigma has a one-sided ASD of
    # sigma*sqrt(2/f_s), drawn as the dashed line. Flatness above f_low is what
    # is being checked here, not the absolute level.
    ax = axes[0][1]
    spectra = np.fft.rfft(noise_subset, axis=-1)
    n_time = noise_subset.shape[-1]
    psd = 2.0 * np.abs(spectra) ** 2 / (n_time * sample_rate)
    asd = np.sqrt(psd.mean(axis=(0, 1)))
    freqs = np.fft.rfftfreq(n_time, d=1.0 / sample_rate)
    ax.loglog(freqs[1:], asd[1:], lw=0.8, color="tab:blue")
    ax.axhline(sigma * np.sqrt(2.0 / sample_rate), color="r", ls="--", lw=1.5,
               label=r"white at the measured $\sigma$")
    ax.axvline(f_low, color="k", ls=":", lw=1.5,
               label=f"cutoff {f_low:g} Hz")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("ASD [1/$\\sqrt{\\rm Hz}$]")
    ax.set_title("(b) Whitened noise spectrum\n(should be flat above the cutoff)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")

    # (c) Norm of the stored waveforms ---------------------------------------
    # Every waveform was divided by its own optimal network SNR before being
    # written, so these should all be close to the same value.
    ax = axes[1][0]
    norms = np.sqrt(np.sum(waveforms.astype(np.float64) ** 2, axis=(1, 2)))
    median = np.median(norms)
    fractional_spread = norms.std() / max(median, 1e-30)
    if fractional_spread < 1e-6 and abs(median - sigma) < 0.5:
        # Every norm is identical to floating-point precision. A histogram here
        # would just zoom into rounding noise, so state the value instead.
        ax.axvline(median, color="tab:orange", lw=3, label=f"all = {median:.4f}")
        ax.set_xlim(median - 0.5, median + 0.5)
        ax.text(0.5, 0.55, "identical to machine precision",
                transform=ax.transAxes, ha="center", fontsize=9)
    else:
        ax.hist(norms, bins=60, color="tab:orange", alpha=0.85)
        ax.axvline(median, color="k", ls="--", lw=1.5,
                   label=f"median = {median:.3f}")
    # A signal of matched-filter SNR rho in noise of per-sample standard
    # deviation sigma satisfies ||rho*w|| = rho*sigma, so a waveform stored at
    # unit network SNR has norm sigma -- NOT 1. The target therefore tracks the
    # measured noise amplitude.
    ax.axvline(sigma, color="r", ls=":", lw=1.5,
               label=f"target = $\\sigma$ = {sigma:.1f}")
    ax.set_xlabel(r"$\|w\|$ over both detectors")
    ax.set_ylabel("Count")
    ax.set_title("(c) Stored waveform norm\n(all normalised to network SNR = 1)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # (d) Merger position within the segment ---------------------------------
    # Estimated as the time of peak absolute amplitude, summed over detectors.
    ax = axes[1][1]
    envelope = np.sum(np.abs(waveforms.astype(np.float64)), axis=1)
    peak_times = np.argmax(envelope, axis=-1) / sample_rate
    ax.hist(peak_times, bins=60, color="tab:green", alpha=0.85)
    ax.set_xlabel("Time of peak amplitude [s]")
    ax.set_ylabel("Count")
    placement = attrs.get("merger_placement_seconds", "")
    if isinstance(placement, bytes):
        placement = placement.decode()
    ax.set_title(f"(d) Merger position\nintended window: {placement}")
    ax.grid(alpha=0.25)

    noise_type = attrs.get("noise_type", "unknown")
    if isinstance(noise_type, bytes):
        noise_type = noise_type.decode()
    approximant = attrs.get("approximant", "unknown")
    if isinstance(approximant, bytes):
        approximant = approximant.decode()
    fig.suptitle(f"Dataset diagnostics - {noise_type} noise, {approximant} "
                 f"waveforms ({n_used} noise segments)", fontsize=13)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {outpath}")


def main():
    parser = ArgumentParser(
        description="Plot and sanity-check a dataset from generate_dataset.py.")
    parser.add_argument("input_file", type=str,
                        help="HDF5 file written by generate_dataset.py.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output path stem. Two PNGs are written: "
                             "<stem>_examples.png and <stem>_diagnostics.png. "
                             "Default: the input filename without its extension.")
    parser.add_argument("-g", "--group", type=str, default="auto",
                        help="Which split to plot. Default: auto, which takes "
                             "the first of 'testing', 'test', 'validation', "
                             "'training' that the file contains.")
    parser.add_argument("--snr", type=float, default=15.0,
                        help="Network SNR at which to inject the example "
                             "signal. Default: 15.")
    parser.add_argument("-i", "--index", type=int, default=0,
                        help="Index of the example to plot. Default: 0.")
    parser.add_argument("--max-samples", type=int, default=400,
                        help="Number of noise segments used for the spectrum "
                             "and histogram diagnostics. Default: 400.")
    parser.add_argument("--summary-only", action="store_true",
                        help="Print the file contents and exit.")

    args = parser.parse_args()

    print_summary(args.input_file)
    if args.summary_only:
        return

    try:
        group = resolve_group(args.input_file, args.group)
        noises, waveforms, attrs = load_split(args.input_file, group)
    except KeyError as exc:
        # Most commonly: asking for 'validation' in a file generated without
        # --validation-samples. Report it plainly rather than as a traceback.
        print(f"\nError: {exc.args[0]}", file=sys.stderr)
        if args.group == "validation":
            print("Hint: regenerate with "
                  "'--validation-samples N_INJ N_NOISE' to create this split.",
                  file=sys.stderr)
        sys.exit(1)

    if waveforms.shape[0] == 0:
        raise ValueError(f"Group '{args.group}' contains no injections to plot.")
    if not 0 <= args.index < waveforms.shape[0]:
        raise IndexError(
            f"--index {args.index} is out of range; this group has "
            f"{waveforms.shape[0]} injections.")

    stem = args.output
    if stem is None:
        stem = os.path.splitext(args.input_file)[0]
    parent = os.path.dirname(stem)
    if parent:
        os.makedirs(parent, exist_ok=True)

    plot_examples(noises, waveforms, attrs, args.index, args.snr,
                  f"{stem}_examples.png")
    plot_diagnostics(noises, waveforms, attrs, f"{stem}_diagnostics.png",
                     max_samples=args.max_samples)


if __name__ == "__main__":
    main()
