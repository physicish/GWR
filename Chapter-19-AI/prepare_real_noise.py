#!/usr/bin/env python
"""
Prepare real LIGO detector noise for training the baseline transformer.

``generate_dataset.py`` can inject simulated signals into either simulated
Gaussian noise or *real* detector noise. This script produces the real-noise
file that its ``--real-noise-file`` option expects, in two stages.

------------------------------------------------------------------------------
THE TWO STAGES
------------------------------------------------------------------------------
Stage 1, ``split``
    Take the raw MLGWSC-1 noise file (long, continuous strain segments, one
    HDF5 group per detector) and divide it into TWO files along segment
    boundaries.

    This exists for one reason, and it is a methodological one. If the same
    stretch of detector noise appears in both training and testing, the
    measured performance is inflated: the network can recognise noise it has
    already seen, and the apparent sensitivity is not a prediction of how the
    search will behave on new data. Splitting at the *segment* level -- not at
    the level of individual training samples -- guarantees that no second of
    strain is shared. Stage 2 below cuts overlapping slices out of each
    segment, so a split made after slicing would leak badly.

Stage 2, ``slice``
    Take one of those files, whiten each segment, cut it into fixed-length
    samples, and record the noise power spectral density (PSD) of each segment.

    The PSDs must be kept because ``generate_dataset.py`` whitens the injected
    waveform with the PSD of the *particular* noise segment it is being added
    to. Using a mismatched PSD would leave the signal incorrectly normalised
    and its stated SNR would be wrong.

------------------------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------------------------
Stage 1 needs only numpy, h5py and tqdm -- it can be run on a laptop. Stage 2
additionally needs pycbc, which is imported only when it is actually used.

    # Stage 1: hold out roughly 11.6 days (1e6 s) of noise for training,
    # and keep the remainder as a disjoint test set.
    python prepare_real_noise.py split real_noise_file.hdf \
        train_noise.hdf test_noise.hdf --duration 1e6 --verbose

    # Stage 2: whiten and slice the training noise into 1 s samples.
    python prepare_real_noise.py slice train_noise.hdf \
        -o sliced_train_noise.hdf --detectors 2 --verbose

    # Split the output across several files if it would be inconveniently large
    python prepare_real_noise.py slice train_noise.hdf \
        -o sliced_train_noise.hdf --chunk-size 20000 --verbose
    #   -> sliced_train_noise_0000.hdf, sliced_train_noise_0001.hdf, ...

    # Stage 3 (a different script): inject signals into the sliced noise
    python generate_dataset.py -o data_o3.h5 \
        --real-noise-file sliced_train_noise.hdf --verbose

------------------------------------------------------------------------------
Adapted for this chapter from ``split_noise_file.py``, ``slice_real_noise.py``
and ``apply.py`` of the MLGWSC-1 submission by Ondrej Zelenka
(https://github.com/ondrzel/ml-gw-search), Apache License 2.0.
Copyright 2022 Ondrej Zelenka. Modifications copyright 2026 Chayan Chatterjee.
------------------------------------------------------------------------------
"""

from argparse import ArgumentParser
import logging
import os
import os.path
import sys

import h5py
import numpy as np
from tqdm import tqdm


# =============================================================================
# Fixed configuration (matches generate_dataset.py -- keep the two in step)
# =============================================================================
DELTA_T = 1.0 / 2048.0
SAMPLE_RATE = int(round(1.0 / DELTA_T))

# Length of the Welch segments used to estimate the PSD of each noise segment.
WELCH_SEGMENT_DURATION = 0.5

# Whitening filter length. After whitening, MAX_FILTER_DURATION/2 seconds are
# discarded from each end of a segment, where the truncated filter has wrapped
# around and corrupted the data.
MAX_FILTER_DURATION = 0.25

LOW_FREQUENCY_CUTOFF = 20.0

# Default distance between the start of consecutive slices. With the default
# 1 s slice length this means consecutive samples overlap by 50%, which yields
# more training data from a fixed amount of strain.
#
# NOTE: overlapping slices are strongly correlated with one another. They are
# acceptable as *training* data, but they are the reason stage 1 must split the
# noise before stage 2 slices it. Splitting a set of overlapping slices would
# put near-duplicate samples on both sides of the train/test boundary.
DEFAULT_STEP_SIZE = 0.5


# =============================================================================
# Small helpers
# =============================================================================
def copy_attrs(in_obj, out_obj):
    """Copy every HDF5 attribute from one object to another."""
    for key, attr in in_obj.attrs.items():
        out_obj.attrs[key] = attr


def dataset_write_kwargs(compress):
    """h5py creation options: gzip level 9 with the shuffle filter, or none."""
    if compress:
        return {"compression": "gzip", "compression_opts": 9, "shuffle": True}
    return {}


def configure_logging(verbose, debug):
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARN
    logging.basicConfig(format="%(levelname)s | %(asctime)s: %(message)s",
                        level=level, datefmt="%d-%m-%Y %H:%M:%S")


def check_output_path(path, force):
    if os.path.isfile(path) and not force:
        raise RuntimeError(f"Output file '{path}' exists. Use --force to overwrite.")


# =============================================================================
# STAGE 1 -- split the raw noise file into two disjoint files
# =============================================================================
def split_noise_file(inputfile, outputfiles, duration, seed, compress=False,
                     force=False, verbose=False):
    """Divide a raw noise file into two files with no shared strain.

    Segments are shuffled and then assigned to the first output file until its
    accumulated live time exceeds ``duration``; everything else goes to the
    second. Because the unit of assignment is a whole segment, the two outputs
    are guaranteed to contain disjoint stretches of detector data.

    Parameters
    ----------
    inputfile : str
        Raw MLGWSC-1 noise file: one HDF5 group per detector, each containing
        one dataset per continuous segment, with a ``delta_t`` attribute.
    outputfiles : (str, str)
        Destination paths. The first receives at least ``duration`` seconds.
    duration : float
        Minimum live time, in seconds, for the first output file.
    seed : int
        Seed for the shuffle, so the split is reproducible. The original code
        used an unseeded generator, which made the split impossible to repeat.
    """
    for path in outputfiles:
        check_output_path(path, force)

    rng = np.random.default_rng(seed)

    with h5py.File(inputfile, "r") as infile, \
            h5py.File(outputfiles[0], "w") as outfile1, \
            h5py.File(outputfiles[1], "w") as outfile2:

        total_datasets = sum(len(grp) for grp in infile.values())
        # Segment names and durations are read from the first detector group;
        # every detector is assumed to hold the same set of segments.
        reference_group = next(iter(infile.values()))
        segment_keys = list(reference_group.keys())
        durations = [reference_group[key].attrs["delta_t"] * len(reference_group[key])
                     for key in segment_keys]
        total_duration = float(np.sum(durations))
        logging.info("Input contains %i segments totalling %.1f s (%.2f days)",
                     len(segment_keys), total_duration, total_duration / 86400.0)

        if duration >= total_duration:
            raise ValueError(
                f"Requested duration {duration:.4g} s for the first output file "
                f"is not less than the total available {total_duration:.4g} s; "
                f"the second file would be empty.")

        # Shuffle, then fill the first file until it is long enough.
        keys_first = set()
        accumulated = 0.0
        for index in rng.permutation(len(segment_keys)):
            keys_first.add(segment_keys[index])
            accumulated += durations[index]
            if accumulated > duration:
                break
        logging.info("First file: %i segments, %.1f s (%.2f days). "
                     "Second file: %i segments, %.1f s (%.2f days).",
                     len(keys_first), accumulated, accumulated / 86400.0,
                     len(segment_keys) - len(keys_first),
                     total_duration - accumulated,
                     (total_duration - accumulated) / 86400.0)

        write_kwargs = dataset_write_kwargs(compress)
        copy_attrs(infile, outfile1)
        copy_attrs(infile, outfile2)

        with tqdm(desc="Splitting segments", disable=not verbose, ascii=True,
                  total=total_datasets) as pbar:
            for detector_name, in_detector_group in infile.items():
                out_group1 = outfile1.create_group(detector_name)
                out_group2 = outfile2.create_group(detector_name)
                copy_attrs(in_detector_group, out_group1)
                copy_attrs(in_detector_group, out_group2)
                for segment_name, in_segment in in_detector_group.items():
                    target = out_group1 if segment_name in keys_first else out_group2
                    out_segment = target.create_dataset(
                        segment_name, data=in_segment[()], **write_kwargs)
                    copy_attrs(in_segment, out_segment)
                    pbar.update(1)

    logging.info("Wrote %s and %s", *outputfiles)


# =============================================================================
# STAGE 2 -- whiten and slice
# =============================================================================
def regularize_psd(psd, low_frequency_cutoff, inf_value=np.inf):
    """Set the PSD to infinity below the analysis cutoff.

    Dividing by an infinite PSD sends those bins to zero, cleanly removing the
    enormous low-frequency seismic noise instead of letting it dominate.
    Identical to the function of the same name in ``generate_dataset.py``.
    """
    import pycbc.types
    values = np.asarray(psd, dtype=np.float64).copy()
    values[np.asarray(psd.sample_frequencies) <= low_frequency_cutoff] = inf_value
    return pycbc.types.FrequencySeries(values, delta_f=psd.delta_f, copy=False)


def whiten_segment(strain, low_frequency_cutoff, delta_t=DELTA_T,
                   welch_segment_duration=WELCH_SEGMENT_DURATION,
                   max_filter_duration=MAX_FILTER_DURATION):
    """Whiten one detector's strain, estimating the PSD from the data itself.

    This differs from the whitening in ``generate_dataset.py`` in one important
    respect: there, the PSD is supplied; here it is *measured* from the segment
    being whitened, by Welch averaging. Real detector noise drifts, so each
    segment gets its own PSD.

    Returns
    -------
    white : ndarray
        Whitened strain, shorter than the input by ``max_filter_duration``
        (half removed from each end, where the filter has wrapped around).
    psd : pycbc.types.FrequencySeries
        The estimated PSD, *before* interpolation and truncation. This is the
        one stored in the output file and later reused to whiten injected
        signals consistently with this noise.
    """
    import pycbc.psd
    import pycbc.types

    colored = pycbc.types.TimeSeries(np.asarray(strain), delta_t=delta_t)
    estimated_psd = colored.psd(welch_segment_duration)

    interpolated = pycbc.psd.interpolate(estimated_psd, colored.delta_f)
    filter_length = int(max_filter_duration * colored.sample_rate)
    truncated = pycbc.psd.inverse_spectrum_truncation(
        interpolated,
        max_filter_len=filter_length,
        low_frequency_cutoff=low_frequency_cutoff,
        trunc_method="hann",
    )
    white = (colored.to_frequencyseries() * (1.0 / truncated) ** 0.5).to_timeseries()
    white = white.numpy()[filter_length // 2: len(colored) - filter_length // 2]
    return white, estimated_psd


class RealNoiseSlicer:
    """Iterate over fixed-length whitened noise slices from a noise file.

    Iterating yields ``(slice_array, psd_index)`` where ``slice_array`` has
    shape ``(n_detectors, slice_length)`` and ``psd_index`` identifies which
    stored PSD describes the segment the slice came from.

    Segments are visited in a shuffled order so that consecutive slices written
    to disk are not all from the same stretch of time.
    """

    def __init__(self, fpath, detector_names, slice_length, step_size, rng,
                 low_frequency_cutoff=LOW_FREQUENCY_CUTOFF,
                 welch_segment_duration=WELCH_SEGMENT_DURATION,
                 max_filter_duration=MAX_FILTER_DURATION, verbose=False):
        self.fpath = fpath
        self.detectors = list(detector_names)
        self.slice_length = slice_length
        self.step_size = step_size
        self.rng = rng
        self.low_frequency_cutoff = low_frequency_cutoff
        self.welch_segment_duration = welch_segment_duration
        self.max_filter_duration = max_filter_duration
        self.verbose = verbose

        self.index_step = int(round(step_size / DELTA_T))

        # Resolution at which the PSDs are stored. This is chosen to match the
        # padded segment length that generate_dataset.py whitens its waveforms
        # at: slice_length samples plus the whitening filter. Storing the PSD
        # at the matching delta_f avoids an unnecessary interpolation later.
        self.psd_delta_f = 1.0 / (DELTA_T * slice_length + max_filter_duration)

        self.all_psds = []            # one (n_detectors, n_freqs) array per segment
        self.all_whitening_psds = []

    def __iter__(self):
        import pycbc.psd

        with h5py.File(self.fpath, "r") as infile:
            missing = [d for d in self.detectors if d not in infile]
            if missing:
                raise KeyError(
                    f"Detector group(s) {missing} not found in {self.fpath}. "
                    f"Available: {list(infile.keys())}")

            segment_keys = list(infile[self.detectors[0]].keys())
            self.rng.shuffle(segment_keys)
            logging.info("Slicing %i segments from %s",
                         len(segment_keys), self.fpath)

            for segment_key in tqdm(segment_keys, desc="Slicing segments",
                                    disable=not self.verbose, ascii=True):
                whitened = []
                raw_psds = []
                for detector in self.detectors:
                    white, psd = whiten_segment(
                        infile[detector][segment_key][()],
                        self.low_frequency_cutoff,
                        welch_segment_duration=self.welch_segment_duration,
                        max_filter_duration=self.max_filter_duration)
                    whitened.append(white)
                    raw_psds.append(psd)

                # Store the PSDs at the common resolution, plus the regularised
                # versions used for whitening injected signals.
                psds = [pycbc.psd.interpolate(p, self.psd_delta_f).astype(np.float64)
                        for p in raw_psds]
                whitening_psds = [regularize_psd(p, self.low_frequency_cutoff)
                                  for p in psds]
                psd_index = len(self.all_psds)
                self.all_psds.append(np.stack(psds, axis=0))
                self.all_whitening_psds.append(np.stack(whitening_psds, axis=0))

                # Detectors may differ by a sample; use the common length.
                data = np.stack([w[:min(len(x) for x in whitened)]
                                 for w in whitened], axis=0)
                n_samples = data.shape[1]
                if n_samples < self.slice_length:
                    logging.debug("Segment %s too short after whitening (%i < %i), "
                                  "skipping", segment_key, n_samples,
                                  self.slice_length)
                    continue
                for start in range(0, n_samples - self.slice_length + 1,
                                   self.index_step):
                    yield data[:, start:start + self.slice_length], psd_index

    def write_metadata(self, h5py_object, **kwargs):
        """Write the PSD arrays and detector names into an open output file."""
        psds_ds = h5py_object.create_dataset(
            "psds", data=np.stack(self.all_psds, axis=0), **kwargs)
        psds_ds.attrs["delta_f"] = self.psd_delta_f
        w_psds_ds = h5py_object.create_dataset(
            "whitening_psds", data=np.stack(self.all_whitening_psds, axis=0), **kwargs)
        w_psds_ds.attrs["delta_f"] = self.psd_delta_f
        h5py_object.attrs["detectors"] = self.detectors


def write_chunk(path, noises, psd_indices, slicer, force, write_kwargs,
                slice_length, step_size):
    """Write one output file in the layout generate_dataset.py expects."""
    check_output_path(path, force)
    with h5py.File(path, "w" if force else "w-") as outf:
        outf.create_dataset("noises", data=np.stack(noises, axis=0), **write_kwargs)
        outf.create_dataset("psd_indices", data=np.array(psd_indices), **write_kwargs)
        slicer.write_metadata(outf, **write_kwargs)
        outf.attrs["sample_rate"] = SAMPLE_RATE
        outf.attrs["slice_length"] = slice_length
        outf.attrs["step_size_seconds"] = step_size
        outf.attrs["low_frequency_cutoff"] = LOW_FREQUENCY_CUTOFF
        outf.attrs["whitened"] = True
        outf.attrs["generator"] = "prepare_real_noise.py"
    logging.info("Wrote %s (%i slices)", path, len(noises))


def slice_noise_file(inputfile, output, detectors, slice_length, step_size,
                     chunk_size, seed, compress=False, force=False, verbose=False):
    """Whiten every segment of a noise file and cut it into fixed-length slices.

    With ``chunk_size <= 0`` everything is written to ``output``. Otherwise the
    slices are spread over ``<stem>_0000.hdf``, ``<stem>_0001.hdf``, ... with at
    most ``chunk_size`` slices each; every chunk is self-contained, carrying the
    PSDs its own ``psd_indices`` refer to.
    """
    detector_names = ("H1", "L1", "V1", "K1")[:detectors]
    rng = np.random.default_rng(seed)
    write_kwargs = dataset_write_kwargs(compress)

    chunked = chunk_size and chunk_size > 0
    stem, ext = os.path.splitext(output)
    if not ext:
        ext = ".hdf"
    parent = os.path.dirname(output)
    if parent:
        os.makedirs(parent, exist_ok=True)

    slicer = RealNoiseSlicer(inputfile, detector_names, slice_length, step_size,
                             rng, verbose=verbose)

    noises, psd_indices, chunk_index, total = [], [], 0, 0
    for noise, psd_index in slicer:
        noises.append(noise)
        psd_indices.append(psd_index)
        total += 1
        if chunked and len(noises) >= chunk_size:
            write_chunk(f"{stem}_{chunk_index:04d}{ext}", noises, psd_indices,
                        slicer, force, write_kwargs, slice_length, step_size)
            noises, psd_indices = [], []
            chunk_index += 1

    if not noises and total == 0:
        raise RuntimeError(
            f"No slices were produced from {inputfile}. Are the segments longer "
            f"than {slice_length} samples plus the whitening filter?")
    if noises:
        path = f"{stem}_{chunk_index:04d}{ext}" if chunked else output
        write_chunk(path, noises, psd_indices, slicer, force, write_kwargs,
                    slice_length, step_size)

    logging.info("Done: %i slices of %i samples from %i segments.",
                 total, slice_length, len(slicer.all_psds))


# =============================================================================
# Command-line interface
# =============================================================================
def main():
    parser = ArgumentParser(
        description="Prepare real detector noise for generate_dataset.py. "
                    "Run 'split' first, then 'slice' on the training half.")
    subparsers = parser.add_subparsers(dest="command", required=True,
                                       metavar="{split,slice}")

    # -- shared options ------------------------------------------------------
    common = ArgumentParser(add_help=False)
    common.add_argument("--seed", type=int, default=2026,
                        help="Random seed, so the result is reproducible. "
                             "Default: 2026.")
    common.add_argument("--compress", action="store_true",
                        help="gzip the output datasets. Smaller files, slower "
                             "to write and read.")
    common.add_argument("--verbose", action="store_true", help="Print progress.")
    common.add_argument("--debug", action="store_true", help="Print debug messages.")
    common.add_argument("--force", action="store_true",
                        help="Overwrite existing output files.")

    # -- stage 1 -------------------------------------------------------------
    p_split = subparsers.add_parser(
        "split", parents=[common],
        help="Divide a raw noise file into two files with no shared strain.",
        description="Stage 1: split the raw MLGWSC-1 noise file into disjoint "
                    "training and testing halves, at segment boundaries, so "
                    "that no stretch of detector data appears in both.")
    p_split.add_argument("inputfile", type=str,
                         help="Raw noise file, e.g. real_noise_file.hdf as "
                              "pulled by the MLGWSC-1 generate_data.py script.")
    p_split.add_argument("outputfiles", type=str, nargs=2,
                         metavar=("TRAIN_FILE", "TEST_FILE"),
                         help="The two destination paths. The first receives at "
                              "least --duration seconds of noise.")
    p_split.add_argument("-d", "--duration", type=float, default=1.0e6,
                         help="Minimum live time in seconds for the first "
                              "output file. Default: 1e6 (about 11.6 days).")

    # -- stage 2 -------------------------------------------------------------
    p_slice = subparsers.add_parser(
        "slice", parents=[common],
        help="Whiten and cut a noise file into fixed-length samples.",
        description="Stage 2: whiten each segment, estimate its PSD, and cut "
                    "the result into fixed-length slices in the layout that "
                    "generate_dataset.py --real-noise-file expects.")
    p_slice.add_argument("inputfile", type=str,
                         help="Noise file to slice, typically the first output "
                              "of the 'split' stage.")
    p_slice.add_argument("-o", "--output", type=str, required=True,
                         help="Output path. With --chunk-size, this becomes a "
                              "stem: <stem>_0000.hdf, <stem>_0001.hdf, ...")
    p_slice.add_argument("-d", "--detectors", type=int, default=2,
                         choices=[1, 2, 3, 4],
                         help="Number of detectors, taken in the order "
                              "H1, L1, V1, K1. Default: 2.")
    p_slice.add_argument("-s", "--sample-length", type=int, default=2048,
                         help="Length of each slice in samples, at 2048 Hz. "
                              "Default: 2048 (1 s). Must match the "
                              "--sample-length used by generate_dataset.py.")
    p_slice.add_argument("--step-size", type=float, default=DEFAULT_STEP_SIZE,
                         help="Seconds between the start of consecutive slices. "
                              "Default: 0.5, i.e. 50%% overlap for 1 s slices.")
    p_slice.add_argument("--chunk-size", type=int, default=0,
                         help="Maximum slices per output file. Default: 0, "
                              "meaning write a single file.")

    args = parser.parse_args()
    configure_logging(args.verbose, args.debug)

    try:
        if args.command == "split":
            split_noise_file(args.inputfile, args.outputfiles, args.duration,
                             args.seed, compress=args.compress, force=args.force,
                             verbose=args.verbose)
        else:
            slice_noise_file(args.inputfile, args.output, args.detectors,
                             args.sample_length, args.step_size, args.chunk_size,
                             args.seed, compress=args.compress, force=args.force,
                             verbose=args.verbose)
    except (RuntimeError, ValueError, KeyError, OSError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
