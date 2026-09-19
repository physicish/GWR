#!/usr/bin/env python
"""
Apply the trained baseline transformer to an MLGWSC-1 dataset.

This runs the baseline as an actual search: it whitens each continuous data
segment, slides a 1 s window along it, evaluates the network at every step,
clusters the resulting triggers into events, and writes them in the format the
MLGWSC-1 evaluation script expects.

Use it to produce the sensitive-distance-versus-FAR curve on dataset 4, which
is the measurement the ROC curve could not make.

------------------------------------------------------------------------------
WHERE THE BACKGROUND COMES FROM
------------------------------------------------------------------------------
No time slides are performed here. MLGWSC-1 supplies, for each dataset, a
background file containing one month of noise with no injections alongside a
foreground file containing the same noise with signals added. Run this script
once on each; the challenge's evaluation script then reads the two event lists
and derives the false-alarm rate from the background one directly.

------------------------------------------------------------------------------
MATCHING THE TRAINING CONDITIONS
------------------------------------------------------------------------------
The network sees whitened strain, and it will only behave as trained if the
whitening here resembles the whitening used to build its training data. The
defaults below are chosen to match generate_dataset.py: a 20 Hz low-frequency
cutoff and a 0.25 s whitening filter. The PSD, however, is necessarily
estimated from the data rather than supplied analytically, and that difference
is itself an instance of the train/test mismatch discussed in the chapter. It
is worth checking that the whitened test data has the same amplitude and
spectral flatness as the training data before trusting any result --
check_whitening.py does exactly that. Note the target is not unit variance:
pycbc's whitening gives a standard deviation of sqrt(f_s/2), about 32 at
2048 Hz.

------------------------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------------------------
Requires: torch, numpy, h5py, pycbc, tqdm, and train_baseline.py alongside.

    # Foreground: injections in dataset 4
    python apply_baseline.py \
        --inputfile ds4_fg.hdf --outputfile events_ds4_fg.hdf \
        --checkpoint runs/baseline_gaussian/best_state_dict.pt \
        --device cuda --verbose

    # Background: the same dataset without injections
    python apply_baseline.py \
        --inputfile ds4_bg.hdf --outputfile events_ds4_bg.hdf \
        --checkpoint runs/baseline_gaussian/best_state_dict.pt \
        --device cuda --verbose

    # Re-use a whitened file prepared earlier, skipping the whitening stage.
    # Only do this if that file was whitened with the settings below.
    python apply_baseline.py \
        --inputfile  /path/debug_white_ds4_fg_20Hz.hdf \
        --outputfile /path/output_triggers_baseline_ds4_fg.hdf \
        --checkpoint runs/baseline_gaussian/best_state_dict.pt \
        --white \
        --score-threshold 0.0 \
        --cluster-threshold 0.35 \
        --eval-workers 6 \
        --batch-size 1024 \
        --device cuda --verbose

Then score the result with the MLGWSC-1 tooling, e.g.

    python evaluate.py --injection-file injections.hdf \
        --foreground-events events_ds4_fg.hdf \
        --foreground-files ds4_fg.hdf \
        --background-events events_ds4_bg.hdf --output results_ds4.hdf

------------------------------------------------------------------------------
Adapted for this chapter from apply_carma_td_frame_corrected.py, the CASTOR
application script. The whitening and clustering follow the MLGWSC-1
submission of Ondrej Zelenka (https://github.com/ondrzel/ml-gw-search),
Apache License 2.0. Copyright 2026 Chayan Chatterjee.
------------------------------------------------------------------------------
"""

from argparse import ArgumentParser
import gc
import logging
import os
import sys
import time as clock

import h5py
import numpy as np
from tqdm import tqdm

import torch

import pycbc.psd
import pycbc.types
from pycbc.filter import highpass_fir, lowpass_fir

from train_baseline import DTYPE, load_checkpoint, usr_statistic


# =============================================================================
# Whitening
# =============================================================================
def whiten(strain, delta_t, segment_duration=4.0, max_filter_duration=0.25,
           trunc_method="hann", low_frequency_cutoff=20.0,
           bandpass_lower=20.0, bandpass_upper=None, remove_corrupted=False):
    """Whiten a strain segment, estimating its PSD by Welch averaging.

    Accepts a 1-D segment or a 2-D (n_detectors, n_samples) block. The PSD is
    measured from the data itself, because real detector noise drifts and no
    analytic curve describes a given stretch of it.

    ``max_filter_duration`` should match the value used when the training data
    was built, since it controls how sharply the whitening can follow spectral
    features and therefore what the whitened data looks like.
    """
    if strain.ndim == 2:
        return np.stack([
            whiten(row, delta_t, segment_duration, max_filter_duration,
                   trunc_method, low_frequency_cutoff, bandpass_lower,
                   bandpass_upper, remove_corrupted)
            for row in strain
        ])
    if strain.ndim != 1:
        raise ValueError(f"Expected a 1-D or 2-D array, got {strain.ndim}-D.")

    colored = pycbc.types.TimeSeries(np.asarray(strain), delta_t=delta_t)
    sample_rate = float(colored.sample_rate)
    duration = len(colored) / sample_rate

    # Fall back to shorter Welch segments if the data cannot support the
    # requested length, rather than failing on a short trailing segment.
    seg_duration = float(segment_duration)
    if duration < seg_duration:
        for candidate in (2.0, 1.0, 0.5):
            if duration >= candidate:
                seg_duration = candidate
                break
        else:
            raise ValueError(f"Segment of {duration:.3f} s is too short to "
                             f"estimate a PSD from.")

    psd = pycbc.psd.interpolate(colored.psd(seg_duration), colored.delta_f)
    filter_length = int(min(float(max_filter_duration), seg_duration) * sample_rate)
    psd = pycbc.psd.inverse_spectrum_truncation(
        psd, max_filter_len=filter_length,
        low_frequency_cutoff=low_frequency_cutoff, trunc_method=trunc_method)

    white = (colored.to_frequencyseries() * (1.0 / psd) ** 0.5).to_timeseries()

    # An explicit high pass removes the residual low-frequency power that
    # inverse spectrum truncation leaves behind.
    if bandpass_lower:
        white = highpass_fir(white, frequency=bandpass_lower,
                             order=max(128, int(2 * white.sample_rate / bandpass_lower)))
    if bandpass_upper and bandpass_upper < 0.5 * float(white.sample_rate):
        white = lowpass_fir(white, frequency=bandpass_upper,
                            order=max(128, int(2 * white.sample_rate / bandpass_upper)))

    values = white.numpy()
    if remove_corrupted and len(values) > filter_length:
        values = values[filter_length // 2: len(values) - filter_length // 2]
    return values


def prewhiten_file(infile, outfile, detectors, **whiten_kwargs):
    """Whiten every segment of an MLGWSC-1 file once, and cache the result.

    Whitening is done up front rather than per window because the PSD must be
    estimated from a long stretch of data, and because neighbouring windows
    overlap and would otherwise re-whiten the same samples repeatedly.
    """
    if os.path.exists(outfile):
        logging.info("Whitened file already exists, reusing: %s", outfile)
        return
    with h5py.File(infile, "r") as fin, h5py.File(outfile, "w") as fout:
        groups = {d: fout.require_group(d) for d in detectors}
        for key in tqdm(list(fin[detectors[0]].keys()), desc="Pre-whitening",
                        ascii=True):
            delta_t = fin[detectors[0]][key].attrs["delta_t"]
            start_time = fin[detectors[0]][key].attrs["start_time"]
            for detector in detectors:
                values = whiten(fin[detector][key][()], delta_t, **whiten_kwargs)
                dataset = groups[detector].create_dataset(
                    key, data=values.astype(np.float32), compression="gzip")
                dataset.attrs["delta_t"] = delta_t
                dataset.attrs["start_time"] = start_time


# =============================================================================
# Slicing
# =============================================================================
class SegmentSlicer(torch.utils.data.IterableDataset):
    """Stream fixed-length windows from one segment of a whitened file.

    Windows advance by ``step_size`` seconds, so consecutive windows overlap
    and a signal is seen at several positions. Both detectors are read at the
    same sample index: the analysis is zero-lag throughout.
    """

    def __init__(self, fpath, key, detectors=("H1", "L1"), step_size=0.1,
                 slice_length=2048):
        super().__init__()
        self.fpath = fpath
        self.key = key
        self.detectors = list(detectors)
        self.slice_length = int(slice_length)

        with h5py.File(fpath, "r") as handle:
            dataset = handle[self.detectors[0]][key]
            self.delta_t = float(dataset.attrs["delta_t"])
            self.start_time = float(dataset.attrs["start_time"])
            self.n_samples = int(dataset.shape[0])

        self.index_step = int(round(step_size / self.delta_t))
        self.time_step = self.delta_t * self.index_step
        self.num_slices = 1 + max(
            0, (self.n_samples - self.slice_length) // self.index_step)

    def __len__(self):
        return self.num_slices

    @property
    def analysed_seconds(self):
        """Live time contributed by this segment, for the FAR denominator."""
        return self.num_slices * self.time_step

    def __iter__(self):
        self._file = h5py.File(self.fpath, "r")
        self._data = [self._file[d][self.key] for d in self.detectors]

        info = torch.utils.data.get_worker_info()
        worker_id = 0 if info is None else info.id
        n_workers = 1 if info is None else info.num_workers
        per_worker, remainder = divmod(self.num_slices, n_workers)
        first = worker_id * per_worker + min(worker_id, remainder)
        last = (worker_id + 1) * per_worker + min(worker_id + 1, remainder)

        self._index = first * self.index_step
        self._limit = last * self.index_step
        self._time = self.start_time + first * self.time_step
        return self

    def __next__(self):
        if (self._index >= self._limit
                or self._index + self.slice_length > self.n_samples):
            if hasattr(self, "_file"):
                self._file.close()
            raise StopIteration
        window = np.stack([data[self._index: self._index + self.slice_length]
                           for data in self._data])
        time = self._time
        self._index += self.index_step
        self._time += self.time_step
        return (torch.from_numpy(window.astype(np.float32)),
                torch.tensor(time, dtype=torch.float64))


def decimate(x, factor):
    """Reduce the sample rate by an integer factor, if required."""
    return x if factor == 1 else x[:, :, ::factor]


# =============================================================================
# Inference
# =============================================================================
@torch.no_grad()
def evaluate_segments(slicers, model, device, trigger_threshold, batch_size,
                      workers, model_sample_rate, kernel_length, scale=1.0,
                      collect_all=True, verbose=False, desc="Segments"):
    """Run the network over every window of every segment.

    Returns
    -------
    triggers : dict
        Segment key -> list of ``[event_time, statistic]`` above threshold.
    all_scores : ndarray
        Every statistic computed, used to characterise the full distribution.
    live_time : float
        Total analysed live time in seconds.
    """
    model.eval()
    triggers = {}
    all_scores = []
    live_time = 0.0
    # Report the event at the centre of its window; the network localises no
    # more precisely than that.
    event_offset = 0.5 * kernel_length

    for slicer in tqdm(slicers, desc=desc, ascii=True, disable=not verbose):
        loader = torch.utils.data.DataLoader(
            slicer, batch_size=batch_size, num_workers=workers,
            pin_memory=(device.type == "cuda"), persistent_workers=False)

        input_rate = int(round(1.0 / slicer.delta_t))
        if input_rate % model_sample_rate:
            raise ValueError(f"Input sample rate {input_rate} is not an integer "
                             f"multiple of the model rate {model_sample_rate}.")
        factor = input_rate // model_sample_rate

        segment_triggers = []
        for windows, times in tqdm(loader, desc=f"  {slicer.key}", leave=False,
                                   ascii=True, disable=not verbose):
            windows = decimate(windows.to(device, non_blocking=True), factor)
            if scale not in (None, 1.0):
                windows = windows / float(scale)
            scores = usr_statistic(model(windows)).float().cpu().numpy()
            event_times = times.numpy().astype(np.float64) + event_offset

            if collect_all:
                all_scores.append(scores)
            above = scores > trigger_threshold
            if above.any():
                segment_triggers.extend(
                    [float(t), float(s)]
                    for t, s in zip(event_times[above], scores[above]))

        triggers[slicer.key] = segment_triggers
        live_time += slicer.analysed_seconds

    flat = (np.concatenate(all_scores).astype(np.float32) if all_scores
            else np.array([], dtype=np.float32))
    return triggers, flat, live_time


# =============================================================================
# Clustering
# =============================================================================
def cluster_trigger_list(trigger_list, cluster_threshold=0.35):
    """Collapse a stream of triggers into events, keeping the loudest of each.

    Clustering is transitive with an inclusive boundary: a trigger joins the
    current cluster when its separation from the preceding trigger is at most
    ``cluster_threshold``, so a new cluster begins only on a strictly larger
    gap.
    """
    if not trigger_list:
        return []
    ordered = sorted(trigger_list, key=lambda item: item[0])
    clusters = [[ordered[0]]]
    for trigger in ordered[1:]:
        if trigger[0] - clusters[-1][-1][0] > cluster_threshold:
            clusters.append([trigger])
        else:
            clusters[-1].append(trigger)

    loudest = []
    for cluster in clusters:
        values = np.asarray([item[1] for item in cluster])
        winner = cluster[int(np.argmax(values))]
        loudest.append([float(winner[0]), float(winner[1])])
    return loudest


def get_clusters(triggers, cluster_threshold=0.35, var=0.5):
    """Cluster every segment and return arrays in the MLGWSC-1 layout."""
    events = []
    for trigger_list in triggers.values():
        events.extend(cluster_trigger_list(trigger_list, cluster_threshold))
    times = np.array([e[0] for e in events], dtype=np.float64)
    stats = np.array([e[1] for e in events], dtype=np.float32)
    return times, stats, np.full(len(events), var, dtype=np.float32)


# =============================================================================
# Command-line interface
# =============================================================================
def main():
    parser = ArgumentParser(
        description="Apply the baseline transformer to an MLGWSC-1 dataset "
                    "and write events in the challenge's format.")

    parser.add_argument("--inputfile", required=True,
                        help="MLGWSC-1 strain file, e.g. the dataset-4 "
                             "foreground or background.")
    parser.add_argument("--outputfile", required=True,
                        help="Event file to write.")
    parser.add_argument("--checkpoint", required=True,
                        help="Checkpoint from train_baseline.py. The "
                             "architecture is read from the file, so no model "
                             "options need to be repeated here.")
    parser.add_argument("--whitened-file", default=None,
                        help="Where to cache the whitened strain. Default: the "
                             "output file with a '_whitened' suffix.")
    parser.add_argument("--white", action="store_true",
                        help="The input is already whitened; skip whitening.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing output file.")

    group = parser.add_argument_group("whitening")
    group.add_argument("--whitening-segment-duration", type=float, default=4.0,
                       help="Welch segment length for the PSD estimate. "
                            "Default: 4.")
    group.add_argument("--whitening-max-filter-duration", type=float,
                       default=0.25,
                       help="Whitening filter length. Default: 0.25, matching "
                            "the training data.")
    group.add_argument("--low-frequency-cutoff", type=float, default=15.0,
                       help="Cutoff for inverse spectrum truncation. "
                            "Default: 20, matching the training data.")
    group.add_argument("--bandpass-lower", type=float, default=20.0,
                       help="High-pass corner applied after whitening. "
                            "Default: 20.")
    group.add_argument("--bandpass-upper", type=float, default=None,
                       help="Optional low-pass corner. Default: none.")

    group = parser.add_argument_group("search")
    group.add_argument("--slice-duration", type=float, default=1.0,
                       help="Analysis window in seconds. Must match the "
                            "training sample length. Default: 1.")
    group.add_argument("--step-size", type=float, default=0.1,
                       help="Stride between consecutive windows, in seconds. "
                            "Default: 0.1.")
    group.add_argument("--score-threshold", type=float, default=0.0,
                       help="Keep windows whose statistic exceeds this. "
                            "Default: 0, i.e. anything the network calls more "
                            "signal-like than not.")
    group.add_argument("--cluster-threshold", type=float, default=0.35,
                       help="Maximum separation within one cluster, in "
                            "seconds. Default: 0.35.")
    group.add_argument("--event-var", type=float, default=0.2,
                       help="Timing uncertainty reported per event, in "
                            "seconds. Default: 0.5.")
    group.add_argument("--n-detectors", type=int, default=2, choices=[1, 2],
                       help="Number of detectors. Default: 2.")

    group = parser.add_argument_group("runtime")
    group.add_argument("--device", type=str, default="cpu",
                       help="Device for inference, e.g. 'cuda'. Default: cpu.")
    group.add_argument("--batch-size", type=int, default=1024,
                       help="Inference batch size. Default: 1024.")
    group.add_argument("--eval-workers", type=int, default=6,
                       help="DataLoader worker processes. Default: 6.")
    group.add_argument("--model-sample-rate", type=int, default=2048,
                       help="Sample rate the model expects. Default: 2048.")
    group.add_argument("--scale", type=float, default=1.0,
                       help="Divide the whitened input by this before "
                            "inference. Default: 1, since generate_dataset.py "
                            "applies no scaling.")
    group.add_argument("--verbose", action="store_true", help="Print progress.")
    group.add_argument("--debug", action="store_true", help="Debug messages.")

    args = parser.parse_args()
    logging.basicConfig(
        format="%(levelname)s | %(asctime)s: %(message)s",
        level=logging.DEBUG if args.debug else
              (logging.INFO if args.verbose else logging.WARN),
        datefmt="%d-%m-%Y %H:%M:%S")

    if os.path.isfile(args.outputfile) and not args.force:
        print(f"\nError: '{args.outputfile}' exists. Use --force to overwrite.",
              file=sys.stderr)
        sys.exit(1)

    detectors = ("H1", "L1")[:args.n_detectors]
    device = torch.device(args.device)
    started = clock.time()

    # -- 1. whiten ---------------------------------------------------------
    if args.white:
        eval_file = args.inputfile
        logging.warning(
            "Using '%s' as already-whitened input. Check that it was whitened "
            "with the same settings the training data used (%.2f s filter, "
            "%.0f Hz cutoff). A file prepared for a differently trained model "
            "will silently shift the input distribution.",
            args.inputfile, args.whitening_max_filter_duration,
            args.low_frequency_cutoff)
    else:
        eval_file = args.whitened_file or (
            os.path.splitext(args.outputfile)[0] + "_whitened.hdf")
        prewhiten_file(
            args.inputfile, eval_file, detectors,
            segment_duration=args.whitening_segment_duration,
            max_filter_duration=args.whitening_max_filter_duration,
            low_frequency_cutoff=args.low_frequency_cutoff,
            bandpass_lower=args.bandpass_lower,
            bandpass_upper=args.bandpass_upper)

    # -- 2. model ----------------------------------------------------------
    model = load_checkpoint(args.checkpoint, device)
    logging.info("Loaded %s", args.checkpoint)

    with h5py.File(eval_file, "r") as handle:
        keys = list(handle[detectors[0]].keys())
        if not keys:
            print(f"\nError: no segments found in '{eval_file}'.",
                  file=sys.stderr)
            sys.exit(1)
        delta_t = float(handle[detectors[0]][keys[0]].attrs["delta_t"])
    input_rate = int(round(1.0 / delta_t))
    slice_length = int(round(args.slice_duration * input_rate))

    # -- 3. search ---------------------------------------------------------
    slicers = [SegmentSlicer(eval_file, key, detectors=detectors,
                             step_size=args.step_size,
                             slice_length=slice_length) for key in keys]
    triggers, all_scores, live_time = evaluate_segments(
        slicers, model, device, args.score_threshold, args.batch_size,
        args.eval_workers, args.model_sample_rate, args.slice_duration,
        scale=args.scale, verbose=args.verbose, desc="Searching")

    times, stats, variances = get_clusters(
        triggers, cluster_threshold=args.cluster_threshold, var=args.event_var)
    logging.info("Found %i events in %.1f s of live time.",
                 len(times), live_time)

    with h5py.File(args.outputfile, "w") as fout:
        fout.create_dataset("time", data=times)
        fout.create_dataset("stat", data=stats)
        fout.create_dataset("var", data=variances)
        fout.create_dataset("all_vals", data=all_scores)
        fout.attrs["live_time"] = live_time
        fout.attrs["statistic"] = "usr"
        fout.attrs["design"] = "coherent"
        fout.attrs["checkpoint"] = os.path.abspath(args.checkpoint)
    del triggers, all_scores
    gc.collect()

    print(f"\n  Events written   {len(times):,}")
    print(f"  Live time        {live_time:.1f} s")
    print(f"\nWrote {args.outputfile} in {clock.time() - started:.1f} s")


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("forkserver")
    except RuntimeError:
        pass
    main()
