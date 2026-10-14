"""Run the coincident transformer as a search, with optional time slides.

Scores each detector separately, caches the per-detector outputs, and
combines them into a ranking statistic. Used for two measurements in the
chapter.

Without ``--num-timeslides`` it produces Dataset 4 candidate events for
Figure 1.2(d), in the same format as the coherent script.

With ``--num-timeslides`` it also builds a time-slide background: Livingston's
cached outputs are shifted relative to Hanford's by more than the
inter-detector light travel time, so surviving coincidences are accidental by
construction. The shifted pairs go through the same ranking, thresholding and
clustering as the zero-lag data, which matters because a difference in event
counts could otherwise come from the processing rather than from the noise.
Section 1.5 uses this to validate the false-alarm rates in Figure 1.3.
"""

import pathlib
import sys
import argparse
import json
import logging
import math
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
import pycbc.psd
import pycbc.types
from pycbc.filter import highpass_fir, lowpass_fir

# Make the repository root importable so ``gwsearch`` resolves whether or not
# the package has been installed with ``pip install -e .``.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from gwsearch.coincident_model import (CoincidentTransformer, MODEL_DEFAULTS,
    per_detector_usr, combined_usr, combined_usr_and)

# Matches the continuous-data whitening used to evaluate the model.
# The amplitude divisor is selected separately from the checkpoint or CLI.
WHITEN_DEFAULTS = dict(segment_duration=4.0, max_filter_duration=4.0,
                       low_frequency_cutoff=15.0, bandpass_lower=20.0,
                       bandpass_upper=None)


def whiten(strain, delta_t, segment_duration=4.0, max_filter_duration=4.0,
           low_frequency_cutoff=15.0, bandpass_lower=20.0,
           bandpass_upper=None):
    """PyCBC whitening with a FIR high-pass and no edge crop."""
    colored = pycbc.types.TimeSeries(np.asarray(strain), delta_t=delta_t)
    duration = len(colored) * delta_t
    seg_dur = float(segment_duration)
    if duration < seg_dur:
        if duration >= 2.: seg_dur = 2.
        elif duration >= 1.: seg_dur = 1.
        elif duration >= 0.5: seg_dur = 0.5
        else: raise ValueError(f"Strain segment too short: {duration:g} s")
    psd = pycbc.psd.interpolate(colored.psd(seg_dur), colored.delta_f)
    filter_len = int(min(float(max_filter_duration), seg_dur) * colored.sample_rate)
    psd = pycbc.psd.inverse_spectrum_truncation(
        psd, max_filter_len=filter_len,
        low_frequency_cutoff=low_frequency_cutoff, trunc_method="hann")
    white_ts = (colored.to_frequencyseries() * (1. / psd)**0.5).to_timeseries()
    if bandpass_lower is not None and bandpass_lower > 0:
        white_ts = highpass_fir(white_ts, frequency=bandpass_lower,
            order=max(128, int(2 * white_ts.sample_rate / bandpass_lower)))
    if bandpass_upper is not None and bandpass_upper < white_ts.sample_rate / 2:
        white_ts = lowpass_fir(white_ts, frequency=bandpass_upper,
            order=max(128, int(2 * white_ts.sample_rate / bandpass_upper)))
    return white_ts.numpy().astype(np.float32)


def prewhiten_file(source, destination, config):
    info = os.stat(source)
    provenance = json.dumps(dict(source=str(Path(source).resolve()),
        size=info.st_size, mtime_ns=info.st_mtime_ns, config=config, version=1),
        sort_keys=True)
    if os.path.exists(destination):
        with h5py.File(destination) as f:
            if f.attrs.get("provenance") != provenance:
                raise ValueError(f"Stale/unknown whitened cache: {destination}")
        return
    parent = str(Path(destination).resolve().parent)
    fd, partial = tempfile.mkstemp(prefix=".partial-white-", suffix=".hdf", dir=parent)
    os.close(fd)
    try:
        with h5py.File(source, "r") as fin, h5py.File(partial, "w") as fout:
            fout.attrs["provenance"] = provenance
            for ifo in ("H1", "L1"):
                grp = fout.create_group(ifo)
                for key in tqdm(fin[ifo], desc=f"Whitening {ifo}"):
                    ds = fin[ifo][key]
                    dt = float(ds.attrs["delta_t"])
                    values = whiten(ds[()], dt, **config)
                    out = grp.create_dataset(key, data=values, compression="lzf")
                    out.attrs["delta_t"] = dt
                    out.attrs["start_time"] = ds.attrs["start_time"]
        os.replace(partial, destination)
    finally:
        if os.path.exists(partial): os.remove(partial)


class SegmentWindows(IterableDataset):
    def __init__(self, path, key, step=0.1, model_sample_rate=2048):
        self.path, self.key = str(path), key
        with h5py.File(path) as f:
            a, b = f["H1"][key], f["L1"][key]
            self.dt, self.start = float(a.attrs["delta_t"]), float(a.attrs["start_time"])
            if (a.shape != b.shape or float(b.attrs["delta_t"]) != self.dt
                    or abs(float(b.attrs["start_time"]) - self.start) > self.dt/2):
                raise ValueError(f"H1 and L1 are not aligned in segment {key}")
            self.length = len(a)
        input_rate = int(round(1.0 / self.dt))
        if abs(1.0 / self.dt - input_rate) > 1e-4:
            raise ValueError(f"Input sample rate must be an integer: {1.0/self.dt:g} Hz")
        if input_rate < model_sample_rate or input_rate % model_sample_rate:
            raise ValueError(f"Input rate {input_rate} Hz must be an integer multiple "
                             f"of model rate {model_sample_rate} Hz")
        self.decim = input_rate // model_sample_rate
        self.window_samples = input_rate  # one second at the input rate
        self.step = int(round(step / self.dt))
        if self.step <= 0: raise ValueError("Step must be >= one sample")
        self.count = max(0, 1 + (self.length - self.window_samples)//self.step)

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        worker, workers = (0, 1) if wi is None else (wi.id, wi.num_workers)
        lo = worker * self.count // workers
        hi = (worker + 1) * self.count // workers
        with h5py.File(self.path, "r") as f:
            a, b = f["H1"][self.key], f["L1"][self.key]
            for ix in range(lo, hi):
                start = ix * self.step
                window = np.stack((a[start:start+self.window_samples],
                                   b[start:start+self.window_samples]))
                yield torch.from_numpy(window.astype(np.float32)), self.start + start*self.dt


def cluster(triggers, gap, timing_var):
    """Cluster independently per segment using exactly the same rule for fg/bg."""
    events = []
    for key, stream in triggers.items():
        stream.sort(key=lambda x: x[0])
        if not stream: continue
        previous = stream[0][0]
        best = stream[0]
        start = previous
        for time, score in stream[1:]:
            if time - previous > gap:
                events.append(best)
                if previous-start > 3:
                    logging.warning("Long trigger cluster in %s: %.1f s; check threshold", key, previous-start)
                best = (time, score)
                start = time
            elif score > best[1]:
                best = (time, score)
            previous = time
        events.append(best)
        if previous-start > 3:
            logging.warning("Long trigger cluster in %s: %.1f s; check threshold", key, previous-start)
    return (np.asarray([e[0] for e in events], dtype=np.float64),
            np.asarray([e[1] for e in events], dtype=np.float32),
            np.full(len(events), timing_var, dtype=np.float32))


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = ckpt.get("model_config", MODEL_DEFAULTS)
    if config != MODEL_DEFAULTS:
        logging.info("Using architecture stored in checkpoint: %s", config)
    if ckpt.get("model_family") != "coincident_transformer_v1":
        raise ValueError("Expected a coincident transformer checkpoint")
    model = CoincidentTransformer(**config)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, ckpt


def rank_from_terms(s_h, s_l, usr_type="or"):
    """Combine the two cached per-detector log odds into one score.

    Both detectors are scored independently, so a time slide only has to pair
    different cached values -- the network is never run again.
    """
    sh = torch.as_tensor(s_h, dtype=torch.float32)
    sl = torch.as_tensor(s_l, dtype=torch.float32)
    usr = combined_usr(sh, sl) if usr_type == "or" else combined_usr_and(sh, sl)
    return usr.cpu().numpy()


def time_slides(cache_path, num_slides, slide_step, threshold, cluster_gap,
                event_var, usr_type="or"):
    """Shift only L1's cached terms; cluster each slide and segment separately."""
    event_times, event_scores = [], []
    livetime = 0.
    with h5py.File(cache_path, "r") as cache:
        logging.info("Time slides: %d slides across %d cached segments; "
                     "this phase runs on the CPU", num_slides, len(cache))
        with tqdm(total=num_slides * len(cache), desc="Time-slide segment pairs",
                  unit="pair", mininterval=5) as progress:
          for key in cache:
            group = cache[key]
            times = group["times"][()]
            stride = float(group.attrs["stride_seconds"])
            for slide in range(1, num_slides + 1):
                progress.update(1)
                if len(times) < 2:
                    continue
                shift = round(slide * slide_step / stride)
                if shift < 1 or shift >= len(times):
                    continue
                n = len(times) - shift
                livetime += n * stride
                stream = []
                # Read only a manageable slice of profiles per ranking batch.
                for lo in range(0, n, 8192):
                    hi = min(lo + 8192, n)
                    scores = rank_from_terms(
                        group["sH"][lo:hi], group["sL"][lo+shift:hi+shift],
                        usr_type)
                    stream.extend((float(t), float(s)) for t, s in
                                  zip(times[lo:hi], scores) if np.isfinite(s) and s > threshold)
                ct, cs, _ = cluster({key: stream}, cluster_gap, event_var)
                event_times.extend(ct)
                event_scores.extend(cs)
        logging.info("Time slides complete: %d clustered background events; "
                     "%.3f days of accumulated slide livetime",
                     len(event_scores), livetime / 86400.)
    return (np.asarray(event_times, dtype=np.float64),
            np.asarray(event_scores, dtype=np.float32), livetime)


def zero_lag_from_cache(cache_path, threshold, usr_type):
    """Re-rank saved per-IFO terms without whitening or model inference."""
    triggers = {}
    with h5py.File(cache_path, "r") as cache:
        for key in cache:
            group = cache[key]
            needed = ("times", "sH", "sL")
            if any(name not in group for name in needed):
                raise ValueError(f"Incomplete per-detector cache for segment {key}")
            n = len(group["times"])
            if any(len(group[name]) != n for name in needed[1:]):
                raise ValueError(f"Mismatched per-detector cache lengths in {key}")
            stream = []
            for lo in range(0, n, 8192):
                hi = min(lo + 8192, n)
                scores = rank_from_terms(
                    group["sH"][lo:hi], group["sL"][lo:hi], usr_type)
                times = group["times"][lo:hi]
                stream.extend((float(t), float(s)) for t, s in zip(times, scores)
                              if np.isfinite(s) and s > threshold)
            triggers[key] = stream
    return triggers


def save_events(args, triggers, cache_path, scale, config):
    times, values, var = cluster(triggers, args.cluster_threshold, args.event_var)
    bg_t = bg_s = None
    if args.num_timeslides:
        bg_t, bg_s, livetime = time_slides(
            cache_path, args.num_timeslides, args.timeslide_step,
            args.score_threshold, args.cluster_threshold, args.event_var,
            args.usr_type)
    with h5py.File(args.outputfile, "w") as f:
        f.create_dataset("time", data=times)
        f.create_dataset("stat", data=values)
        f.create_dataset("var", data=var)
        f.attrs["checkpoint"] = str(Path(args.checkpoint).resolve())
        f.attrs["ranking_statistic"] = f"coincident_{args.usr_type}_usr"
        f.attrs["usr_type"] = args.usr_type
        f.attrs["amplitude_scale"] = scale
        f.attrs["cache_file"] = str(Path(cache_path).resolve())
        f.attrs["preprocessing"] = json.dumps(config, sort_keys=True)
        if bg_t is not None:
            f.create_dataset("bg_time", data=bg_t)
            f.create_dataset("bg_stat", data=bg_s)
            f.create_dataset("bg_var", data=np.full(len(bg_t), args.event_var, dtype=np.float32))
            f.attrs["bg_livetime_seconds"] = livetime
    logging.info("Saved %d events to %s", len(times), args.outputfile)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("inputfile", "outputfile", "checkpoint"):
        p.add_argument("--" + arg, required=True)
    p.add_argument("--whitened-file", default=None)
    p.add_argument("--cache-file", default=None,
                   help="HDF5 cache of the per-detector log odds")
    p.add_argument("--reuse-cache", action="store_true",
                   help="Re-rank an existing --cache-file without rerunning inference")
    p.add_argument("--white", action="store_true", help="Input already whitened with the evaluation convention")
    p.add_argument("--force", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--eval-workers", type=int, default=4)
    p.add_argument("--step-size", type=float, default=0.1)
    p.add_argument("--model-sample-rate", type=int, default=2048,
                   help="Checkpoint input rate in Hz; input data may be an integer multiple")
    p.add_argument("--score-threshold", type=float, default=2.,
                   help="Threshold on the coincident log odds")
    p.add_argument("--usr-type", choices=("or", "and"), default="or",
                   help="Per-detector network USR combination (default: or)")
    p.add_argument("--num-timeslides", type=int, default=0,
                   help="Optional cached time slides (default: none)")
    p.add_argument("--timeslide-step", type=float, default=5.,
                   help="Seconds of relative L1 shift per slide (default: 5)")
    p.add_argument("--cluster-threshold", type=float, default=0.35)
    p.add_argument("--event-offset", type=float, default=0.7)
    p.add_argument("--event-var", type=float, default=0.2)
    p.add_argument("--amplitude-scale", "--scale", dest="amplitude_scale",
                   type=float, default=None, metavar="FACTOR",
                   help="Divide both whitened detectors by FACTOR before inference. "
                        "Default: the checkpoint input_scale (normally 32). "
                        "For a different divisor, pass --amplitude-scale explicitly.")
    p.add_argument("--whitening-segment-duration", type=float, default=4.)
    p.add_argument("--whitening-max-filter-duration", type=float, default=4.)
    p.add_argument("--low-frequency-cutoff", type=float, default=15.)
    p.add_argument("--bandpass-lower", type=float, default=20.)
    p.add_argument("--bandpass-upper", type=float, default=None)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if (args.amplitude_scale is not None and
            (not math.isfinite(args.amplitude_scale) or args.amplitude_scale <= 0)):
        p.error("--amplitude-scale must be finite and positive")
    if args.batch_size <= 0 or args.eval_workers < 0:
        p.error("Invalid batch size or worker count")
    if args.model_sample_rate <= 0:
        p.error("--model-sample-rate must be positive")
    if args.step_size <= 0 or not 0 <= args.event_offset <= 1 or args.event_var <= 0:
        p.error("Invalid step size, event offset or event tolerance")
    if (args.num_timeslides < 0 or
            not math.isfinite(args.timeslide_step) or args.timeslide_step <= 0 or
            not math.isfinite(args.cluster_threshold) or args.cluster_threshold <= 0 or
            not math.isfinite(args.score_threshold)):
        p.error("Invalid time-slide, cluster or score setting")
    if args.num_timeslides and args.timeslide_step <= 1.0:
        p.error("--timeslide-step must exceed the 1 s analysis window")
    if Path(args.outputfile).exists() and not args.force:
        p.error("Output already exists (use --force to overwrite)")
    if Path(args.inputfile).resolve() == Path(args.outputfile).resolve():
        p.error("Output must not overwrite input")
    Path(args.outputfile).parent.mkdir(parents=True, exist_ok=True)
    config = dict(segment_duration=args.whitening_segment_duration,
        max_filter_duration=args.whitening_max_filter_duration,
        low_frequency_cutoff=args.low_frequency_cutoff,
        bandpass_lower=args.bandpass_lower, bandpass_upper=args.bandpass_upper)
    if args.reuse_cache:
        if not args.cache_file:
            p.error("--reuse-cache requires --cache-file")
        cache_path = args.cache_file
        if not Path(cache_path).is_file():
            p.error("--cache-file does not exist")
        if Path(cache_path).resolve() in (Path(args.inputfile).resolve(),
                                          Path(args.outputfile).resolve()):
            p.error("Cache file must differ from input and output")
        model, ckpt = load_model(args.checkpoint, args.device)
        if args.model_sample_rate != model.config["sample_rate"]:
            p.error("--model-sample-rate must match the checkpoint sample_rate "
                    f"({model.config['sample_rate']} Hz)")
        checkpoint_scale = float(ckpt.get("input_scale", 32.))
        if not math.isfinite(checkpoint_scale) or checkpoint_scale <= 0:
            p.error("Invalid input_scale in checkpoint")
        scale = checkpoint_scale if args.amplitude_scale is None else args.amplitude_scale
        with h5py.File(cache_path, "r") as cache:
            expected = str(Path(args.checkpoint).resolve())
            if cache.attrs.get("checkpoint") != expected:
                p.error("Cache checkpoint differs from --checkpoint")
            if not math.isclose(float(cache.attrs.get("scale", float("nan"))), scale):
                p.error("Cache amplitude scale differs from --amplitude-scale/checkpoint")
            if cache.attrs.get("whitening") != json.dumps(config, sort_keys=True):
                p.error("Cache preprocessing parameters differ from the CLI")
            if "source_file" in cache.attrs and cache.attrs["source_file"] != str(Path(args.inputfile).resolve()):
                p.error("Cache was produced from a different --inputfile")
            if "event_offset" in cache.attrs and not math.isclose(
                    float(cache.attrs["event_offset"]), args.event_offset):
                p.error("Cache event offset differs from --event-offset")
            if "event_offset" not in cache.attrs and args.event_offset != 0.7:
                p.error("Older cache has no event offset metadata; use the original 0.7 s offset")
        logging.info("Re-ranking saved per-detector terms using %s-type USR", args.usr_type)
        triggers = zero_lag_from_cache(cache_path, args.score_threshold,
                                       args.usr_type)
        save_events(args, triggers, cache_path, scale, config)
        return
    white_path = args.inputfile
    if not args.white:
        white_path = args.whitened_file or str(Path(args.outputfile).with_suffix("")) + "_whitened.hdf"
        if Path(white_path).resolve() in (Path(args.inputfile).resolve(), Path(args.outputfile).resolve()):
            p.error("Whitened cache path must differ from input and output")
        prewhiten_file(args.inputfile, white_path, config)
    model, ckpt = load_model(args.checkpoint, args.device)
    if args.model_sample_rate != model.config["sample_rate"]:
        p.error("--model-sample-rate must match the checkpoint sample_rate "
                f"({model.config['sample_rate']} Hz)")
    checkpoint_scale = float(ckpt.get("input_scale", 32.))
    if not math.isfinite(checkpoint_scale) or checkpoint_scale <= 0:
        p.error("Invalid input_scale in checkpoint")
    scale = checkpoint_scale if args.amplitude_scale is None else args.amplitude_scale
    if not math.isclose(scale, checkpoint_scale):
        logging.warning("Evaluation divisor %g differs from checkpoint divisor %g",
                        scale, checkpoint_scale)
    if config != WHITEN_DEFAULTS and not args.white:
        logging.warning("Whitening settings differ from the evaluation defaults")
    cache_path = args.cache_file or str(Path(args.outputfile).with_suffix("")) + "_cache.hdf"
    if Path(cache_path).resolve() in (Path(args.inputfile).resolve(),
                                      Path(args.outputfile).resolve(),
                                      Path(white_path).resolve()):
        p.error("Cache file must differ from input, output and whitened data")
    if Path(cache_path).exists() and not args.force:
        p.error("Cache already exists (use --force to regenerate)")
    triggers = {}
    with h5py.File(white_path) as f:
        keys = sorted(f["H1"].keys())
        if set(keys) != set(f["L1"].keys()):
            raise ValueError("H1 and L1 keys differ")
    partial = cache_path + ".partial"
    try:
        with h5py.File(partial, "w") as cache:
            cache.attrs["checkpoint"] = str(Path(args.checkpoint).resolve())
            cache.attrs["source_file"] = str(Path(args.inputfile).resolve())
            cache.attrs["scale"] = scale
            cache.attrs["whitening"] = json.dumps(config, sort_keys=True)
            cache.attrs["event_offset"] = args.event_offset
            cache.attrs["step_size"] = args.step_size
            for key in tqdm(keys, desc="Evaluating segments"):
                dataset = SegmentWindows(white_path, key, args.step_size,
                                         model_sample_rate=args.model_sample_rate)
                loader = DataLoader(dataset, batch_size=args.batch_size,
                                    num_workers=args.eval_workers)
                times_parts, s_parts = [], []
                with torch.inference_mode():
                    for windows, starts in loader:
                        windows = windows.to(args.device, non_blocking=True).float()
                        windows = windows[:, :, ::dataset.decim]
                        predictions = model(windows/scale)
                        s_parts.append(per_detector_usr(predictions["class_logits"]).cpu().numpy())
                        times_parts.append(starts.numpy() + args.event_offset)
                if times_parts:
                    times = np.concatenate(times_parts)
                    odds = np.concatenate(s_parts)
                    order = np.argsort(times)
                    times, odds = times[order], odds[order]
                else:
                    times = np.empty(0, dtype=np.float64)
                    odds = np.empty((0, 2), dtype=np.float32)
                group = cache.create_group(key)
                group.attrs["stride_seconds"] = dataset.step * dataset.dt
                group.create_dataset("times", data=times)
                for name, values in (("sH", odds[:, 0]), ("sL", odds[:, 1])):
                    opts = {"compression": "lzf"} if len(values) else {}
                    group.create_dataset(name, data=values, **opts)
                scores = (rank_from_terms(odds[:, 0], odds[:, 1], args.usr_type)
                          if len(times) else np.empty(0))
                triggers[key] = [(float(t), float(s)) for t, s in zip(times, scores)
                                 if s > args.score_threshold]
        os.replace(partial, cache_path)
    finally:
        if os.path.exists(partial):
            os.unlink(partial)
    save_events(args, triggers, cache_path, scale, config)


if __name__ == "__main__":
    main()
