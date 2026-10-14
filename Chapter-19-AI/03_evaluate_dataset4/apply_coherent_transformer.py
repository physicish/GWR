"""Run the coherent transformer over MLGWSC-1 data as a search.

Slides a one-second window through continuous strain at a 0.1 s stride,
assigns a ranking statistic to each position, applies a threshold, and
clusters nearby triggers into candidate events. Writes the events in the
format the MLGWSC-1 scoring script expects.

Run once on the foreground file and once on the background file; scoring the
pair gives sensitive distance as a function of false-alarm rate.

Because both detectors enter the network together, the score cannot be
decomposed into per-detector terms, so a time slide requires a fresh pass
over the data. Section 1.4.3 and the coincident model address that.
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

from gwsearch.coherent_model import CoherentTransformer, MODEL_DEFAULTS, usr

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
    def __init__(self, path, key, step=0.1):
        self.path, self.key = str(path), key
        with h5py.File(path) as f:
            a, b = f["H1"][key], f["L1"][key]
            self.dt, self.start = float(a.attrs["delta_t"]), float(a.attrs["start_time"])
            if (a.shape != b.shape or float(b.attrs["delta_t"]) != self.dt
                    or abs(float(b.attrs["start_time"]) - self.start) > self.dt/2):
                raise ValueError(f"H1 and L1 are not aligned in segment {key}")
            if abs(self.dt - 1/2048) > 1e-9:
                raise ValueError("Model requires 2048 Hz input")
            self.length = len(a)
        self.step = int(round(step / self.dt))
        if self.step <= 0: raise ValueError("Step must be >= one sample")
        self.count = max(0, 1 + (self.length - 2048)//self.step)

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        worker, workers = (0, 1) if wi is None else (wi.id, wi.num_workers)
        lo = worker * self.count // workers
        hi = (worker + 1) * self.count // workers
        with h5py.File(self.path, "r") as f:
            a, b = f["H1"][self.key], f["L1"][self.key]
            for ix in range(lo, hi):
                start = ix * self.step
                window = np.stack((a[start:start+2048], b[start:start+2048]))
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
    model = CoherentTransformer(**config)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, ckpt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("inputfile", "outputfile", "checkpoint"):
        p.add_argument("--" + arg, required=True)
    p.add_argument("--whitened-file", default=None)
    p.add_argument("--white", action="store_true", help="Input already whitened with the evaluation convention")
    p.add_argument("--force", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--eval-workers", type=int, default=4)
    p.add_argument("--step-size", type=float, default=0.1)
    p.add_argument("--score-threshold", type=float, default=-8.)
    p.add_argument("--cluster-threshold", type=float, default=0.35)
    p.add_argument("--event-offset", type=float, default=0.7)
    p.add_argument("--event-var", type=float, default=0.2)
    p.add_argument("--amplitude-scale", "--scale", dest="amplitude_scale",
                   type=float, default=None, metavar="FACTOR",
                   help="Divide both whitened detectors by FACTOR before inference. "
                        "Default: the checkpoint input_scale (32 for older checkpoints). "
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
    if args.step_size <= 0 or not 0 <= args.event_offset <= 1 or args.event_var <= 0:
        p.error("Invalid step size, event offset or event tolerance")
    if Path(args.outputfile).exists() and not args.force:
        p.error("Output already exists (use --force to overwrite)")
    if Path(args.inputfile).resolve() == Path(args.outputfile).resolve():
        p.error("Output must not overwrite input")
    config = dict(segment_duration=args.whitening_segment_duration,
        max_filter_duration=args.whitening_max_filter_duration,
        low_frequency_cutoff=args.low_frequency_cutoff,
        bandpass_lower=args.bandpass_lower, bandpass_upper=args.bandpass_upper)
    white_path = args.inputfile
    if not args.white:
        white_path = args.whitened_file or str(Path(args.outputfile).with_suffix("")) + "_whitened.hdf"
        if Path(white_path).resolve() in (Path(args.inputfile).resolve(), Path(args.outputfile).resolve()):
            p.error("Whitened cache path must differ from input and output")
        prewhiten_file(args.inputfile, white_path, config)
    model, ckpt = load_model(args.checkpoint, args.device)
    checkpoint_scale = float(ckpt.get("input_scale", 32.))
    if not math.isfinite(checkpoint_scale) or checkpoint_scale <= 0:
        p.error("Invalid input_scale in checkpoint")
    scale = checkpoint_scale if args.amplitude_scale is None else args.amplitude_scale
    if not math.isclose(scale, checkpoint_scale):
        logging.warning("Evaluation divisor %g differs from checkpoint divisor %g",
                        scale, checkpoint_scale)
    if config != WHITEN_DEFAULTS and not args.white:
        logging.warning("Whitening settings differ from the evaluation defaults")
    triggers = {}
    with h5py.File(white_path) as f:
        keys = sorted(f["H1"].keys())
        if set(keys) != set(f["L1"].keys()):
            raise ValueError("H1 and L1 keys differ")
    for key in tqdm(keys, desc="Evaluating segments"):
        dataset = SegmentWindows(white_path, key, args.step_size)
        stream = []
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            num_workers=args.eval_workers)
        with torch.inference_mode():
            for windows, starts in loader:
                values = usr(model(windows.to(args.device, non_blocking=True).float()/scale))
                times = starts.numpy() + args.event_offset
                scores = values.cpu().numpy()
                stream.extend((float(t), float(s)) for t, s in zip(times, scores)
                              if s > args.score_threshold)
        triggers[key] = stream
    times, values, var = cluster(triggers, args.cluster_threshold, args.event_var)
    with h5py.File(args.outputfile, "w") as f:
        f.create_dataset("time", data=times)
        f.create_dataset("stat", data=values)
        f.create_dataset("var", data=var)
        f.attrs["checkpoint"] = str(Path(args.checkpoint).resolve())
        f.attrs["ranking_statistic"] = "joint_signal_logit_minus_noise_logit"
        f.attrs["amplitude_scale"] = scale
        f.attrs["preprocessing"] = json.dumps(config, sort_keys=True)
    logging.info("Saved %d events to %s", len(times), args.outputfile)


if __name__ == "__main__":
    main()
