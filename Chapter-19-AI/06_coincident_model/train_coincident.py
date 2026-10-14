"""Train the coincident transformer on the same mixture as Section 1.4.2.

Section 1.4.3. The architecture is the coherent model with its first
convolutional layer narrowed from two input channels to one. The same
weights process Hanford and Livingston independently, producing a separate
pair of scores for each detector, and the two are combined afterwards by a
closed-form expression. At 664,042 parameters it is marginally smaller than
the coherent model's 664,522, so a difference in performance can be
attributed to the change in design rather than to capacity.

The motivation is the cost of a background estimate. Per-detector outputs can
be computed once and cached; a time slide then pairs outputs from different
times and repeats only the combination and the clustering, without evaluating
the network again.
"""
import argparse
import json
import logging
import math
import os
import random
import pathlib
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ml4gw import gw, transforms
from ml4gw.dataloading import ChunkedTimeSeriesDataset, Hdf5TimeSeriesDataset
from ml4gw.waveforms.cbc import IMRPhenomPv2
from ml4gw.waveforms.generator import TimeDomainCBCWaveformGenerator
# Make the repository root importable so ``gwsearch`` resolves whether or not
# the package has been installed with ``pip install -e .``.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from gwsearch.coincident_model import (CoincidentTransformer, MODEL_DEFAULTS,
    per_detector_usr, combined_usr)

SIGNAL_CLASS = 1


def bad_rows(tensor):
    """Return batch indices containing a nonfinite value (GPU safe)."""
    return torch.where(~torch.isfinite(tensor).reshape(len(tensor), -1).all(1))[0]


def describe_bad(stage, tensor):
    rows = bad_rows(tensor).cpu().tolist()
    return (f"{stage}: {len(rows)}/{len(tensor)} nonfinite rows; "
            f"first batch indices {rows[:10]}")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def noise_files(directory):
    paths = sorted(p.resolve() for p in Path(directory).iterdir()
                   if p.suffix.lower() in (".h5", ".hdf", ".hdf5") and p.is_file())
    if not paths:
        raise ValueError(f"No HDF5 files in {directory}")
    return paths


def split_noise_files(args):
    train_files = noise_files(args.noise_dir)
    if args.validation_noise_dir:
        valid_files = noise_files(args.validation_noise_dir)
        overlap = set(train_files) & set(valid_files)
        if overlap and args.allow_shared_noise_files:
            logging.warning("Validation and training share %d O3 files; "
                            "validation loss is not an independent-noise estimate",
                            len(overlap))
        elif overlap and set(train_files) == set(valid_files):
            # Supplying the same directory twice is common. Split its files
            # instead of reusing identical stretches of noise in both sets.
            if len(train_files) < 2:
                raise ValueError("The train/validation directories contain only "
                                 "one shared noise file; supply another file "
                                 "or explicitly pass --allow-shared-noise-files")
            rng = np.random.default_rng(args.seed)
            indices = rng.permutation(len(train_files))
            nval = min(len(train_files) - 1, max(1, round(
                len(train_files) * args.validation_file_fraction)))
            valid_files = [train_files[i] for i in indices[:nval]]
            train_files = [train_files[i] for i in indices[nval:]]
            logging.info("Same noise files supplied for training and validation; "
                         "split into %d training and %d validation files",
                         len(train_files), len(valid_files))
        elif overlap:
            # A partially overlapping validation directory should not leak
            # its noise into the training data.
            train_files = [p for p in train_files if p not in overlap]
            if not train_files:
                raise ValueError("All training noise files also occur in the "
                                 "validation directory; use another training "
                                 "file or --allow-shared-noise-files")
            logging.warning("Excluded %d overlapping O3 files from training",
                            len(overlap))
    else:
        if len(train_files) < 2:
            raise ValueError("Supply --validation-noise-dir or >=2 noise files")
        rng = np.random.default_rng(args.seed)
        indices = rng.permutation(len(train_files))
        nval = max(1, round(len(train_files) * args.validation_file_fraction))
        nval = min(nval, len(train_files) - 1)
        valid_files = [train_files[i] for i in indices[:nval]]
        train_files = [train_files[i] for i in indices[nval:]]
    needed = math.ceil(max(args.chunk_length,
                           args.psd_length + args.kernel_length + args.fduration)
                       * args.sample_rate)
    for path in train_files + valid_files:
        with h5py.File(path, "r") as f:
            for ifo in ("H1", "L1"):
                if ifo not in f or not isinstance(f[ifo], h5py.Dataset):
                    raise ValueError(f"{path}: expected root dataset {ifo}")
                if f[ifo].ndim != 1 or len(f[ifo]) < needed:
                    raise ValueError(f"{path}/{ifo} shorter than {needed} samples")
            if len(f["H1"]) != len(f["L1"]):
                raise ValueError(f"{path}: H1/L1 sample counts differ")
            # This format has no mandatory sample-rate metadata. If present,
            # reject mismatches rather than silently treating them as 2048 Hz.
            rate = f.attrs.get("sample_rate")
            if rate is not None and float(rate) != args.sample_rate:
                raise ValueError(f"{path}: sample_rate={rate}, expected {args.sample_rate}")
    return train_files, valid_files


class ML4GWBatchMaker:
    """Sample fresh Pv2 signals, project, rescale to network SNR and whiten."""

    def __init__(self, args, device):
        self.args, self.device = args, device
        self.n_psd = round(args.psd_length * args.sample_rate)
        self.n_white = round((args.kernel_length + args.fduration) * args.sample_rate)
        self.n_final = round(args.kernel_length * args.sample_rate)
        self.n_wave = round(args.waveform_duration * args.sample_rate)
        self.pad = round(args.fduration * args.sample_rate / 2)
        self.spectral = transforms.SpectralDensity(
            args.sample_rate, args.fftlength,
            average=args.psd_average, fast=False).to(device)
        self.whitener = transforms.Whiten(
            args.fduration, args.sample_rate, highpass=args.highpass).to(device)
        self.generator = TimeDomainCBCWaveformGenerator(
            approximant=IMRPhenomPv2().to(device),
            sample_rate=args.sample_rate, duration=args.waveform_duration,
            f_min=args.f_min, f_ref=args.f_ref,
            right_pad=args.waveform_right_pad).to(device)
        tensors, vertices = gw.get_ifo_geometry("H1", "L1")
        self.tensors, self.vertices = tensors.to(device), vertices.to(device)
        if self.n_white <= 2 * self.pad:
            raise ValueError("Whitening leaves no output window")
        if args.waveform_right_pad < self.pad / args.sample_rate + args.kernel_length - 0.6:
            raise ValueError("waveform-right-pad too short for merger at 0.6 s")
        if args.waveform_duration - args.waveform_right_pad < (
                self.pad / args.sample_rate + 0.8):
            raise ValueError("waveform-duration too short for whitening window")

    def _parameters(self, n):
        a = self.args
        rand = lambda *shape: torch.rand(shape, device=self.device, dtype=torch.float64)
        m = a.mass_min + (a.mass_max - a.mass_min) * rand(n, 2)
        m1, m2 = m.max(-1).values, m.min(-1).values
        params = {
            "mass_1": m1, "mass_2": m2,
            "chirp_mass": (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2,
            "mass_ratio": m2 / m1,  # verified ml4gw convention: <=1
            "distance": torch.full_like(m1, a.reference_distance),
            "phic": 2 * math.pi * rand(n),
            # Dataset 4: isotropic orbital orientation.
            "inclination": torch.acos((2 * rand(n) - 1).clamp(-1, 1)),
        }
        for k in ("s1", "s2"):
            magnitude = a.spin_max * rand(n)
            costheta = 2 * rand(n) - 1
            azimuth = 2 * math.pi * rand(n)
            sintheta = torch.sqrt((1 - costheta**2).clamp_min(0))
            params[f"{k}x"] = magnitude * sintheta * torch.cos(azimuth)
            params[f"{k}y"] = magnitude * sintheta * torch.sin(azimuth)
            params[f"{k}z"] = magnitude * costheta
        return params

    @torch.no_grad()
    def __call__(self, raw, n_signal=None):
        a = self.args
        raw = raw.to(device=self.device, dtype=torch.float64)
        n, d, samples = raw.shape
        if d != 2 or samples != self.n_psd + self.n_white:
            raise ValueError(f"Expected (N,2,{self.n_psd+self.n_white}), got {raw.shape}")
        if n_signal is None:
            n_signal = n // 2
        if not 0 <= n_signal <= n:
            raise ValueError("Invalid signal count")
        if len(bad_rows(raw)):
            raise FloatingPointError(describe_bad("raw O3 strain", raw) +
                                     "; check the HDF5 noise files for NaNs/Infs")
        # Keep the PSD estimate disjoint from the injection window; both
        # validation and training use precisely the same generation path.
        psd = self.spectral(raw[..., :self.n_psd])
        if len(bad_rows(psd)):
            raise FloatingPointError(describe_bad("estimated PSD", psd))
        # The ml4gw whitening filter inverts the PSD above the high-pass.
        # A zero-valued/gap-filled strain segment produces zero power there.
        freqs = torch.linspace(0, a.sample_rate / 2, psd.size(-1),
                               device=self.device)
        nonpositive = (psd[..., freqs >= a.highpass] <= 0).reshape(n, -1).any(1)
        if bool(nonpositive.any()):
            rows = torch.where(nonpositive)[0].cpu().tolist()
            raise FloatingPointError(
                f"PSD has zero/negative power above {a.highpass:g} Hz in "
                f"{len(rows)}/{n} rows (first indices {rows[:10]}). "
                "Check for zero-filled O3 gaps in the source noise files.")
        x = raw[..., self.n_psd:].clone()
        labels = torch.zeros(n, dtype=torch.long, device=self.device)
        if n_signal:
            indices = torch.randperm(n, device=self.device)[:n_signal]
            for group in indices.split(a.waveform_microbatch):
                b = len(group)
                response_psd = F.interpolate(
                    psd[group].reshape(b * d, 1, -1),
                    size=self.n_white // 2 + 1,
                    mode="linear", align_corners=False).reshape(b, d, -1)
                if len(bad_rows(response_psd)):
                    raise FloatingPointError(describe_bad(
                        "interpolated PSD for signal SNR", response_psd))
                last_bad = None
                # Pv2 numerical singularities may occur for isolated parameter
                # draws. Redraw a whole small group, never insert NaNs or
                # quietly relabel signals as noise.
                for attempt in range(5):
                    hc, hp = self.generator(**self._parameters(b))
                    if len(bad_rows(hc)) or len(bad_rows(hp)):
                        last_bad = (describe_bad("Pv2 h_cross", hc) + "; " +
                                    describe_bad("Pv2 h_plus", hp))
                        continue
                    dec = torch.asin(2 * torch.rand(b, device=self.device, dtype=torch.float64) - 1)
                    psi = math.pi * torch.rand(b, device=self.device, dtype=torch.float64)
                    phi = (2 * torch.rand(b, device=self.device, dtype=torch.float64) - 1) * math.pi
                    response = gw.compute_observed_strain(
                        dec=dec, psi=psi, phi=phi,
                        detector_tensors=self.tensors,
                        detector_vertices=self.vertices,
                        sample_rate=a.sample_rate, cross=hc, plus=hp)
                    if len(bad_rows(response)):
                        last_bad = describe_bad("projected Pv2 waveform", response)
                        continue
                    # The generator places coalescence at duration - right_pad.
                    # Put it at 0.6-0.8 s in the cropped 1 s network input.
                    merger_pos = .6 + .2 * torch.rand(b, device=self.device)
                    merger_idx = round((a.waveform_duration - a.waveform_right_pad)
                                       * a.sample_rate)
                    starts = merger_idx - self.pad - torch.round(
                        merger_pos * a.sample_rate).long()
                    offsets = torch.arange(self.n_white, device=self.device)
                    response = response[
                        torch.arange(b, device=self.device)[:, None, None],
                        torch.arange(d, device=self.device)[None, :, None],
                        (starts[:, None, None] + offsets[None, None, :])]
                    snr = a.snr_low + (a.snr_high - a.snr_low) * torch.rand(
                        b, device=self.device, dtype=torch.float64)
                    response = gw.reweight_snrs(
                        response, target_snrs=snr, psd=response_psd,
                        sample_rate=a.sample_rate, highpass=a.highpass)
                    if len(bad_rows(response)):
                        last_bad = describe_bad("SNR-scaled Pv2 response", response)
                        continue
                    if attempt:
                        logging.warning("Redrew %d waveforms after %d invalid Pv2 attempts",
                                        b, attempt)
                    break
                else:
                    raise FloatingPointError(
                        f"Waveform generation failed after 5 draws: {last_bad}")
                x[group] += response
                labels[group] = SIGNAL_CLASS
        if len(bad_rows(x)):
            raise FloatingPointError(describe_bad("strain after injection", x))
        white = self.whitener(x, psd)
        if white.shape != (n, d, self.n_final):
            raise ValueError(f"Whitened batch has wrong shape: {white.shape}")
        # The online ml4gw path stays unscaled (output_scale=1), while
        # the pregenerated PyCBC-whitened path uses --amplitude-scale. The model's
        # per-sample input normalization makes those amplitudes comparable.
        if len(bad_rows(white)):
            rows = bad_rows(white)
            signal_count = int(labels[rows].sum().item())
            raise FloatingPointError(
                describe_bad("ml4gw whitening output", white) +
                f"; {signal_count} signal / {len(rows)-signal_count} noise rows. "
                "Input strain and PSD were finite; inspect PSD scale and "
                "whitening in the installed ml4gw version.")
        white = white * a.output_scale
        return white.float(), labels


class FixedValidation(Dataset):
    def __init__(self, path):
        self.path = str(path)
        self.file = h5py.File(self.path, "r")
        self.strain = self.file["strain"]
        self.labels = self.file["labels"]
        try:
            if self.labels.shape != (20000,) or self.strain.shape[0] != 20000:
                raise ValueError("Validation file must have 20,000 examples")
            if int((self.labels[:] == 1).sum()) != 10000:
                raise ValueError("Validation file must contain 10,000 signals")
        except Exception:
            self.file.close()
            raise

    def __del__(self):
        if hasattr(self, "file"):
            self.file.close()

    def __len__(self):
        return 20000

    def __getitem__(self, index):
        return torch.from_numpy(self.strain[index]), int(self.labels[index])


def make_noise_loader(files, args, batches, seed, batch_size):
    # Each HDF5 file has raw H1/L1 at its root. "coincident=True" keeps
    # real inter-detector correlations and glitches; train/validation have
    # disjoint file lists, avoiding direct background leakage.
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    batches_per_chunk = min(10, batches)
    chunks = math.ceil(batches / batches_per_chunk)
    source = Hdf5TimeSeriesDataset(
        fnames=[str(p) for p in files], channels=["H1", "L1"],
        kernel_size=round(args.chunk_length * args.sample_rate),
        batch_size=args.reads_per_chunk, batches_per_epoch=chunks,
        coincident=True)
    ds = ChunkedTimeSeriesDataset(
        source, kernel_size=round((args.psd_length + args.kernel_length +
                                   args.fduration) * args.sample_rate),
        batch_size=batch_size, batches_per_chunk=batches_per_chunk,
        coincident=True)
    return DataLoader(ds, batch_size=None, num_workers=0)


def clean_noise_batches(files, args, target_batches, seed, batch_size):
    """Refill full batches from finite raw strain windows before injecting.

    Reject the entire two-detector PSD + injection window if either detector
    contains a NaN or infinity. Draw extra windows only as needed; a finite
    draw budget makes highly corrupted input fail with a useful error.
    """
    max_draws = math.ceil(target_batches * 1.2) + 32
    loader = make_noise_loader(files, args, max_draws, seed, batch_size)
    pending = None
    produced = rejected = examined = 0
    for raw in loader:
        examined += len(raw)
        good = torch.isfinite(raw).reshape(len(raw), -1).all(dim=1)
        rejected += int((~good).sum().item())
        raw = raw[good]
        if not len(raw):
            continue
        pending = raw if pending is None else torch.cat((pending, raw), dim=0)
        while len(pending) >= batch_size and produced < target_batches:
            batch, pending = pending[:batch_size], pending[batch_size:]
            produced += 1
            yield batch
        if produced == target_batches:
            if rejected:
                logging.warning("Discarded %d/%d raw noise windows with NaN/Inf "
                                "while preparing %d full batches", rejected,
                                examined, produced)
            return
    raise RuntimeError(
        f"Only {produced}/{target_batches} full clean noise batches available "
        f"after inspecting {examined} raw windows ({rejected} NaN/Inf). "
        "Check the input O3 files or increase the clean sampling budget.")


def validation_metadata(args, valid_files):
    # Hash settings and file identities to prevent accidental cache reuse.
    cfg = dict(schema_version=4, seed=args.seed + 1,
               snr=[args.snr_low, args.snr_high],
               waveform="IMRPhenomPv2", mass=[args.mass_min, args.mass_max],
               spin_max=args.spin_max, fftlength=args.fftlength,
               psd_average=args.psd_average,
               psd_length=args.psd_length, fduration=args.fduration,
               highpass=args.highpass, f_min=args.f_min, f_ref=args.f_ref,
               sample_rate=args.sample_rate, kernel_length=args.kernel_length,
               output_scale=args.output_scale, chunk_length=args.chunk_length,
               reads_per_chunk=args.reads_per_chunk,
               validation_batch_size=args.validation_batch_size,
               waveform_microbatch=args.waveform_microbatch,
               waveform_duration=args.waveform_duration,
               waveform_right_pad=args.waveform_right_pad,
               reference_distance=args.reference_distance,
               validation_files=[(str(p), p.stat().st_size,
                                  p.stat().st_mtime_ns) for p in valid_files])
    return json.dumps(cfg, sort_keys=True)


def create_validation(path, valid_files, args, maker):
    metadata = validation_metadata(args, valid_files)
    if path.exists() and not args.rebuild_validation:
        with h5py.File(path, "r") as f:
            if f.attrs.get("configuration") != metadata:
                raise ValueError(f"Validation provenance mismatch at {path}; "
                                 "choose a new filename or --rebuild-validation")
        return FixedValidation(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    if tmp.exists():
        tmp.unlink()
    seed_everything(args.seed + 1)
    num_batches = math.ceil(20000 / args.validation_batch_size)
    loader = clean_noise_batches(valid_files, args, num_batches,
                                 args.seed + 1, args.validation_batch_size)
    try:
        with h5py.File(tmp, "w") as f:
            strain = f.create_dataset("strain", (20000, 2,
                                      round(args.kernel_length * args.sample_rate)),
                                      dtype="f4", chunks=(min(64, args.validation_batch_size),
                                                          2, round(args.kernel_length * args.sample_rate)),
                                      compression="lzf")
            labels = f.create_dataset("labels", (20000,), dtype="i1")
            f.attrs["configuration"] = metadata
            position = 0
            # Explicitly allocate exactly 10,000 per class, including the last
            # partial batch. Class order is shuffled within each batch.
            n_signal_remaining = 10000
            for raw in tqdm(loader, total=num_batches, desc="Generating validation",
                            disable=not (args.verbose and sys.stderr.isatty())):
                count = min(raw.shape[0], 20000 - position)
                if count <= 0:
                    break
                n_signal = min(n_signal_remaining, count // 2)
                if position + count == 20000:
                    n_signal = n_signal_remaining
                x, y = maker(raw[:count], n_signal=n_signal)
                strain[position:position+count] = x.cpu().numpy()
                labels[position:position+count] = y.cpu().numpy()
                position += count
                n_signal_remaining -= n_signal
            if position != 20000 or n_signal_remaining != 0:
                raise RuntimeError("Noise loader did not produce 20,000 validation examples")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return FixedValidation(path)


class PairedPregen(Dataset):
    """Load prewhitened data with the original, PSD-associated noise pairing.

    The first N noise rows in each file accompany its N waveform rows.
    Pure-noise examples draw from all available noise rows, including those
    paired with waveforms; a separate pure-noise pool is not required.
    """
    def __init__(self, directory, group, fraction=1.0, seed=2026,
                 snr_low=7., snr_high=20., scale=32., deterministic=False):
        if not 0 < fraction <= 1:
            raise ValueError("Pregenerated fraction must lie in (0, 1]")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Pregenerated amplitude divisor must be finite and positive")
        paths = noise_files(directory)
        waves, signal_noises, all_noises = [], [], []
        for path in paths:
            with h5py.File(path) as f:
                if group not in f:
                    raise ValueError(f"Missing {group} in {path}")
                g = f[group]
                w, n = np.asarray(g["waveforms"], dtype=np.float32), np.asarray(g["noises"], dtype=np.float32)
                if w.ndim != 3 or w.shape[1:] != (2, 2048) or n.ndim != 3 or n.shape[1:] != (2, 2048):
                    raise ValueError(f"Expected (N,2,2048) in {path}/{group}")
                if not len(w) or len(n) < len(w):
                    raise ValueError(f"Need at least one paired noise row per waveform in {path}")
                waves.append(w)
                signal_noises.append(n[:len(w)])
                all_noises.append(n)
        self.waves = torch.from_numpy(np.concatenate(waves))
        # Concatenate the per-file paired slices separately: waveform i must
        # retain its own noise i even if earlier files had extra noise rows.
        self.signal_noises = torch.from_numpy(np.concatenate(signal_noises))
        self.noises = torch.from_numpy(np.concatenate(all_noises))
        nw = len(self.waves)
        rng = np.random.default_rng(seed)
        n_keep = max(1, round(nw * fraction))
        self.signal_ids = np.sort(rng.choice(nw, n_keep, replace=False))
        self.noise_ids = np.sort(rng.choice(len(self.noises), n_keep, replace=False))
        self.snr_low, self.snr_high = snr_low, snr_high
        self.scale, self.seed, self.deterministic = scale, seed, deterministic
        logging.info("%s: %d fixed-pair signals, %d pure noises (fraction %.2f)",
                     group, n_keep, n_keep, fraction)

    def __len__(self):
        return 2 * len(self.signal_ids)

    def __getitem__(self, idx):
        if idx >= len(self.signal_ids):
            noise = self.noises[int(self.noise_ids[idx-len(self.signal_ids)])] / self.scale
            return noise, 0
        source_idx = int(self.signal_ids[idx])
        if self.deterministic:
            snr = float(np.random.default_rng([self.seed, source_idx]).uniform(self.snr_low, self.snr_high))
        else:
            snr = self.snr_low + (self.snr_high-self.snr_low)*float(torch.rand(()))
        clean = (snr * self.waves[source_idx]) / self.scale
        return self.signal_noises[source_idx] / self.scale + clean, 1


class EMAShadow:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: p.detach().clone() for k,p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            self.shadow[name].lerp_(p.detach(), 1.-self.decay)

    def state_dict(self, model):
        result = {k: v.detach().clone() for k,v in model.state_dict().items()}
        for k, v in self.shadow.items():
            result[k] = v.detach().clone()
        return result


def loss_terms(predictions, labels):
    """Shared per-IFO binary cross entropy."""
    logits = predictions["class_logits"]
    per_ifo_labels = labels[:, None].expand(-1, 2).reshape(-1)
    return F.cross_entropy(logits.reshape(-1, 2), per_ifo_labels)


@torch.inference_mode()
def validate(model, online, pregen, device, batch_size):
    result = {}
    model.eval()
    for name, ds in (("online", online), ("pregen", pregen)):
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        total_loss, cls_total, correct, count = 0., 0., 0, 0
        all_scores, all_labels = [], []
        for batch, labels in tqdm(loader, desc=f"Validating {name}", leave=False,
                                         disable=not sys.stderr.isatty()):
            # Online validation was cached after ml4gw whitening without a
            # divisor; PairedPregen applies the configured divisor on reading.
            batch = batch.to(device, non_blocking=True).float()
            labels = labels.to(device).long()
            predictions = model(batch)
            logits = predictions["class_logits"]
            loss = cls = loss_terms(predictions, labels)
            total_loss += float(loss) * len(batch)
            cls_total += float(cls) * len(batch)
            correct += int((logits.argmax(dim=-1) == labels[:, None]).sum().item())
            count += len(batch)
            odds = per_detector_usr(logits)
            all_scores.append(combined_usr(odds[:, 0], odds[:, 1]).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        scores, ys = np.concatenate(all_scores), np.concatenate(all_labels)
        negatives, positives = scores[ys == 0], scores[ys == 1]
        cutoff = np.quantile(negatives, 0.999)
        result[name] = dict(loss=total_loss/count, cls=cls_total/count,
                            accuracy=correct/(2*count),
                            tpr_fpr_1e3=float((positives > cutoff).mean()))
    return result


def save_checkpoint(path, model, weights, args, epoch, metric, extra=None):
    payload = dict(model_config=model.config, model_state_dict=weights,
                   model_family="coincident_transformer_v1",
                   input_scale=args.amplitude_scale, online_training_scale=1.,
                   pregen_training_scale=args.amplitude_scale, class_order=["noise", "signal"],
                   online_preprocessing=dict(psd_length=args.psd_length,
                     fftlength=args.fftlength,
                     average=args.psd_average, fduration=args.fduration,
                     highpass=args.highpass, output_scale=args.output_scale),
                   pregen_preprocessing=f"stored whitened HDF, fixed PSD-associated pairing, divide by {args.amplitude_scale:g}",
                   eval_preprocessing=f"PSD 4s, filter 4s, cutoff 15Hz, FIR high-pass 20Hz, divide by {args.amplitude_scale:g}",
                   epoch=epoch, validation_loss=metric, weights_kind="ema")
    if extra: payload.update(extra)
    tmp = str(path) + ".partial"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def fit(args):
    if args.output_scale != 1.:
        raise ValueError("Online whitening is unscaled: use --output-scale 1")
    if not (0 < args.pregen_val_fraction <= 1):
        raise ValueError("Validation fraction must be in (0,1]")
    seed_everything(args.seed)
    out = Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=True)
    train_files, validation_files = split_noise_files(args)
    logging.info("Training noise: %d files; held-out noise: %d files", len(train_files), len(validation_files))
    maker = ML4GWBatchMaker(args, args.train_device)
    online_val = create_validation(out / "online_validation.hdf", validation_files, args, maker)
    pregen_train = PairedPregen(args.pregen_dir, "training", seed=args.seed,
        snr_low=args.snr_low, snr_high=args.snr_high, scale=args.amplitude_scale)
    pregen_val = PairedPregen(args.pregen_dir, "validation", seed=args.seed+1,
        snr_low=args.snr_low, snr_high=args.snr_high,
        fraction=args.pregen_val_fraction, scale=args.amplitude_scale,
        deterministic=True)
    pregen_loader = DataLoader(pregen_train, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.pregen_workers, drop_last=True,
                               pin_memory=args.train_device.startswith("cuda"))
    if len(pregen_loader) == 0:
        raise ValueError("Pregenerated training set shorter than a full batch")
    pregen_iter = iter(pregen_loader)
    model_config = dict(MODEL_DEFAULTS)
    model_config["input_instance_norm"] = not args.no_input_instance_norm
    model = CoincidentTransformer(**model_config).to(args.train_device)
    if args.no_input_instance_norm:
        logging.warning("With InstanceNorm disabled, online examples are unscaled but pregenerated examples are divided by %g",
                        args.amplitude_scale)
    logging.info("Joint model has %d trainable parameters", sum(p.numel() for p in model.parameters()))
    ema = EMAShadow(model, args.ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                   weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate,
        pct_start=0.1, total_steps=args.epochs*args.steps_per_epoch)
    best, start_epoch = float("inf"), 0
    if args.resume:
        saved = torch.load(args.resume, map_location=args.train_device, weights_only=False)
        if (saved.get("model_family") != "coincident_transformer_v1" or
                saved.get("model_config") != model.config or saved.get("weights_kind") != "raw"):
            raise ValueError("Resume requires last_coincident_training.pt from this architecture")
        if not math.isclose(float(saved.get("pregen_training_scale", 32.)),
                            args.amplitude_scale):
            raise ValueError("--amplitude-scale must match the resumed checkpoint")
        if saved.get("total_steps") != args.epochs*args.steps_per_epoch:
            raise ValueError("Cannot resume OneCycleLR with a different total step count")
        model.load_state_dict(saved["model_state_dict"], strict=True)
        ema.shadow = {k:v.to(args.train_device) for k,v in saved["ema_shadow"].items()}
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        schedule.load_state_dict(saved["scheduler_state_dict"])
        best, start_epoch = float(saved["best_metric"]), int(saved["epoch"])
    csv = out / "metrics.csv"
    losses_path = out / "losses.txt"
    if start_epoch == 0:
        with csv.open("w") as f:
            f.write("epoch,step,training_loss,training_cls,"
                    "online_val_loss,online_val_cls,"
                    "pregen_val_loss,pregen_val_cls,"
                    "online_tpr_fpr1e-3,pregen_tpr_fpr1e-3,pregen_batches\n")
        with losses_path.open("w") as f:
            f.write("# epoch  train_loss  train_acc  online_val_loss  "
                    "online_val_acc  pregen_val_loss  pregen_val_acc\n")
    elif not csv.exists():
        raise FileNotFoundError(f"Cannot resume without {csv}")
    elif not losses_path.exists():
        # Older checkpoints have metrics.csv but no accuracy history.
        # Start this file at the resumed epoch without fabricating old values.
        with losses_path.open("w") as f:
            f.write("# epoch  train_loss  train_acc  online_val_loss  "
                    "online_val_acc  pregen_val_loss  pregen_val_acc\n")
    for epoch in range(start_epoch, args.epochs):
        seed_everything(args.seed + 1000 + epoch)
        frac = min(1., epoch/max(1, args.pregen_ramp_epochs))
        prob = args.pregen_prob_start + (args.pregen_prob_end-args.pregen_prob_start)*frac
        sources = np.random.default_rng(args.seed + epoch).random(args.steps_per_epoch) < prob
        n_online = int((~sources).sum())
        online_iter = iter(clean_noise_batches(train_files, args, n_online,
                                               args.seed+1000+epoch, args.batch_size)) if n_online else None
        model.train()
        total_loss, total_cls = 0., 0.
        train_correct, train_seen = 0, 0
        for use_pregen in tqdm(sources, desc=f"Epoch {epoch+1}/{args.epochs}",
                               disable=not sys.stderr.isatty()):
            if use_pregen:
                try: strain, labels = next(pregen_iter)
                except StopIteration:
                    pregen_iter = iter(pregen_loader)
                    strain, labels = next(pregen_iter)
                strain = strain.to(args.train_device, non_blocking=True).float()
                labels = labels.to(args.train_device).long()
            else:
                raw = next(online_iter)
                strain, labels = maker(raw, n_signal=len(raw)//2)
                # Online ml4gw whitening is passed in unscaled;
                # only pregenerated inputs are divided by --amplitude-scale in PairedPregen.
            optimizer.zero_grad(set_to_none=True)
            predictions = model(strain)
            logits = predictions["class_logits"]
            loss = cls = loss_terms(predictions, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch+1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
            optimizer.step()
            schedule.step()
            ema.update(model)
            total_loss += float(loss.detach())
            total_cls += float(cls.detach())
            train_correct += int((logits.detach().argmax(dim=-1) == labels[:, None]).sum().item())
            train_seen += 2 * labels.numel()
        raw_weights = {k:v.detach().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(ema.state_dict(model), strict=True)
        val = validate(model, online_val, pregen_val, args.train_device,
                       args.validation_batch_size)
        metric = (val["online"]["loss"] + val["pregen"]["loss"])/2
        if metric < best:
            best = metric
            save_checkpoint(out/"best_coincident_ema.pt", model, model.state_dict(), args, epoch+1, metric)
        if args.save_every_epoch:
            save_checkpoint(out/f"coincident_epoch_{epoch+1:04d}_ema.pt", model,
                            model.state_dict(), args, epoch+1, metric)
        model.load_state_dict(raw_weights, strict=True)
        save_checkpoint(out/"last_coincident_training.pt", model, raw_weights, args,
            epoch+1, metric, extra=dict(weights_kind="raw", total_steps=args.epochs*args.steps_per_epoch,
                 best_metric=best, ema_shadow=ema.shadow,
                 optimizer_state_dict=optimizer.state_dict(),
                 scheduler_state_dict=schedule.state_dict()))
        with csv.open("a") as f:
            f.write(f"{epoch+1},{(epoch+1)*args.steps_per_epoch},"
                    f"{total_loss/args.steps_per_epoch:.8f},"
                    f"{total_cls/args.steps_per_epoch:.8f},"
                    f"{val['online']['loss']:.8f},{val['online']['cls']:.8f},"
                    f"{val['pregen']['loss']:.8f},{val['pregen']['cls']:.8f},"
                    f"{val['online']['tpr_fpr_1e3']:.6f},{val['pregen']['tpr_fpr_1e3']:.6f},"
                    f"{int(sources.sum())}\n")
        train_loss = total_loss / args.steps_per_epoch
        train_accuracy = train_correct / train_seen
        with losses_path.open("a") as f:
            f.write(f"{epoch+1:04d}  {train_loss:.6f}  {train_accuracy:.6f}  "
                    f"{val['online']['loss']:.6f}  {val['online']['accuracy']:.6f}  "
                    f"{val['pregen']['loss']:.6f}  {val['pregen']['accuracy']:.6f}\n")
        logging.info("epoch %04d | train loss %.5f acc %.4f | "
                     "online valid loss %.5f acc %.4f | "
                     "pregen valid loss %.5f acc %.4f | "
                     "class %.4f | "
                     "best val loss %.5f | online/pregen TPR@FPR=1e-3 %.4f/%.4f",
                     epoch+1, train_loss, train_accuracy,
                     val["online"]["loss"], val["online"]["accuracy"],
                     val["pregen"]["loss"], val["pregen"]["accuracy"],
                     total_cls/args.steps_per_epoch,
                     best,
                     val["online"]["tpr_fpr_1e3"], val["pregen"]["tpr_fpr_1e3"])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--noise-dir", required=True, type=Path,
                   help="HDF5 real-O3 files with root H1 and L1 raw 2048-Hz arrays")
    p.add_argument("--validation-noise-dir", type=Path, default=None)
    p.add_argument("--allow-shared-noise-files", action="store_true",
                   help="Allow the same raw O3 file in training and validation. "
                        "This makes online validation optimistic; by default, "
                        "identical directories are split by file and partial "
                        "overlap is excluded from training.")
    p.add_argument("--validation-file-fraction", type=float, default=.2)
    p.add_argument("--pregen-dir", required=True, type=Path,
                   help="HDF5 files each with training/validation groups of noises and waveforms")
    p.add_argument("--amplitude-scale", type=float, default=32., metavar="FACTOR",
                   help="Divide assembled pregenerated signal/noise inputs in both training "
                        "and validation by FACTOR. Online ml4gw inputs remain unscaled. Default: 32")
    p.add_argument("--pregen-val-fraction", type=float, default=.2)
    p.add_argument("--pregen-workers", type=int, default=2)
    p.add_argument("--output-directory", required=True, type=Path)
    p.add_argument("--no-loss-curve", action="store_true",
                   help="Skip the training-history figure written at the end "
                        "of the run.")
    p.add_argument("--loss-curve-log", action="store_true",
                   help="Use a logarithmic loss axis in that figure.")
    p.add_argument("--resume", type=Path)
    p.add_argument("--rebuild-validation", action="store_true")
    p.add_argument("--save-every-epoch", action="store_true")
    p.add_argument("--no-input-instance-norm", action="store_true",
                   help="Ablation: disable per-detector input normalization")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--train-device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--steps-per-epoch", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--validation-batch-size", type=int, default=256)
    p.add_argument("--pregen-prob-start", type=float, default=.1)
    p.add_argument("--pregen-prob-end", type=float, default=.3)
    p.add_argument("--pregen-ramp-epochs", type=int, default=100)
    p.add_argument("--ema-decay", type=float, default=.999)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--reads-per-chunk", type=int, default=40)
    p.add_argument("--chunk-length", type=float, default=128.)
    p.add_argument("--waveform-microbatch", type=int, default=32)
    p.add_argument("--sample-rate", type=int, default=2048)
    p.add_argument("--kernel-length", type=float, default=1.)
    p.add_argument("--psd-length", type=float, default=16.)
    p.add_argument("--fftlength", type=float, default=2.)
    p.add_argument("--psd-average", choices=["mean", "median"], default="median")
    p.add_argument("--fduration", type=float, default=2.)
    p.add_argument("--highpass", type=float, default=20.)
    p.add_argument("--output-scale", type=float, default=1.)
    p.add_argument("--waveform-duration", type=float, default=8.)
    p.add_argument("--waveform-right-pad", type=float, default=3.)
    p.add_argument("--f-min", type=float, default=20.)
    p.add_argument("--f-ref", type=float, default=20.)
    p.add_argument("--reference-distance", type=float, default=1000.)
    p.add_argument("--mass-min", type=float, default=7.)
    p.add_argument("--mass-max", type=float, default=50.)
    p.add_argument("--spin-max", type=float, default=.99)
    p.add_argument("--snr-low", type=float, default=7.)
    p.add_argument("--snr-high", type=float, default=20.)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if not (0 < args.pregen_val_fraction <= 1 and 0 < args.validation_file_fraction < 1):
        p.error("Validation fractions must lie in (0,1]")
    if not (0 <= args.pregen_prob_start <= 1 and 0 <= args.pregen_prob_end <= 1):
        p.error("Pregenerated mixing probabilities must lie in [0,1]")
    if (args.batch_size < 2 or args.batch_size % 2 or args.epochs < 1
            or args.steps_per_epoch < 1 or args.validation_batch_size < 2):
        p.error("Need an even batch size >=2, positive epochs/steps")
    if args.snr_low <= 0 or args.snr_high <= args.snr_low:
        p.error("Invalid SNR range")
    if not math.isfinite(args.amplitude_scale) or args.amplitude_scale <= 0:
        p.error("--amplitude-scale must be finite and positive")
    if args.sample_rate != 2048 or args.kernel_length != 1:
        p.error("The coincident model requires 1 s at 2048 Hz")
    if args.pregen_workers < 0 or args.pregen_ramp_epochs < 0:
        p.error("Invalid worker count or ramp duration")
    if not 0 < args.ema_decay < 1:
        p.error("EMA decay must lie in (0,1)")
    return args


def main():
    args = parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    log_path = args.output_directory / "training.log"
    logging.basicConfig(level=logging.INFO, force=True,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stderr),
                                  logging.FileHandler(log_path,
                                      mode="a" if args.resume else "w")])
    logging.info("Writing epoch loss/accuracy to %s and %s; TPR to %s",
                 log_path, args.output_directory / "losses.txt",
                 args.output_directory / "metrics.csv")
    fit(args)

    # Training history figure. Drawn from losses.txt, so it can also be
    # regenerated later without retraining:
    #     python -c "from gwsearch.plotting import plot_loss_curves; \
    #                plot_loss_curves('runs/<name>/losses.txt', 'runs/<name>')"
    if not args.no_loss_curve:
        try:
            from gwsearch.plotting import plot_loss_curves
            written = plot_loss_curves(args.output_directory / "losses.txt",
                                       args.output_directory,
                                       log_scale=args.loss_curve_log)
            logging.info("Loss curve written to %s", " and ".join(written))
        except Exception as exc:
            # A missing matplotlib or a display-less node must not discard a
            # finished training run.
            logging.warning("Could not draw the loss curve (%s). The history "
                            "is in %s and can be plotted later.",
                            exc, args.output_directory / "losses.txt")


if __name__ == "__main__":
    main()
