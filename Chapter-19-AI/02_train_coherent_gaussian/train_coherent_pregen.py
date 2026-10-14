"""Train the coherent transformer on a pre-generated dataset.

Used twice in the chapter: on Gaussian-noise examples in Section 1.2, and
again on real-O3a examples in Section 1.4.1. The script does not care which
it is given.

The model reads both detectors together as two input channels. Three
convolutional layers produce 64 tokens, which pass through six transformer
layers of width 90, attention pooling and a two-layer classification head:
664,522 parameters in total. Training minimises cross-entropy; the ranking
statistic is the unbounded softmax replacement, the difference of the two
output scores.

Signal examples are assembled on the fly as ``x = n + rho * h`` with the
noise segment and waveform drawn independently, so no noise realisation is
tied to a class label (Box 1).
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

# Make the repository root importable so ``gwsearch`` resolves whether or not
# the package has been installed with ``pip install -e .``.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from gwsearch.coherent_model import CoherentTransformer, MODEL_DEFAULTS, usr

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
    source = Path(directory)
    if source.is_file():
        if source.suffix.lower() not in (".h5", ".hdf", ".hdf5"):
            raise ValueError(f"Expected an HDF5 file: {source}")
        return [source.resolve()]
    paths = sorted(p.resolve() for p in source.iterdir()
                   if p.suffix.lower() in (".h5", ".hdf", ".hdf5") and p.is_file())
    if not paths:
        raise ValueError(f"No HDF5 files in {directory}")
    return paths
















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
            return self.noises[int(self.noise_ids[idx-len(self.signal_ids)])] / self.scale, 0
        source_idx = int(self.signal_ids[idx])
        if self.deterministic:
            snr = float(np.random.default_rng([self.seed, source_idx]).uniform(self.snr_low, self.snr_high))
        else:
            snr = self.snr_low + (self.snr_high-self.snr_low)*float(torch.rand(()))
        return (self.signal_noises[source_idx] + snr * self.waves[source_idx]) / self.scale, 1


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


def source_loss(logits, labels):
    # Binary CE on two logits is equivalent to BCEWithLogits(z_signal-z_noise).
    return F.binary_cross_entropy_with_logits(usr(logits), labels.float())


@torch.inference_mode()
def validate(model, pregen, device, batch_size):
    result = {}
    model.eval()
    for name, ds in (("pregen", pregen),):
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        total_loss, correct, count = 0., 0, 0
        all_scores, all_labels = [], []
        for batch, labels in tqdm(loader, desc=f"Validating {name}", leave=False,
                                  disable=not sys.stderr.isatty()):
            # PairedPregen applies the configured divisor on reading.
            batch = batch.to(device, non_blocking=True).float()
            labels = labels.to(device).long()
            logits = model(batch)
            loss = source_loss(logits, labels)
            total_loss += float(loss) * len(batch)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            count += len(batch)
            all_scores.append(usr(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        scores, ys = np.concatenate(all_scores), np.concatenate(all_labels)
        negatives, positives = scores[ys == 0], scores[ys == 1]
        cutoff = np.quantile(negatives, 0.999)
        result[name] = dict(loss=total_loss/count, accuracy=correct/count,
                            tpr_fpr_1e3=float((positives > cutoff).mean()))
    return result


def save_checkpoint(path, model, weights, args, epoch, metric, extra=None):
    payload = dict(model_config=model.config, model_state_dict=weights,
                   input_scale=args.amplitude_scale, training_data_source="pregen",
                   pregen_training_scale=args.amplitude_scale, class_order=["noise", "signal"],
                   pregen_preprocessing=f"stored whitened HDF, fixed PSD-associated pairing, divide by {args.amplitude_scale:g}",
                   eval_preprocessing=f"PSD 4s, filter 4s, cutoff 15Hz, FIR high-pass 20Hz, divide by {args.amplitude_scale:g}",
                   epoch=epoch, validation_loss=metric, weights_kind="ema")
    if extra: payload.update(extra)
    tmp = str(path) + ".partial"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def fit(args):
    if not (0 < args.pregen_val_fraction <= 1):
        raise ValueError("Validation fraction must be in (0,1]")
    seed_everything(args.seed)
    out = Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=True)
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
    model = CoherentTransformer(**model_config).to(args.train_device)
    logging.info("Joint model has %d trainable parameters", sum(p.numel() for p in model.parameters()))
    ema = EMAShadow(model, args.ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                   weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate,
        pct_start=0.1, total_steps=args.epochs*args.steps_per_epoch)
    best, start_epoch = float("inf"), 0
    if args.resume:
        saved = torch.load(args.resume, map_location=args.train_device, weights_only=False)
        if saved.get("training_data_source") != "pregen":
            raise ValueError("Resume requires a pregenerated-only training checkpoint")
        if saved.get("model_config") != model.config or saved.get("weights_kind") != "raw":
            raise ValueError("Resume requires last_training.pt from this exact architecture")
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
            f.write("epoch,step,training_loss,pregen_val_loss,pregen_tpr_fpr1e-3,pregen_batches\n")
        with losses_path.open("w") as f:
            f.write("# epoch  train_loss  train_acc  pregen_val_loss  pregen_val_acc\n")
    elif not csv.exists():
        raise FileNotFoundError(f"Cannot resume without {csv}")
    elif not losses_path.exists():
        # Older checkpoints have metrics.csv but no accuracy history.
        # Start this file at the resumed epoch without fabricating old values.
        with losses_path.open("w") as f:
            f.write("# epoch  train_loss  train_acc  pregen_val_loss  pregen_val_acc\n")
    for epoch in range(start_epoch, args.epochs):
        seed_everything(args.seed + 1000 + epoch)
        model.train()
        total_loss, train_correct, train_seen = 0., 0, 0
        for _ in tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch+1}/{args.epochs}",
                      disable=not sys.stderr.isatty()):
            try:
                strain, labels = next(pregen_iter)
            except StopIteration:
                pregen_iter = iter(pregen_loader)
                strain, labels = next(pregen_iter)
            strain = strain.to(args.train_device, non_blocking=True).float()
            labels = labels.to(args.train_device).long()
            optimizer.zero_grad(set_to_none=True)
            logits = model(strain)
            loss = source_loss(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch+1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
            optimizer.step()
            schedule.step()
            ema.update(model)
            total_loss += float(loss.detach())
            train_correct += int((logits.detach().argmax(dim=1) == labels).sum().item())
            train_seen += labels.numel()
        raw_weights = {k:v.detach().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(ema.state_dict(model), strict=True)
        val = validate(model, pregen_val, args.train_device, args.validation_batch_size)
        metric = val["pregen"]["loss"]
        if metric < best:
            best = metric
            save_checkpoint(out/"best_ema.pt", model, model.state_dict(), args, epoch+1, metric)
        if args.save_every_epoch:
            save_checkpoint(out/f"epoch_{epoch+1:04d}_ema.pt", model,
                            model.state_dict(), args, epoch+1, metric)
        model.load_state_dict(raw_weights, strict=True)
        save_checkpoint(out/"last_training.pt", model, raw_weights, args,
            epoch+1, metric, extra=dict(weights_kind="raw", total_steps=args.epochs*args.steps_per_epoch,
                 best_metric=best, ema_shadow=ema.shadow,
                 optimizer_state_dict=optimizer.state_dict(),
                 scheduler_state_dict=schedule.state_dict()))
        with csv.open("a") as f:
            f.write(f"{epoch+1},{(epoch+1)*args.steps_per_epoch},"
                    f"{total_loss/args.steps_per_epoch:.8f},"
                    f"{val['pregen']['loss']:.8f},"
                    f"{val['pregen']['tpr_fpr_1e3']:.6f},"
                    f"{args.steps_per_epoch}\n")
        train_loss = total_loss / args.steps_per_epoch
        train_accuracy = train_correct / train_seen
        with losses_path.open("a") as f:
            f.write(f"{epoch+1:04d}  {train_loss:.6f}  {train_accuracy:.6f}  "
                    f"{val['pregen']['loss']:.6f}  {val['pregen']['accuracy']:.6f}\n")
        logging.info("epoch %04d | train loss %.5f acc %.4f | "
                     "pregen valid loss %.5f acc %.4f | "
                     "best val loss %.5f | pregen TPR@FPR=1e-3 %.4f",
                     epoch+1, train_loss, train_accuracy,
                     val["pregen"]["loss"], val["pregen"]["accuracy"], best,
                     val["pregen"]["tpr_fpr_1e3"])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pregen-dir", required=True, type=Path,
                   help="Single HDF5 file or directory of HDF5 files with training/validation groups of noises and waveforms")
    p.add_argument("--amplitude-scale", type=float, default=32., metavar="FACTOR",
                   help="Divide assembled pregenerated signal/noise inputs in both training "
                        "and validation by FACTOR. Default: 32")
    p.add_argument("--pregen-val-fraction", type=float, default=.2)
    p.add_argument("--pregen-workers", type=int, default=2)
    p.add_argument("--output-directory", required=True, type=Path)
    p.add_argument("--no-loss-curve", action="store_true",
                   help="Skip the training-history figure written at the end "
                        "of the run.")
    p.add_argument("--loss-curve-log", action="store_true",
                   help="Use a logarithmic loss axis in that figure.")
    p.add_argument("--resume", type=Path)
    p.add_argument("--save-every-epoch", action="store_true")
    p.add_argument("--no-input-instance-norm", action="store_true",
                   help="Ablation: disable per-detector input normalization")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--train-device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--steps-per-epoch", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--validation-batch-size", type=int, default=256)
    p.add_argument("--ema-decay", type=float, default=.999)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--sample-rate", type=int, default=2048)
    p.add_argument("--kernel-length", type=float, default=1.)
    p.add_argument("--snr-low", type=float, default=7.)
    p.add_argument("--snr-high", type=float, default=20.)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if not (0 < args.pregen_val_fraction <= 1):
        p.error("Validation fractions must lie in (0,1]")
    if (args.batch_size < 2 or args.batch_size % 2 or args.epochs < 1
            or args.steps_per_epoch < 1 or args.validation_batch_size < 2):
        p.error("Need an even batch size >=2, positive epochs/steps")
    if args.snr_low <= 0 or args.snr_high <= args.snr_low:
        p.error("Invalid SNR range")
    if not math.isfinite(args.amplitude_scale) or args.amplitude_scale <= 0:
        p.error("--amplitude-scale must be finite and positive")
    if args.sample_rate != 2048 or args.kernel_length != 1:
        p.error("The current coherent model requires 1 s at 2048 Hz")
    if args.pregen_workers < 0:
        p.error("Invalid worker count")
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
