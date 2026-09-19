#!/usr/bin/env python
"""
Train the baseline transformer on pregenerated gravitational-wave data.

This trains the *deliberately plain* model of the chapter. It is not intended
to be competitive on the MLGWSC-1 benchmark: it is intended to be the simplest
thing a machine-learning practitioner would reasonably build, so that we can
then measure how far that falls short of a usable search and diagnose why.

------------------------------------------------------------------------------
WHAT MAKES THIS A "BASELINE"
------------------------------------------------------------------------------
Three choices are deliberately naive, and each is revisited later in the
chapter. They are flagged in the code with "NOTE (deliberate ...)".

1. COHERENT configuration. Both detectors are fed to one network as two input
   channels, so the network forms its statistic jointly across detectors.
   This is the obvious design, and it has a hidden cost: estimating the
   background by time slides requires re-running the network on every shifted
   pairing, because the statistic cannot be recomputed from cached
   single-detector outputs.

2. Trained and selected on balanced classification loss and accuracy. A search
   operates at a false-alarm rate of about one per month, which over a month of
   data at a 0.1 s stride is roughly one false positive in 2.6e7 trials.
   Accuracy on a balanced validation set says almost nothing about behaviour
   that far into the tail. We report accuracy here precisely so that the
   chapter can show how misleading it is.

3. A single fixed 1 s window and a plain fixed-stride tokenizer. Longer,
   lower-mass signals do not fit, so the model is biased against them from the
   outset.

What is NOT deliberate is overfitting to a finite set of noise realisations.
By default each example draws a random noise segment, and a random waveform if
it is a signal, so no noise realisation is tied to a label and the number of
distinct examples is the product of the two pools rather than their sum. Add
AdamW weight decay, a cosine learning-rate schedule and early stopping, and the
validation loss tracks the training loss instead of turning upward. See
``--fixed-pairing``, ``--weight-decay`` and ``--patience``.

------------------------------------------------------------------------------
HOW THE DATA IS ASSEMBLED
------------------------------------------------------------------------------
``generate_dataset.py`` stores noise and signals separately: entry ``i`` of
``waveforms`` is normalised to network SNR = 1 and belongs with entry ``i`` of
``noises``. A training example is built on the fly as

    x = noises[i] + snr * waveforms[i]        for i < n_injections   (label 1)
    x = noises[i]                             otherwise              (label 0)

with ``snr`` drawn fresh each time from ``--snr``. The same stored example is
therefore seen at many different amplitudes over the course of training.

------------------------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------------------------
Requires: torch, numpy, h5py, tqdm

    # Stage 1 of the chapter: injections in Gaussian noise, a single file
    python train_baseline.py -d data_gaussian.h5 -o runs/baseline_gaussian \
        --epochs 50 --train-device cuda --verbose

    # Stage 2: injections in real O3 noise. If prepare_real_noise.py was run
    # with --chunk-size, generate_dataset.py produces one file per chunk;
    # pass them all, or just the directory, or a glob.
    python train_baseline.py -d data_o3_*.h5   -o runs/baseline_o3 --verbose
    python train_baseline.py -d data_o3_files/ -o runs/baseline_o3 --verbose
    python train_baseline.py -d "data_o3_*.h5" -o runs/baseline_o3 --verbose

    # Quick smoke test before committing to a long run
    python train_baseline.py -d data_gaussian.h5 -o runs/smoke --epochs 1 --verbose

    # Reproduce the overfitting failure deliberately, for the chapter figure
    python train_baseline.py -d data_gaussian.h5 -o runs/overfit \
        --fixed-pairing --weight-decay 0 --patience 0 --lr-schedule none \
        --epochs 200 --verbose

    # Resume from, or fine-tune, an existing checkpoint
    python train_baseline.py -d data_o3.h5 -o runs/finetune \
        --weights runs/baseline_gaussian/best_state_dict.pt --verbose

Outputs written to the ``-o`` directory:
    losses.txt          epoch, train loss/accuracy, validation loss/accuracy
    loss_curve.pdf      publication-quality training/validation loss curve
    loss_curve.png      the same figure as a raster image
    best_state_dict.pt  weights at the lowest validation loss
    last_state_dict.pt  weights after the final epoch
    config.json         the full run configuration, for reproducibility

The loss curve is drawn automatically when training finishes. To restyle it
without repeating the run -- for example to add an accuracy panel or switch to
a logarithmic axis -- regenerate it from the saved history:

    python train_baseline.py -o runs/baseline_gaussian --plot-only \
        --plot-accuracy --log-loss

------------------------------------------------------------------------------
Adapted for this chapter from the MLGWSC-1 training script by Ondrej Zelenka
(https://github.com/ondrzel/ml-gw-search), Apache License 2.0.
Copyright 2022 Ondrej Zelenka. Modifications copyright 2026 Chayan Chatterjee.
------------------------------------------------------------------------------
"""

from argparse import ArgumentParser
import copy
import glob
import json
import logging
import os
import os.path
import random
import sys

import h5py
import numpy as np
import torch
from tqdm import tqdm


DTYPE = torch.float32

# Class convention used throughout: 0 = pure noise, 1 = signal present.
NOISE_CLASS = 0
SIGNAL_CLASS = 1


def usr_statistic(logits):
    """Unbounded Softmax Replacement: the signal-versus-noise log-odds.

    This is the ranking statistic used at search time, and it is defined here
    rather than in the evaluation script so that the two cannot disagree.

    Note that the network is NOT trained on this quantity by name -- but it is
    exactly what the loss optimises. For two classes the softmax depends only
    on the difference of the logits,

        p(signal) = e^{z_s} / (e^{z_s} + e^{z_n}) = sigmoid(z_s - z_n),

    so cross-entropy on the logit pair is identical to binary cross-entropy on
    sigmoid(USR). Training therefore calibrates USR as a log-odds directly, and
    reading it out at evaluation time introduces nothing new.

    Its virtue over the softmax probability is range. A softmax saturates at
    1.0 in float32 once the logit difference exceeds about 17, whereas the
    significance of a candidate keeps rising well beyond that; the difference
    stays resolved, which is what the low false-alarm-rate tail depends on.
    """
    return logits[..., SIGNAL_CLASS] - logits[..., NOISE_CLASS]


# =============================================================================
# Reproducibility
# =============================================================================
def seed_everything(seed):
    """Seed Python, NumPy and PyTorch so a run can be repeated."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Locating the dataset files
# =============================================================================
DATASET_EXTENSIONS = (".h5", ".hdf", ".hdf5")


def resolve_dataset_files(paths):
    """Expand user-supplied paths into a sorted list of dataset files.

    Accepts any mixture of:
      * a single file            data_gaussian.h5
      * several files            data_o3_0000.h5 data_o3_0001.h5
      * a directory              data_o3_files/      (searched for HDF5 files)
      * a glob pattern           "data_o3_*.h5"      (quoted, so the shell
                                                      leaves it to us)

    This is what makes the script agnostic to how the data was produced: one
    file for the Gaussian-noise study, many for real noise sliced in chunks.
    """
    resolved = []
    for path in paths:
        expanded = os.path.expanduser(path)
        if os.path.isdir(expanded):
            found = [p for p in sorted(glob.glob(os.path.join(expanded, "*")))
                     if p.lower().endswith(DATASET_EXTENSIONS)]
            if not found:
                raise FileNotFoundError(
                    f"Directory '{path}' contains no files ending in "
                    f"{', '.join(DATASET_EXTENSIONS)}.")
            resolved.extend(found)
        elif any(ch in expanded for ch in "*?["):
            found = sorted(glob.glob(expanded))
            if not found:
                raise FileNotFoundError(f"Pattern '{path}' matched no files.")
            resolved.extend(found)
        elif os.path.isfile(expanded):
            resolved.append(expanded)
        else:
            raise FileNotFoundError(f"'{path}' is not a file, directory or "
                                    f"matching pattern.")

    # Preserve order but drop duplicates, which are easy to introduce when
    # combining a glob with an explicit filename.
    unique = list(dict.fromkeys(os.path.abspath(p) for p in resolved))
    if not unique:
        raise FileNotFoundError("No dataset files found.")
    return unique


# =============================================================================
# Dataset
# =============================================================================
class InjectionDataset(torch.utils.data.Dataset):
    """Assembles training examples from stored noise and unit-SNR waveforms.

    The first ``n_injections`` entries carry a signal; the rest are pure noise.
    A fresh SNR is drawn on every access, so the effective training set is much
    larger than the number of stored waveforms.

    Parameters
    ----------
    noises : ndarray, shape (n_total, n_detectors, sample_length)
    waveforms : ndarray, shape (n_injections, n_detectors, sample_length)
        ``n_injections`` must not exceed ``n_total``; entry ``i`` pairs with
        ``noises[i]``.
    fixed_pairing : bool
        If False (the default), each example draws a RANDOM noise segment and,
        for the signal class, a RANDOM waveform.

        This matters more than it looks. With fixed pairing, noise realisation
        ``i`` carries a signal if and only if ``i < n_injections``, so every
        noise segment is permanently bound to one label. A network with enough
        capacity will eventually memorise which noise realisations belong to
        which class instead of learning what a signal looks like, and the
        validation loss turns upward while the training loss keeps falling.
        Random pairing breaks that association and, as a bonus, turns
        n_injections x n_noise into the number of distinct examples.

        Set True to restore the original one-to-one pairing. That is exact for
        real noise, where each waveform was whitened with the PSD of its own
        segment; under random pairing the injected SNR becomes approximate if
        the PSDs differ appreciably between segments. For Gaussian noise drawn
        from a single analytic PSD, random pairing is exact.
    deterministic : bool
        If True, the SNR of each example is a fixed function of its index and
        pairing is one-to-one, so the dataset returns identical examples every
        epoch. Used for validation, where a moving target would make the loss
        curve noisy and early stopping unreliable.
    snr_range : (float, float)
        Network SNR is drawn uniformly from this interval.

        NOTE (deliberate simplification): a uniform SNR over a narrow range is
        the simplest choice, but it does not reflect an astrophysical
        population, in which quiet signals vastly outnumber loud ones. It also
        sets a hard lower edge below which signals are labelled "signal" while
        being indistinguishable from noise -- the class-overlap problem.
    store_device : str
        Where the arrays live. 'cuda' keeps the whole dataset in GPU memory,
        which is much faster but needs the data to fit. Batches are moved to
        the training device in the training loop, not here.
    """

    def __init__(self, noises, waveforms, snr_range=(5.0, 15.0),
                 store_device="cpu", seed=0, fixed_pairing=False,
                 deterministic=False):
        super().__init__()
        if len(waveforms) > len(noises):
            raise ValueError(
                f"More waveforms ({len(waveforms)}) than noise segments "
                f"({len(noises)}); the file does not follow the layout written "
                f"by generate_dataset.py.")
        self.noises = torch.from_numpy(np.asarray(noises)).to(
            dtype=DTYPE, device=store_device)
        self.waveforms = torch.from_numpy(np.asarray(waveforms)).to(
            dtype=DTYPE, device=store_device)
        self.n_injections = len(self.waveforms)
        self.snr_range = tuple(snr_range)
        self.seed = seed
        self.fixed_pairing = fixed_pairing
        self.deterministic = deterministic
        self._rng = None

    def as_deterministic(self):
        """A view of the same data that yields identical examples every epoch.

        The returned object shares the underlying tensors, so this costs no
        extra memory. Used to build a stable validation split.
        """
        clone = copy.copy(self)
        clone.deterministic = True
        clone.fixed_pairing = True
        clone._rng = None
        return clone

    @property
    def rng(self):
        """A generator that is distinct in every DataLoader worker process.

        Workers are forked copies, so a generator created in __init__ would be
        duplicated and every worker would draw the same sequence of SNRs. It is
        therefore created lazily, seeded by the worker id as well.
        """
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            worker_id = 0 if info is None else info.id
            self._rng = np.random.default_rng([self.seed, worker_id])
        return self._rng

    def __len__(self):
        return len(self.noises)

    def __getitem__(self, index):
        # Whether this entry carries a signal is fixed by its index, which keeps
        # the class balance exactly as generated. WHICH noise (and which
        # waveform) is used is not, unless fixed_pairing is set.
        is_signal = index < self.n_injections
        if self.deterministic:
            # Same example every epoch: SNR is a pure function of the index.
            rng = np.random.default_rng([self.seed, int(index)])
            noise_index, wave_index = index, index
        else:
            rng = self.rng
            if self.fixed_pairing:
                noise_index, wave_index = index, index
            else:
                noise_index = int(rng.integers(len(self.noises)))
                wave_index = int(rng.integers(self.n_injections))

        if is_signal:
            snr = rng.uniform(*self.snr_range)
            sample = self.noises[noise_index] + snr * self.waveforms[wave_index]
            return sample, SIGNAL_CLASS
        return self.noises[noise_index], NOISE_CLASS


def load_group(path, group):
    """Read ``noises`` and ``waveforms`` from one group, or return None."""
    with h5py.File(path, "r") as handle:
        if group not in handle:
            return None
        return handle[group]["noises"][()], handle[group]["waveforms"][()]


def build_datasets(files, snr_range, store_device, validation_fraction, seed,
                   fixed_pairing=False):
    """Load every file and return (training_dataset, validation_dataset).

    Files that contain a ``validation`` group contribute to the validation set.
    If NO file has one -- which is the default output of ``generate_dataset.py``
    now that ``--validation-samples`` is opt-in -- a random fraction of the
    training data is held out instead, and a warning is issued.
    """
    train_parts, valid_parts = [], []
    total_injections = total_samples = 0

    for index, path in enumerate(files):
        training = load_group(path, "training")
        if training is None:
            raise KeyError(f"File '{path}' has no 'training' group.")
        noises, waveforms = training
        total_samples += len(noises)
        total_injections += len(waveforms)
        train_parts.append(InjectionDataset(
            noises, waveforms, snr_range, store_device, seed=seed + index,
            fixed_pairing=fixed_pairing))
        logging.info("  %s: %i samples (%i injections, %i pure noise)",
                     os.path.basename(path), len(noises), len(waveforms),
                     len(noises) - len(waveforms))

        validation = load_group(path, "validation")
        if validation is not None:
            v_noises, v_waveforms = validation
            # Validation is always deterministic: identical examples every
            # epoch, so the loss curve reflects the model and not the draw.
            valid_parts.append(InjectionDataset(
                v_noises, v_waveforms, snr_range, store_device,
                seed=seed + 10_000 + index, deterministic=True))

    train_ds = torch.utils.data.ConcatDataset(train_parts)

    signal_fraction = total_injections / max(total_samples, 1)
    logging.info("Loaded %i training samples from %i file(s); "
                 "%.1f%% carry a signal.",
                 total_samples, len(files), 100.0 * signal_fraction)
    # NOTE: a heavily imbalanced training set biases the model towards the
    # majority class. generate_dataset.py produces a balanced set by default.
    if not 0.2 <= signal_fraction <= 0.8:
        logging.warning("Training set is strongly imbalanced (%.1f%% signals). "
                        "Consider regenerating with matched injection and "
                        "noise counts.", 100.0 * signal_fraction)

    if valid_parts:
        valid_ds = torch.utils.data.ConcatDataset(valid_parts)
        logging.info("Using the stored validation split (%i samples).",
                     len(valid_ds))
        return train_ds, valid_ds

    # Fall back to holding out part of the training data. The held-out indices
    # are served from deterministic views of the SAME tensors, so validation
    # examples are stable across epochs while training examples stay randomised.
    n_total = len(train_ds)
    n_valid = int(round(validation_fraction * n_total))
    if n_valid < 1:
        raise ValueError("No validation group found and --validation-fraction "
                         "is too small to hold anything out.")
    mirror_ds = torch.utils.data.ConcatDataset(
        [part.as_deterministic() for part in train_parts])
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n_total, generator=generator).tolist()
    valid_indices, train_indices = order[:n_valid], order[n_valid:]
    train_ds = torch.utils.data.Subset(train_ds, train_indices)
    valid_ds = torch.utils.data.Subset(mirror_ds, valid_indices)
    logging.warning(
        "No 'validation' group found; holding out %i of %i training samples "
        "(%.0f%%) at random.", n_valid, n_total,
        100.0 * validation_fraction)
    # NOTE (important for real noise): prepare_real_noise.py cuts slices that
    # overlap by 50%, so neighbouring samples are near-duplicates. A random
    # holdout then puts nearly identical data on both sides of the split and
    # the validation loss becomes optimistic. For real-noise runs, generate a
    # genuine validation split instead, with
    #     generate_dataset.py --validation-samples N_INJ N_NOISE
    # built from noise that prepare_real_noise.py split off beforehand.
    logging.warning("If this is real-noise data with overlapping slices, the "
                    "random holdout is optimistic; see the note in the source.")
    return train_ds, valid_ds


# =============================================================================
# Model
# =============================================================================
class BaselineTransformer(torch.nn.Module):
    """A small, plain transformer classifier for whitened two-detector strain.

    The architecture is the minimum that could be called a transformer:

        Conv1d tokenizer  -- cuts the 1 s input into non-overlapping patches
                             and embeds each one
        + learned positional embedding
        Transformer encoder stack
        mean pooling over tokens
        linear head -> 2 logits  (noise, signal)

    There is no multi-scale tokenizer, no auxiliary loss, no weight averaging
    and no per-detector structure. With the defaults it has well under a
    million parameters and trains in minutes.

    NOTE (deliberate design): the two detectors enter as two channels of a
    single network, i.e. a COHERENT configuration. The network can therefore
    use cross-detector consistency directly, but its output cannot be
    decomposed into per-detector quantities. Every time slide needs a fresh
    forward pass over the shifted data, which is what makes background
    estimation expensive for this design.
    """

    def __init__(self, n_detectors=2, sample_length=2048, patch_size=32,
                 d_model=128, n_heads=4, n_layers=4, dropout=0.1):
        super().__init__()
        if sample_length % patch_size != 0:
            raise ValueError(f"sample_length ({sample_length}) must be divisible "
                             f"by patch_size ({patch_size}).")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by "
                             f"n_heads ({n_heads}).")
        self.config = dict(n_detectors=n_detectors, sample_length=sample_length,
                           patch_size=patch_size, d_model=d_model,
                           n_heads=n_heads, n_layers=n_layers, dropout=dropout)
        n_tokens = sample_length // patch_size

        # One strided convolution turns (batch, n_detectors, sample_length)
        # into n_tokens embeddings of width d_model. At 2048 Hz with
        # patch_size 32, each token covers 15.6 ms of strain.
        self.tokenizer = torch.nn.Conv1d(n_detectors, d_model,
                                         kernel_size=patch_size,
                                         stride=patch_size)
        # Attention is permutation invariant, so the token order has to be
        # supplied explicitly. A learned embedding is the simplest option.
        self.positional = torch.nn.Parameter(torch.zeros(1, n_tokens, d_model))
        torch.nn.init.normal_(self.positional, std=0.02)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer,
                                                   num_layers=n_layers)
        self.norm = torch.nn.LayerNorm(d_model)
        self.head = torch.nn.Linear(d_model, 2)

    def forward(self, x):
        """x : (batch, n_detectors, sample_length) -> (batch, 2) logits."""
        tokens = self.tokenizer(x).transpose(1, 2)     # (batch, n_tokens, d_model)
        tokens = tokens + self.positional
        tokens = self.encoder(tokens)
        pooled = self.norm(tokens).mean(dim=1)         # mean pool over tokens
        return self.head(pooled)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(path, model):
    """Store weights together with the architecture needed to rebuild them."""
    torch.save({"state_dict": model.state_dict(), "config": model.config}, path)


def load_checkpoint(path, device, fallback_config=None):
    """Rebuild a model saved by ``save_checkpoint``.

    Checkpoints written by this script carry their own architecture, so they
    reload correctly whatever the command line says. A bare ``state_dict`` from
    other tooling can also be loaded, but only if ``fallback_config`` describes
    the architecture it belongs to.
    """
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload and "config" in payload:
        model = BaselineTransformer(**payload["config"])
        model.load_state_dict(payload["state_dict"])
    elif fallback_config is not None:
        model = BaselineTransformer(**fallback_config)
        model.load_state_dict(payload)
    else:
        raise ValueError(
            f"'{path}' is a bare state_dict with no stored architecture, and no "
            f"fallback configuration was supplied.")
    return model.to(device)


# =============================================================================
# Loss curve
# =============================================================================
# Colours chosen to stay distinguishable in greyscale and to colour-vision
# deficient readers (Wong palette).
TRAIN_COLOUR = "#0072B2"
VALID_COLOUR = "#D55E00"


def plot_loss_curves(losses_path, output_directory, plot_accuracy=False,
                     log_scale=False, dpi=300, figsize=(6.5, 4.0)):
    """Write a publication-quality training-history figure.

    Reads the whitespace-separated history written during training and saves
    the result twice: a vector PDF, which is what should be included in a
    manuscript or book chapter, and a PNG for quick inspection.

    Parameters
    ----------
    losses_path : str
        The ``losses.txt`` written by :func:`train`.
    output_directory : str
        Where ``loss_curve.pdf`` and ``loss_curve.png`` are written.
    plot_accuracy : bool
        Add a second panel showing classification accuracy. Useful for the
        discussion of why accuracy is the wrong figure of merit for a search:
        it typically saturates near 1 while the loss is still improving, and
        neither quantity says anything about behaviour at the very low
        false-alarm rates a real search operates at.
    log_scale : bool
        Use a logarithmic loss axis, which helps when the loss falls by more
        than an order of magnitude over training.

    Returns
    -------
    list of str
        The files written.
    """
    # Imported lazily and with a non-interactive backend, so that training
    # itself never depends on matplotlib or on a display being available.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = np.loadtxt(losses_path)
    if history.size == 0:
        raise ValueError(f"'{losses_path}' contains no completed epochs.")
    if history.ndim == 1:                      # a single epoch
        history = history[np.newaxis, :]
    epochs, train_loss, train_acc, valid_loss, valid_acc = history.T

    style = {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 10,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
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

    with plt.rc_context(style):
        n_panels = 2 if plot_accuracy else 1
        height = figsize[1] * (1.5 if plot_accuracy else 1.0)
        fig, axes = plt.subplots(n_panels, 1, figsize=(figsize[0], height),
                                 sharex=True, squeeze=False)
        axes = [row[0] for row in axes]

        # Markers are helpful for a short run and become clutter for a long one.
        marker = "o" if len(epochs) <= 40 else None

        ax = axes[0]
        ax.plot(epochs, train_loss, color=TRAIN_COLOUR, lw=1.6, marker=marker,
                ms=3.5, label="Training")
        ax.plot(epochs, valid_loss, color=VALID_COLOUR, lw=1.6, marker=marker,
                ms=3.5, label="Validation")

        # Mark the epoch that was saved as best_state_dict.pt.
        best = int(np.argmin(valid_loss))
        ax.axvline(epochs[best], color="0.65", ls=":", lw=1.0, zorder=0)
        ax.plot(epochs[best], valid_loss[best], ls="none", marker="*", ms=12,
                color=VALID_COLOUR, mec="black", mew=0.5, zorder=5,
                label=f"Best: epoch {int(epochs[best])} "
                      f"(loss {valid_loss[best]:.4f})")

        ax.set_ylabel("Cross-entropy loss")
        if log_scale:
            ax.set_yscale("log")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(loc="best")

        if plot_accuracy:
            ax = axes[1]
            ax.plot(epochs, train_acc, color=TRAIN_COLOUR, lw=1.6,
                    marker=marker, ms=3.5, label="Training")
            ax.plot(epochs, valid_acc, color=VALID_COLOUR, lw=1.6,
                    marker=marker, ms=3.5, label="Validation")
            ax.axvline(epochs[best], color="0.65", ls=":", lw=1.0, zorder=0)
            ax.set_ylabel("Accuracy")
            ax.grid(alpha=0.25, lw=0.5)
            ax.legend(loc="best")

        axes[-1].set_xlabel("Epoch")
        axes[-1].set_xlim(epochs.min(), max(epochs.max(), epochs.min() + 1))
        fig.align_ylabels(axes)
        fig.tight_layout()

        written = []
        for extension in ("pdf", "png"):
            path = os.path.join(output_directory, f"loss_curve.{extension}")
            fig.savefig(path, dpi=dpi)
            written.append(path)
        plt.close(fig)

    return written


# =============================================================================
# Training
# =============================================================================
def run_epoch(model, loader, loss_fn, device, optimizer=None, clip_norm=None,
              verbose=False, desc=""):
    """One pass over a loader. Trains if an optimizer is given, else evaluates.

    Returns (mean loss, accuracy).
    """
    training = optimizer is not None
    model.train(training)

    running_loss, n_correct, n_seen = 0.0, 0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for samples, labels in tqdm(loader, desc=desc, leave=False,
                                    disable=not verbose, ascii=True):
            samples = samples.to(device=device, dtype=DTYPE)
            labels = labels.to(device=device, dtype=torch.long)

            logits = model(samples)
            loss = loss_fn(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   max_norm=clip_norm)
                optimizer.step()

            batch = labels.numel()
            running_loss += loss.item() * batch
            n_correct += (logits.argmax(dim=1) == labels).sum().item()
            n_seen += batch

    return running_loss / max(n_seen, 1), n_correct / max(n_seen, 1)


def train(model, train_ds, valid_ds, output_dir, device, batch_size=128,
          learning_rate=1e-4, epochs=50, clip_norm=100.0, num_workers=0,
          save_every_epoch=False, verbose=False, make_plot=True,
          plot_accuracy=False, log_loss=False, weight_decay=1e-2,
          patience=20, lr_schedule="cosine"):
    """Fit the model, writing checkpoints and a loss history to ``output_dir``."""
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        drop_last=False)
    valid_loader = torch.utils.data.DataLoader(
        valid_ds, batch_size=max(batch_size, 256), shuffle=False,
        num_workers=num_workers)

    # Cross-entropy on raw logits, rather than binary cross-entropy on softmax
    # probabilities. It is numerically stable without the epsilon-regularised
    # workaround the original code needed, and it keeps the two logits
    # available. Their difference, z_signal - z_noise, is the unbounded
    # log-odds used as the ranking statistic at search time -- unlike a softmax
    # probability, it does not saturate at 1 for very significant events.
    loss_fn = torch.nn.CrossEntropyLoss()
    # AdamW rather than Adam: it applies weight decay as true L2 regularisation
    # on the weights rather than folding it into the gradient, which is the
    # standard choice for transformers and the first line of defence against
    # the model memorising a finite training set.
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)

    scheduler = None
    if lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs)
    elif lr_schedule == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=max(1, patience // 3))

    losses_path = os.path.join(output_dir, "losses.txt")
    best_path = os.path.join(output_dir, "best_state_dict.pt")
    last_path = os.path.join(output_dir, "last_state_dict.pt")

    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    with open(losses_path, "w", buffering=1) as history:
        history.write("# epoch  train_loss  train_acc  valid_loss  valid_acc\n")
        for epoch in tqdm(range(1, epochs + 1), desc="Training",
                          disable=not verbose, ascii=True):
            train_loss, train_acc = run_epoch(
                model, train_loader, loss_fn, device, optimizer=optimizer,
                clip_norm=clip_norm, verbose=verbose, desc="  train")
            valid_loss, valid_acc = run_epoch(
                model, valid_loader, loss_fn, device, verbose=verbose,
                desc="  valid")

            history.write(f"{epoch:04d}  {train_loss:.6f}  {train_acc:.6f}  "
                          f"{valid_loss:.6f}  {valid_acc:.6f}\n")
            logging.info("epoch %04i | train loss %.5f acc %.4f | "
                         "valid loss %.5f acc %.4f",
                         epoch, train_loss, train_acc, valid_loss, valid_acc)

            if scheduler is not None:
                if lr_schedule == "plateau":
                    scheduler.step(valid_loss)
                else:
                    scheduler.step()

            save_checkpoint(last_path, model)
            if save_every_epoch:
                save_checkpoint(
                    os.path.join(output_dir, f"state_dict_e_{epoch:04d}.pt"),
                    model)
            # NOTE (deliberate simplification): the "best" model is the one with
            # the lowest validation loss. That is the standard machine-learning
            # criterion, and it is not the quantity we care about -- which is
            # sensitivity at a fixed false-alarm rate. The chapter returns to
            # this once a background has been estimated.
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                save_checkpoint(best_path, model)
            else:
                epochs_without_improvement += 1
                # Early stopping. Past the point where validation loss stops
                # improving, further epochs only fit the training set harder;
                # continuing produces a model strictly worse than the one
                # already saved in best_state_dict.pt.
                if patience and epochs_without_improvement >= patience:
                    logging.info(
                        "Early stopping at epoch %i: no improvement for %i "
                        "epochs (best was epoch %i, loss %.5f).",
                        epoch, patience, best_epoch, best_loss)
                    break

    logging.info("Training finished. Best validation loss %.5f at epoch %i "
                 "-> %s", best_loss, best_epoch, best_path)

    # The figure is a convenience, not part of the result. A missing matplotlib
    # or a plotting error must never discard a training run that has just cost
    # hours of compute, so failures here are reported and swallowed. The plot
    # can always be regenerated later with --plot-only.
    if make_plot:
        try:
            written = plot_loss_curves(losses_path, output_dir,
                                       plot_accuracy=plot_accuracy,
                                       log_scale=log_loss)
            logging.info("Wrote loss curve: %s", ", ".join(written))
        except Exception as exc:                       # noqa: BLE001
            logging.warning("Could not write the loss curve (%s). Regenerate "
                            "later with --plot-only.", exc)

    return model


# =============================================================================
# Command-line interface
# =============================================================================
def main():
    parser = ArgumentParser(
        description="Train the baseline transformer on data written by "
                    "generate_dataset.py.")

    parser.add_argument("-d", "--dataset-file", type=str, nargs="+",
                        default=None, metavar="PATH",
                        help="One or more dataset files, a directory of them, "
                             "or a quoted glob pattern. Single-file and "
                             "multi-file inputs are handled identically.")
    parser.add_argument("-o", "--output-directory", type=str, required=True,
                        help="Directory for checkpoints and the loss history. "
                             "Created if it does not exist.")
    parser.add_argument("-w", "--weights", type=str, default=None,
                        help="Checkpoint to initialise from, for fine-tuning. "
                             "Default: random initialisation.")

    group = parser.add_argument_group("signal population")
    group.add_argument("-s", "--snr", type=float, nargs=2, default=(5.0, 15.0),
                       metavar=("LOW", "HIGH"),
                       help="Range of network SNR for injected signals. "
                            "Default: 5 15.")

    group = parser.add_argument_group("model")
    group.add_argument("--d-model", type=int, default=128,
                       help="Embedding width. Default: 128.")
    group.add_argument("--n-heads", type=int, default=4,
                       help="Attention heads. Default: 4.")
    group.add_argument("--n-layers", type=int, default=4,
                       help="Transformer encoder layers. Default: 4.")
    group.add_argument("--patch-size", type=int, default=32,
                       help="Samples per token. Default: 32 (15.6 ms at "
                            "2048 Hz, giving 64 tokens for a 1 s input).")
    group.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout probability. Default: 0.1.")

    group = parser.add_argument_group("optimisation")
    group.add_argument("--epochs", type=int, default=50,
                       help="Number of passes over the data. Default: 50.")
    group.add_argument("--batch-size", type=int, default=128,
                       help="Mini-batch size. Default: 128.")
    group.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Adam learning rate. Default: 1e-4.")
    group.add_argument("--clip-norm", type=float, default=100.0,
                       help="Gradient-norm clipping. Default: 100.")
    group.add_argument("--weight-decay", type=float, default=1e-2,
                       help="AdamW weight decay. Default: 1e-2. Set 0 to "
                            "disable regularisation.")
    group.add_argument("--patience", type=int, default=20,
                       help="Stop if the validation loss has not improved for "
                            "this many epochs. Default: 20. Set 0 to disable "
                            "early stopping and always run --epochs.")
    group.add_argument("--lr-schedule", type=str, default="cosine",
                       choices=["none", "cosine", "plateau"],
                       help="Learning-rate schedule. Default: cosine.")
    group.add_argument("--fixed-pairing", action="store_true",
                       help="Pair waveform i with noise i, as originally "
                            "written. This binds every noise realisation to a "
                            "single class and invites the network to memorise "
                            "them; the default random pairing avoids that. Use "
                            "only when exact per-segment PSD matching matters.")
    group.add_argument("--validation-fraction", type=float, default=0.1,
                       help="Fraction of training data held out for validation "
                            "when the file has no 'validation' group. "
                            "Default: 0.1.")

    group = parser.add_argument_group("runtime")
    group.add_argument("--train-device", type=str, default="cpu",
                       help="Device for the network, e.g. 'cuda'. Default: cpu.")
    group.add_argument("--store-device", type=str, default="cpu",
                       help="Device holding the dataset. 'cuda' is fastest if "
                            "the data fits in GPU memory. Default: cpu.")
    group.add_argument("--num-workers", type=int, default=0,
                       help="DataLoader worker processes. Must be 0 when "
                            "--store-device is a GPU. Default: 0.")
    group.add_argument("--seed", type=int, default=2026,
                       help="Random seed. Default: 2026.")
    group.add_argument("--save-every-epoch", action="store_true",
                       help="Keep a checkpoint from every epoch, not just the "
                            "best and the last.")

    group = parser.add_argument_group("loss curve")
    group.add_argument("--no-plot", action="store_true",
                       help="Do not write the loss curve after training.")
    group.add_argument("--plot-accuracy", action="store_true",
                       help="Add an accuracy panel below the loss panel.")
    group.add_argument("--log-loss", action="store_true",
                       help="Use a logarithmic loss axis.")
    group.add_argument("--plot-only", action="store_true",
                       help="Skip training entirely: regenerate the loss curve "
                            "from the losses.txt already in the output "
                            "directory. Useful for restyling a figure without "
                            "repeating the run.")
    group.add_argument("--verbose", action="store_true", help="Print progress.")
    group.add_argument("--debug", action="store_true", help="Print debug messages.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing loss history in the output "
                             "directory.")

    args = parser.parse_args()

    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARN
    logging.basicConfig(format="%(levelname)s | %(asctime)s: %(message)s",
                        level=log_level, datefmt="%d-%m-%Y %H:%M:%S")

    # --plot-only regenerates the figure from an existing history and stops.
    if args.plot_only:
        losses_path = os.path.join(args.output_directory, "losses.txt")
        if not os.path.isfile(losses_path):
            print(f"\nError: no training history at '{losses_path}'.",
                  file=sys.stderr)
            sys.exit(1)
        try:
            written = plot_loss_curves(losses_path, args.output_directory,
                                       plot_accuracy=args.plot_accuracy,
                                       log_scale=args.log_loss)
        except Exception as exc:                       # noqa: BLE001
            print(f"\nError: could not write the loss curve: {exc}",
                  file=sys.stderr)
            sys.exit(1)
        print("Wrote " + ", ".join(written))
        return

    if not args.dataset_file:
        print("\nError: -d/--dataset-file is required unless --plot-only is "
              "given.", file=sys.stderr)
        sys.exit(1)

    try:
        files = resolve_dataset_files(args.dataset_file)
    except FileNotFoundError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_directory, exist_ok=True)
    losses_path = os.path.join(args.output_directory, "losses.txt")
    if os.path.isfile(losses_path) and not args.force:
        print(f"\nError: '{losses_path}' exists. Use --force to overwrite.",
              file=sys.stderr)
        sys.exit(1)

    if args.num_workers > 0 and "cuda" in args.store_device:
        logging.warning("--store-device is a GPU, so --num-workers must be 0. "
                        "Overriding.")
        args.num_workers = 0

    seed_everything(args.seed)
    logging.info("Found %i dataset file(s).", len(files))

    train_ds, valid_ds = build_datasets(
        files, tuple(args.snr), args.store_device, args.validation_fraction,
        args.seed, fixed_pairing=args.fixed_pairing)

    # Infer the input shape from the data rather than assuming it, so that a
    # non-default --sample-length in generate_dataset.py is picked up.
    example, _ = train_ds[0]
    n_detectors, sample_length = example.shape
    logging.info("Input shape: %i detectors x %i samples.",
                 n_detectors, sample_length)

    device = torch.device(args.train_device)
    model_config = dict(
        n_detectors=n_detectors, sample_length=sample_length,
        patch_size=args.patch_size, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout)
    if args.weights:
        model = load_checkpoint(args.weights, device,
                                fallback_config=model_config)
        logging.info("Initialised from %s", args.weights)
    else:
        model = BaselineTransformer(**model_config).to(device)
    logging.info("Model has %s trainable parameters.",
                 f"{count_parameters(model):,}")

    with open(os.path.join(args.output_directory, "config.json"), "w") as handle:
        json.dump({"args": vars(args), "files": files,
                   "model": model.config,
                   "n_parameters": count_parameters(model),
                   "n_train": len(train_ds), "n_valid": len(valid_ds)},
                  handle, indent=2, default=str)

    train(model, train_ds, valid_ds, args.output_directory, device,
          batch_size=args.batch_size, learning_rate=args.learning_rate,
          epochs=args.epochs, clip_norm=args.clip_norm,
          num_workers=args.num_workers, save_every_epoch=args.save_every_epoch,
          verbose=args.verbose, make_plot=not args.no_plot,
          plot_accuracy=args.plot_accuracy, log_loss=args.log_loss,
          weight_decay=args.weight_decay, patience=args.patience,
          lr_schedule=args.lr_schedule)


if __name__ == "__main__":
    main()
