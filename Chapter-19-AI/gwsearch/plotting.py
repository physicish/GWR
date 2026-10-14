"""Training-history figures.

One function, used at the end of every training run in this repository, so
that loss curves from the different stages are directly comparable.
"""

import os

import numpy as np

# Wong palette: distinguishable in greyscale and to colour-vision deficient
# readers.
TRAIN_COLOUR = "#0072B2"
VALID_COLOUR = "#D55E00"


def read_history(losses_path):
    """Read a ``losses.txt`` and return (epochs, train_loss, valid_loss).

    Two column layouts are written by the training scripts, and the
    validation series returned here matches the quantity each script
    minimises when it decides which checkpoint to keep:

    5 columns -- ``epoch train_loss train_acc val_loss val_acc``
        Written by the single-stream trainer. The validation series is the
        fourth column.

    7 columns -- ``epoch train_loss train_acc online_val_loss
        online_val_acc pregen_val_loss pregen_val_acc``
        Written by the mixed and coincident trainers, which validate on
        on-the-fly and pre-generated examples separately. The validation
        series is the mean of the two losses, which is the criterion those
        scripts use to select ``best_ema.pt``.

    Keeping these in step matters: the figure marks the best epoch with a
    star, and that star should point at the checkpoint that was actually
    saved.
    """
    history = np.loadtxt(losses_path)
    if history.size == 0:
        raise ValueError(f"'{losses_path}' contains no completed epochs.")
    if history.ndim == 1:                      # a single epoch
        history = history[np.newaxis, :]

    epochs = history[:, 0]
    train_loss = history[:, 1]
    n_columns = history.shape[1]

    if n_columns == 5:
        valid_loss = history[:, 3]
    elif n_columns == 7:
        valid_loss = 0.5 * (history[:, 3] + history[:, 5])
    else:
        raise ValueError(
            f"'{losses_path}' has {n_columns} columns; expected 5 (one "
            "validation stream) or 7 (two validation streams).")

    return epochs, train_loss, valid_loss


def plot_loss_curves(losses_path, output_directory, log_scale=False,
                     dpi=300, figsize=(6.5, 4.0), basename="loss_curve"):
    """Write a publication-quality training-history figure.

    Saves the result twice: a vector PDF for inclusion in a manuscript, and
    a PNG for quick inspection. The epoch with the lowest validation loss is
    marked, and is the epoch saved as the best checkpoint.

    Parameters
    ----------
    losses_path : str
        The ``losses.txt`` written during training.
    output_directory : str
        Where the figure is written.
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

    epochs, train_loss, valid_loss = read_history(losses_path)

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
        fig, ax = plt.subplots(figsize=figsize)

        # Markers help on a short run and become clutter on a long one.
        marker = "o" if len(epochs) <= 40 else None

        ax.plot(epochs, train_loss, color=TRAIN_COLOUR, lw=1.6, marker=marker,
                ms=3.5, label="Training")
        ax.plot(epochs, valid_loss, color=VALID_COLOUR, lw=1.6, marker=marker,
                ms=3.5, label="Validation")

        # Mark the epoch that was saved as the best checkpoint.
        best = int(np.argmin(valid_loss))
        ax.axvline(epochs[best], color="0.65", ls=":", lw=1.0, zorder=0)
        ax.plot(epochs[best], valid_loss[best], ls="none", marker="*", ms=12,
                color=VALID_COLOUR, mec="black", mew=0.5, zorder=5,
                label=f"Best: epoch {int(epochs[best])} "
                      f"(loss {valid_loss[best]:.4f})")

        ax.set_ylabel("Cross-entropy loss")
        ax.set_xlabel("Epoch")
        if log_scale:
            ax.set_yscale("log")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(loc="best")
        ax.set_xlim(epochs.min(), max(epochs.max(), epochs.min() + 1))

        fig.tight_layout()

        written = []
        for extension in ("pdf", "png"):
            path = os.path.join(output_directory, f"{basename}.{extension}")
            fig.savefig(path, dpi=dpi)
            written.append(path)
        plt.close(fig)

    return written
