"""Shared components for the Chapter 19 gravitational-wave search experiments.

Four modules, used by the stage scripts in the numbered directories:

``coherent_model``
    The transformer that reads both detectors together as two input
    channels (664,522 parameters). Sections 1.2 to 1.4.2 of the chapter.
``coincident_model``
    The same network applied to one detector at a time, with the two
    per-detector scores combined afterwards (664,042 parameters).
    Sections 1.4.3 and 1.5.
``evaluate``
    The MLGWSC-1 scoring routine: turns a file of candidate events into
    sensitive distance as a function of false-alarm rate.
``plotting``
    Training-history figures, shared by every training script.

The model classes are imported lazily, so ``gwsearch.plotting`` can be used
to redraw a saved ``losses.txt`` without PyTorch installed.
"""

__all__ = [
    "CoherentTransformer",
    "CoincidentTransformer",
    "usr",
    "per_detector_usr",
    "combined_usr",
    "combined_usr_and",
    "plot_loss_curves",
    "read_history",
]

_LAZY = {
    "CoherentTransformer": "gwsearch.coherent_model",
    "usr": "gwsearch.coherent_model",
    "CoincidentTransformer": "gwsearch.coincident_model",
    "per_detector_usr": "gwsearch.coincident_model",
    "combined_usr": "gwsearch.coincident_model",
    "combined_usr_and": "gwsearch.coincident_model",
    "plot_loss_curves": "gwsearch.plotting",
    "read_history": "gwsearch.plotting",
}


def __getattr__(name):
    """Import on first use (PEP 562) so plotting does not require PyTorch."""
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
