"""Compare zero-lag triggers against the time-slide background for Figure 1.3.

Plots the cumulative number of zero-lag events against inverse false-alarm
rate, together with the expectation from the time slides and its Poisson
intervals.

The expected count at ranking threshold R is ``N = T_0 * FAR(R)``, with
``T_0`` the analysed zero-lag live time and the false-alarm rate estimated by
counting background events above R and dividing by the accumulated slide live
time. Agreement within the Poisson bands indicates that the time-slide
procedure reproduces the coincident-noise tail, and therefore that the
false-alarm rates assigned to candidates can be believed.

The chapter uses two disjoint five-day selections of coincident O3a data: GPS
1239392735-1240303634 for the zero-lag test, and GPS 1238166018-1238997902
for the background.
"""

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
from scipy.stats import poisson
from scipy.special import ndtr


DAY = 86400.0
YEAR = 365.25 * DAY


def read_background(path):
    with h5py.File(path, "r") as file:
        if "bg_stat" not in file:
            raise ValueError(f"{path} has no bg_stat; run the evaluator with --num-timeslides")
        values = np.asarray(file["bg_stat"][()], dtype=np.float64)
        duration = file.attrs.get("bg_livetime_seconds")
        return values[np.isfinite(values)], duration


def read_test(path):
    with h5py.File(path, "r") as file:
        if "stat" not in file or "time" not in file:
            raise ValueError(f"{path} must contain stat and time")
        stats = np.asarray(file["stat"][()], dtype=np.float64)
        times = np.asarray(file["time"][()], dtype=np.float64)
        if len(stats) != len(times):
            raise ValueError(f"Mismatched stat and time lengths in {path}")
        valid = np.isfinite(stats) & np.isfinite(times)
        live = file.attrs.get("analysis_livetime_seconds")
        return stats[valid], times[valid], live


def duration_from_input(path, step_size, window_seconds=1.0):
    """Exactly count zero-lag window exposure using the evaluator's stride."""
    live = 0.0
    with h5py.File(path, "r") as file:
        if "H1" not in file:
            raise ValueError(f"{path} has no H1 segment group")
        for dataset in file["H1"].values():
            dt = float(dataset.attrs["delta_t"])
            rate = int(round(1.0 / dt))
            window_samples = int(round(window_seconds * rate))
            step_samples = int(round(step_size / dt))
            if step_samples < 1:
                raise ValueError("Step size smaller than one input sample")
            count = max(0, 1 + (len(dataset) - window_samples) // step_samples)
            live += count * step_samples * dt
    return live


def assigned_fars(test_stats, bg_stats, bg_live, n_trials, add_one):
    """FAR(s) = n_trials * [N_bg(bg_stat >= s) + add_one] / T_bg."""
    bg_sorted = np.sort(bg_stats)
    n_louder = len(bg_sorted) - np.searchsorted(bg_sorted, test_stats, side="left")
    return n_trials * (n_louder + int(add_one)) / bg_live


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--background-file", required=True,
                   help="Coherent evaluator output containing bg_stat and bg_livetime_seconds")
    p.add_argument("--test-file", required=True,
                   help="Independent noise-only zero-lag evaluator output")
    p.add_argument("--test-duration-seconds", type=float)
    p.add_argument("--test-duration-days", type=float)
    p.add_argument("--test-inputfile", help="Original zero-lag input HDF for exact analyzed exposure")
    p.add_argument("--step-size", type=float, default=.1,
                   help="Evaluator --step-size, for --test-inputfile (default: .1 s)")
    p.add_argument("--background-duration-seconds", type=float,
                   help="Override background-file bg_livetime_seconds")
    p.add_argument("--n-trials", type=int, default=1)
    p.add_argument("--no-add-one", action="store_true")
    p.add_argument("--analysis-threshold", type=float,
                   help="Evaluator --score-threshold; warns if plot extends beyond its FAR")
    p.add_argument("--ifar-min", type=float, default=1e-5,
                   help="Shortest inverse FAR plotted, in years")
    p.add_argument("--ifar-max", type=float, default=.1,
                   help="Longest inverse FAR plotted, in years")
    p.add_argument("--n-grid", type=int, default=300)
    p.add_argument("--output", default="coherent_far_validation.png")
    args = p.parse_args()

    if args.test_duration_seconds is not None and args.test_duration_days is not None:
        p.error("Provide only one explicit test duration")
    if args.n_trials < 1 or args.n_grid < 2 or args.step_size <= 0:
        p.error("n-trials, n-grid and step-size must be positive")
    if not (0 < args.ifar_min < args.ifar_max):
        p.error("Require 0 < --ifar-min < --ifar-max")

    bg_stats, bg_attr = read_background(args.background_file)
    test_stats, _, test_attr = read_test(args.test_file)
    if not len(bg_stats) or not len(test_stats):
        raise ValueError("Both background and zero-lag event sets must be nonempty")

    if args.background_duration_seconds is not None:
        bg_live = args.background_duration_seconds
        bg_source = "--background-duration-seconds"
    elif bg_attr is not None:
        bg_live = float(bg_attr)
        bg_source = "background-file bg_livetime_seconds"
    else:
        raise ValueError("Background file lacks bg_livetime_seconds; supply "
                         "--background-duration-seconds")

    if args.test_duration_seconds is not None:
        test_live = args.test_duration_seconds
        test_source = "--test-duration-seconds"
    elif args.test_duration_days is not None:
        test_live = args.test_duration_days * DAY
        test_source = "--test-duration-days"
    elif test_attr is not None:
        test_live = float(test_attr)
        test_source = "test-file analysis_livetime_seconds"
    elif args.test_inputfile:
        test_live = duration_from_input(args.test_inputfile, args.step_size)
        test_source = "--test-inputfile window exposure"
    else:
        raise ValueError("The event output does not record zero-lag livetime. "
                         "Supply --test-duration-days, --test-duration-seconds, "
                         "or --test-inputfile; trigger time span is not reliable.")
    if not np.isfinite(test_live) or test_live <= 0 or not np.isfinite(bg_live) or bg_live <= 0:
        raise ValueError(f"Invalid livetimes: T_test={test_live}, T_bg={bg_live}")

    fars = assigned_fars(test_stats, bg_stats, bg_live, args.n_trials,
                         add_one=not args.no_add_one)
    fars = np.sort(fars[np.isfinite(fars) & (fars > 0)])
    if not len(fars):
        raise ValueError("No positive empirical zero-lag FARs; remove --no-add-one")
    resolution_years = bg_live / (args.n_trials * YEAR)
    if args.ifar_max > resolution_years:
        print(f"WARNING: --ifar-max={args.ifar_max:g} yr exceeds empirical "
              f"resolution {resolution_years:.4g} yr")

    threshold_far = None
    if args.analysis_threshold is not None:
        threshold_far = args.n_trials * np.count_nonzero(
            bg_stats >= args.analysis_threshold) / bg_live
        if threshold_far > 0 and 1.0 / (args.ifar_min * YEAR) > threshold_far:
            print("WARNING: the plotted high-FAR end may be incomplete above "
                  "the evaluator's score-threshold FAR")

    ifar_years = np.logspace(np.log10(args.ifar_min), np.log10(args.ifar_max),
                            args.n_grid)
    far_grid = 1.0 / (ifar_years * YEAR)
    observed = np.searchsorted(fars, far_grid, side="right")
    expected = far_grid * test_live

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                         "font.size": 12, "axes.labelsize": 14,
                         "axes.linewidth": 1.1, "savefig.bbox": "tight"})
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for sigma, color in ((3, "#E0E0E0"), (2, "#BDBDBD"), (1, "#8C8C8C")):
        coverage = 2.0 * ndtr(sigma) - 1.0
        lo, hi = poisson.interval(coverage, expected)
        ax.fill_between(ifar_years, np.maximum(lo, 1e-3),
                        np.maximum(hi, 1e-3), color=color, linewidth=0,
                        label=fr"{sigma}$\sigma$ Poisson band")
    ax.plot(ifar_years, np.maximum(expected, 1e-3), "--", color="#202020",
            linewidth=1.8, label="Expected background")
    ax.step(ifar_years, np.maximum(observed, 1e-3), where="post",
            color="#00549F", linewidth=2.1, label="Zero-lag triggers")
    ax.set(xscale="log", yscale="log", xlim=(args.ifar_min, args.ifar_max),
           ylim=(.8, 2.0 * max(10., expected.max(), observed.max())),
           xlabel="Inverse false-alarm rate, iFAR (yr)",
           ylabel="Cumulative count")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10))
        axis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * .1))
        axis.set_minor_formatter(NullFormatter())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.grid(True, which="major", color="#D8D8D8", alpha=.65)
    ax.legend(loc="best", frameon=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=400)
    plt.close(fig)

    print(f"Zero-lag: {len(test_stats)} triggers, {test_live / DAY:.6f} days "
          f"({test_source})")
    print(f"Slides: {len(bg_stats)} triggers, {bg_live / DAY:.6f} days "
          f"({bg_source})")
    print(f"Maximum empirical iFAR resolution: {resolution_years:.6g} yr")
    if threshold_far is not None:
        print(f"FAR above analysis threshold: {threshold_far * YEAR:.6g} yr^-1")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
