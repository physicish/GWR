#!/usr/bin/env python
"""
Generate a training/validation dataset of binary-black-hole signals in
gravitational-wave detector noise.

This script produces the data used to train the *baseline* transformer of the
chapter. It writes a single self-describing HDF5 file containing whitened
two-detector (H1, L1) noise samples and, separately, whitened signal templates
normalised to unit network signal-to-noise ratio (SNR).

------------------------------------------------------------------------------
WHY SIGNALS AND NOISE ARE STORED SEPARATELY
------------------------------------------------------------------------------
This is the single most important design choice in the file, and it is not
obvious. We do *not* store "signal + noise" examples. Instead we store:

    noises[i]     : whitened detector noise,          shape (2, sample_length)
    waveforms[i]  : whitened signal at network SNR=1, shape (2, sample_length)

A training example at a chosen SNR rho is then assembled on the fly as

    x = noises[i] + rho * waveforms[i]

Because the whitened noise has (approximately) unit variance per sample, and
each stored waveform has been divided by its own optimal network SNR, the
matched-filter SNR of the assembled example is exactly rho. This makes the
injected SNR a *free parameter at training time*: one generated dataset can be
reused for any SNR distribution, and the SNR can be annealed during training
without regenerating data. It also keeps the file small.

------------------------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------------------------
Requires: pycbc, numpy, h5py, tqdm  (pip install pycbc h5py tqdm)

    # Gaussian noise coloured by the analytic aLIGO design PSD (chapter step 1)
    python generate_dataset.py -o data_gaussian.h5 --verbose

    # Smaller, faster run for a first look
    python generate_dataset.py -o data_small.h5 \
        --training-samples 2000 2000 --validation-samples 500 500 --verbose

    # Real O3 detector noise instead of Gaussian noise (chapter step 6)
    python generate_dataset.py -o data_o3.h5 \
        --real-noise-file real_noise_o3.hdf --verbose

    # Waveforms with precession and higher-order modes (chapter step 8)
    python generate_dataset.py -o data_xphm.h5 -a IMRPhenomXPHM --verbose

Then visualise the result with:

    python plot_dataset.py data_gaussian.h5 -o dataset_overview.png

------------------------------------------------------------------------------
Adapted for this chapter from the MLGWSC-1 mock data challenge generation code
by Ondrej Zelenka (https://github.com/gwastro/ml-mock-data-challenge-1),
Apache License 2.0. Modifications copyright 2026 Chayan Chatterjee.
------------------------------------------------------------------------------
"""

from argparse import ArgumentParser
from itertools import cycle
import hashlib
import json
import logging
import os
import os.path
import random

import h5py
import numpy as np
from tqdm import tqdm

import pycbc.detector
import pycbc.distributions
import pycbc.filter          # needed for sigmasq(); importing pycbc alone is not enough
import pycbc.noise
import pycbc.psd
import pycbc.types
import pycbc.waveform


# =============================================================================
# Fixed configuration
# =============================================================================
# Sampling interval. 2048 Hz is the standard rate for binary-black-hole
# searches: it resolves frequencies up to 1024 Hz, comfortably above the
# few-hundred-Hz merger frequency of a 10-50 solar-mass binary.
DELTA_T = 1.0 / 2048.0
SAMPLE_RATE = int(round(1.0 / DELTA_T))

# Whitening filter length, in samples. The whitening filter is truncated in the
# time domain to this length so that it cannot smear signal power across the
# whole segment. We generate each segment LONGER than we need by exactly this
# amount, then discard MAX_FILTER_LENGTH/2 samples from each end after
# whitening, where the filter has wrapped around and corrupted the data.
MAX_FILTER_LENGTH = 512
MAX_FILTER_DURATION = MAX_FILTER_LENGTH * DELTA_T          # 0.25 s
MAX_FILTER_HALFDURATION = MAX_FILTER_DURATION * 0.5        # 0.125 s

# Seismic and suspension noise below ~20 Hz is overwhelming, so all analyses
# (and the SNR integral) start here.
LOW_FREQUENCY_CUTOFF = 20.0

# Component-mass range of the training population, in solar masses.
#
# NOTE (deliberate simplification, revisited later in the chapter): masses are
# drawn uniformly in (m1, m2). Uniform sampling in component mass does NOT give
# uniform coverage in signal *duration* -- short, high-mass signals dominate.
# This is the "limited sample representation" bias; the standard fix is to
# sample uniformly in the chirp-time coordinates (tau0, tau3) instead.
MASS_MIN = 10.0
MASS_MAX = 50.0

# Sky localisation / merger-time placement window, in seconds from the start of
# the (whitened) analysis segment. Randomising the merger position is important:
# if every training signal peaked at the same sample, the network would learn
# that position rather than the signal morphology.
MERGER_WINDOW_START = 0.4   # merger placed at (sample_duration - 0.4) ...
MERGER_WINDOW_END = 0.2     # ... to (sample_duration - 0.2) seconds


# =============================================================================
# Small reproducibility helpers (kept inline so this script is self-contained)
# =============================================================================
def canonical_json(value):
    """Serialise a config dict to a stable string, for storing in HDF5 attrs."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_file(path, chunk_size=8 << 20):
    """Content hash of a file, so a dataset can be traced back to its inputs."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed):
    """Seed Python and NumPy so a dataset can be regenerated bit-for-bit."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# =============================================================================
# Output container
# =============================================================================
class GeneratedDataset:
    """Holds one split (training or validation) and writes it to HDF5.

    Attributes
    ----------
    noises : ndarray, shape (n_injections + n_noise, n_detectors, sample_length)
        Whitened noise segments. One segment per dataset entry, including the
        entries that will have a signal added to them.
    waveforms : ndarray, shape (n_injections, n_detectors, sample_length)
        Whitened signals, each normalised to network SNR = 1. Entry ``i`` is
        meant to be added to ``noises[i]`` scaled by the desired SNR.
    """

    def __init__(self, noises, waveforms):
        self.noises = np.asarray(noises, dtype=np.float32)
        self.waveforms = np.asarray(waveforms, dtype=np.float32)

    def save(self, h5py_file, group_name):
        if group_name in h5py_file:
            raise IOError(f"Group '{group_name}' already exists")
        group = h5py_file.create_group(group_name)
        group.create_dataset("waveforms", data=self.waveforms, compression="gzip")
        group.create_dataset("noises", data=self.noises, compression="gzip")
        group.attrs["detector_count"] = self.noises.shape[1]
        group.attrs["sample_count"] = self.noises.shape[0]
        group.attrs["waveform_count"] = self.waveforms.shape[0]
        group.attrs["sample_length"] = self.noises.shape[-1]


# =============================================================================
# Whitening
# =============================================================================
def regularize_psd(psd, low_frequency_cutoff, inf_value=np.inf):
    """Set the PSD to infinity below the analysis cutoff.

    Dividing by an infinite PSD sends those frequency bins to zero, which
    cleanly removes the enormous low-frequency seismic noise instead of letting
    it dominate the whitened time series.
    """
    values = np.asarray(psd, dtype=np.float64).copy()
    values[np.asarray(psd.sample_frequencies) <= low_frequency_cutoff] = inf_value
    return pycbc.types.FrequencySeries(values, delta_f=psd.delta_f, copy=False)


def whiten(strain, psd, low_frequency_cutoff, delta_t=DELTA_T,
           max_filter_duration=MAX_FILTER_DURATION):
    """Whiten a strain segment by dividing by the amplitude spectral density.

    Detector noise is strongly coloured: it is orders of magnitude louder at
    20 Hz than at 200 Hz. A network trained on raw strain would spend its
    capacity modelling that (uninformative) shape. Whitening divides each
    frequency bin by sqrt(PSD), producing a time series whose noise is
    approximately white with unit variance, so that every frequency contributes
    to the loss on an equal footing.

    This is also what makes the "add a rescaled waveform" trick of this script
    valid: after whitening, the Euclidean norm of the signal *is* its
    matched-filter SNR.

    Parameters
    ----------
    strain : ndarray
        Either a single segment of shape (n_samples,) or a multi-detector
        block of shape (n_detectors, n_samples). In the latter case ``psd``
        must be a sequence of per-detector PSDs.
    psd : pycbc.types.FrequencySeries or sequence of them
        Noise power spectral density used for whitening.
    low_frequency_cutoff : float
        Frequencies below this are suppressed.

    Returns
    -------
    ndarray
        Whitened strain, SHORTER than the input by ``max_filter_duration``:
        ``max_filter_duration/2`` is removed from each end, where the truncated
        whitening filter has wrapped around and corrupted the data.
    """
    if strain.ndim == 2:
        return np.stack([
            whiten(row, detector_psd, low_frequency_cutoff,
                   delta_t=delta_t, max_filter_duration=max_filter_duration)
            for row, detector_psd in zip(strain, psd)
        ])

    colored = pycbc.types.TimeSeries(strain, delta_t=delta_t)
    interpolated = pycbc.psd.interpolate(psd, colored.delta_f)
    filter_length = int(max_filter_duration * colored.sample_rate)
    truncated = pycbc.psd.inverse_spectrum_truncation(
        interpolated,
        max_filter_len=filter_length,
        low_frequency_cutoff=low_frequency_cutoff,
        trunc_method="hann",
    )
    white = (colored.to_frequencyseries() * (1.0 / truncated) ** 0.5).to_timeseries()
    return white.numpy()[filter_length // 2: len(colored) - filter_length // 2]


# =============================================================================
# Noise sources
# =============================================================================
class GaussianNoiseGenerator:
    """Coloured Gaussian noise, whitened on the fly.

    Each call draws a new noise realisation from a power spectral density and
    whitens it. By default the analytic aLIGO design-sensitivity curve
    (``aLIGOZeroDetHighPower``) is used, which gives stationary, Gaussian,
    glitch-free noise.

    NOTE (deliberate simplification, revisited later in the chapter): this is
    the easiest noise a search will ever see. Real detector data is neither
    stationary nor Gaussian and contains loud non-Gaussian transients
    ("glitches"). A model trained here and tested on real noise will look far
    worse than its validation score suggests -- the "train-test mismatch" and
    "limited feature representation" biases. Use ``RealNoiseGenerator`` to fix
    this.
    """

    def __init__(self, detectors, low_frequency_cutoff, sample_length,
                 fpath=None, max_filter_duration=MAX_FILTER_DURATION):
        if fpath is None:
            psd_fun = pycbc.psd.analytical.aLIGOZeroDetHighPower
            delta_f = 1.0 / (DELTA_T * sample_length)
            self.all_psds = [[
                psd_fun(sample_length // 2 + 1, delta_f, low_frequency_cutoff)
                for _ in detectors
            ]]
        else:
            # A file of measured PSDs: one HDF5 group per detector, each
            # containing one dataset per PSD, with a `delta_f` attribute.
            with h5py.File(fpath, "r") as psd_file:
                self.all_psds = [
                    [pycbc.types.FrequencySeries(psd_ds[()],
                                                 delta_f=psd_ds.attrs["delta_f"],
                                                 dtype=np.float64)
                     for psd_ds in det_grp.values()]
                    for det_grp in psd_file.values()
                ]
                self.all_psds = list(zip(*self.all_psds))

        self.all_whitening_psds = [
            [regularize_psd(psd, low_frequency_cutoff) for psd in psds]
            for psds in self.all_psds
        ]
        self.noise_fun = pycbc.noise.gaussian.frequency_noise_from_psd
        self.low_frequency_cutoff = low_frequency_cutoff
        self.max_filter_duration = max_filter_duration

    def __iter__(self):
        self.psd_iter = iter(zip(cycle(self.all_psds), cycle(self.all_whitening_psds)))
        return self

    def __next__(self):
        psds, whitening_psds = next(self.psd_iter)
        noise = np.stack(
            [self.noise_fun(psd).to_timeseries().numpy() for psd in psds], axis=0
        )
        noise = whiten(noise, whitening_psds, self.low_frequency_cutoff,
                       max_filter_duration=self.max_filter_duration)
        return noise, psds, whitening_psds


class RealNoiseGenerator:
    """Pre-generated segments of real detector noise, read from HDF5.

    IMPORTANT: unlike ``GaussianNoiseGenerator``, the segments in this file are
    expected to be ALREADY WHITENED and already trimmed to ``sample_length``.
    They are returned unchanged. The accompanying PSDs are still needed, because
    the *signal* injected into each segment must be whitened with the PSD that
    matches the noise it is being added to.

    Expected file layout (the MLGWSC-1 real-noise format):
        noises          (n_segments, n_detectors, sample_length)
        psds            (n_psds, n_detectors, n_freqs)  + attr `delta_f`
        whitening_psds  (n_psds, n_detectors, n_freqs)  + attr `delta_f`
        psd_indices     (n_segments,)   which PSD belongs to which segment
        attrs['detectors']  e.g. ['H1', 'L1']
    """

    def __init__(self, fpath, detectors):
        self.detectors = [det.name for det in detectors]
        self.noises = []
        self.psds = []
        self.whitening_psds = []
        with h5py.File(fpath, "r") as inf:
            self.psd_indices = inf["psd_indices"][()]
            noises = inf["noises"][()]
            psds = inf["psds"][()]
            whitening_psds = inf["whitening_psds"][()]
            inf_det_names = np.array(inf.attrs["detectors"])
            for det in self.detectors:
                index = np.where(inf_det_names == det)[0][0]
                self.noises.append(noises[:, index])
                self.psds.append([
                    pycbc.types.FrequencySeries(data[index],
                                                delta_f=inf["psds"].attrs["delta_f"])
                    for data in psds
                ])
                self.whitening_psds.append([
                    pycbc.types.FrequencySeries(
                        data[index], delta_f=inf["whitening_psds"].attrs["delta_f"])
                    for data in whitening_psds
                ])
        self.noises = np.stack(self.noises, axis=1)
        self.psds = list(zip(*self.psds))
        self.whitening_psds = list(zip(*self.whitening_psds))

    def __iter__(self):
        self.noise_iter = iter(zip(self.noises, self.psd_indices))
        return self

    def __next__(self):
        new_noise, new_index = next(self.noise_iter)
        return new_noise, self.psds[new_index], self.whitening_psds[new_index]


# =============================================================================
# Source parameters
# =============================================================================
def generate_spin(low, high, rng):
    """Draw an isotropically oriented spin vector with magnitude in [low, high]."""
    unnormalized_spin = rng.standard_normal(size=3)
    old_spin_norm = np.sqrt(np.sum(unnormalized_spin ** 2))
    new_spin_norm = rng.uniform(low, high)
    return unnormalized_spin * new_spin_norm / old_spin_norm


def add_spins(parameter_dictionary, body_index, spin_vector):
    """Write a spin vector into a PyCBC waveform-parameter dictionary."""
    vec_name = "spin%i" % body_index
    for direction, num in zip(("x", "y", "z"), spin_vector):
        parameter_dictionary[vec_name + direction] = num
    return parameter_dictionary


# =============================================================================
# Dataset generation
# =============================================================================
def generate_dataset(samples, detectors, noise_getter_iter, rng,
                     low_frequency_cutoff=LOW_FREQUENCY_CUTOFF,
                     approximant="IMRPhenomD", sample_length=2048,
                     verbose=False):
    """Build one split of the dataset.

    For every entry we draw a fresh noise segment. For the first
    ``n_injections`` entries we additionally simulate a binary-black-hole
    merger, project it onto the detectors, normalise it to network SNR = 1 and
    whiten it. Signals are returned separately from the noise (see the module
    docstring for why).

    Parameters
    ----------
    samples : (int, int)
        ``(n_injections, n_pure_noise)``.
    detectors : sequence of pycbc.detector.Detector
    noise_getter_iter : iterator
        Yields ``(whitened_noise, psds, whitening_psds)``.
    rng : numpy.random.Generator
        Source of all randomness, passed in explicitly for reproducibility.
    approximant : {'IMRPhenomD', 'IMRPhenomXPHM'}
        Waveform model. ``IMRPhenomD`` is aligned-spin and dominant-mode only;
        ``IMRPhenomXPHM`` adds precession and higher-order modes.

        NOTE (deliberate simplification): the baseline uses IMRPhenomD only.
        Training on a single waveform family that is *simpler* than the signals
        in the test set is a train-test mismatch; the chapter later shows the
        effect of training on a more realistic waveform distribution.
    sample_length : int
        Length of each whitened output segment, in samples.

    Returns
    -------
    GeneratedDataset
    """
    if approximant == "IMRPhenomD":
        spins_required = False
    elif approximant == "IMRPhenomXPHM":
        spins_required = True
    else:
        raise ValueError("Approximant %s not allowed." % approximant)

    skylocation_dist = pycbc.distributions.sky_location.UniformSky()

    num_waveforms, num_noises = samples
    logging.info("Generating dataset with %i injections and %i pure noise samples",
                 num_waveforms, num_noises)

    sample_duration = sample_length * DELTA_T
    # Generate a longer segment so that the whitening edges can be discarded.
    colored_sample_length = sample_length + MAX_FILTER_LENGTH
    colored_sample_duration = colored_sample_length * DELTA_T

    noises = []
    waveforms = []

    for i in tqdm(range(num_waveforms + num_noises), disable=(not verbose), ascii=True):
        noise, psds, whitening_psds = next(noise_getter_iter)

        if noise.shape[-1] != sample_length:
            raise ValueError(
                f"Noise segment has length {noise.shape[-1]} but sample_length is "
                f"{sample_length}. If you are using --real-noise-file, the stored "
                f"segments must already be whitened and trimmed to sample_length."
            )
        noises.append(noise)

        if i >= num_waveforms:
            continue  # this entry stays pure noise

        # ---- Draw the source parameters -------------------------------------
        waveform_kwargs = {
            "delta_t": DELTA_T,
            "f_lower": low_frequency_cutoff,
            "approximant": approximant,
        }
        masses = rng.uniform(MASS_MIN, MASS_MAX, 2)
        waveform_kwargs["mass1"] = max(masses)
        waveform_kwargs["mass2"] = min(masses)

        angles = rng.uniform(0.0, 2 * np.pi, 3)
        waveform_kwargs["coa_phase"] = angles[0]     # orbital phase at merger
        waveform_kwargs["inclination"] = angles[1]   # orbital-plane orientation
        pol_angle = angles[2]                        # polarisation angle
        declination, right_ascension = skylocation_dist.rvs()[0]

        if spins_required:
            for body_index in (1, 2):
                add_spins(waveform_kwargs, body_index, generate_spin(0.0, 0.99, rng))

        # An arbitrary GPS time in the O3a observing run. This only sets the
        # Earth's orientation, and hence the antenna response of each detector.
        injection_time = rng.uniform(1238166018, 1253977218)

        # ---- Simulate and project onto the detectors ------------------------
        h_plus, h_cross = pycbc.waveform.get_td_waveform(**waveform_kwargs)

        start_time = injection_time + h_plus.get_sample_times()[0]
        h_plus.start_time = start_time
        h_cross.start_time = start_time
        # Pad generously so the time slice below never runs off the end.
        for polarisation in (h_plus, h_cross):
            polarisation.append_zeros(colored_sample_length)
            polarisation.prepend_zeros(colored_sample_length)

        # project_wave applies each detector's antenna pattern and the light
        # travel-time delay between sites, so H1 and L1 see genuinely different
        # amplitudes and slightly offset arrival times. This is the physical
        # correlation that a coherent, two-detector model can exploit.
        strains = [
            det.project_wave(h_plus, h_cross, right_ascension, declination, pol_angle)
            for det in detectors
        ]

        # ---- Place the merger at a random position in the segment -----------
        # After whitening removes MAX_FILTER_HALFDURATION from the front, the
        # merger lands uniformly in [sample_duration - 0.4, sample_duration - 0.2]
        # seconds from the start of the output window (0.6-0.8 s for a 1 s window).
        time_placement = rng.uniform(sample_duration - MERGER_WINDOW_START,
                                     sample_duration - MERGER_WINDOW_END)
        time_placement += MAX_FILTER_HALFDURATION
        time_interval = injection_time - time_placement
        # The -1e-3 guards against the slice returning one sample too many.
        time_interval = (time_interval,
                         time_interval + colored_sample_duration - 1.0e-3)
        strains = [strain.time_slice(*time_interval) for strain in strains]
        for strain in strains:
            to_append = colored_sample_length - len(strain)
            if to_append > 0:
                strain.append_zeros(to_append)

        # ---- Normalise to network SNR = 1 and whiten ------------------------
        # sigmasq(h) = <h|h> is the squared optimal matched-filter SNR of the
        # template against its own noise. Summing in quadrature over detectors
        # gives the network SNR; dividing by it leaves a unit-SNR template.
        network_snr = np.sqrt(sum(
            pycbc.filter.matchedfilter.sigmasq(
                strain, psd=psd, low_frequency_cutoff=low_frequency_cutoff)
            for strain, psd in zip(strains, psds)
        ))
        waveform = np.stack([strain.numpy() for strain in strains], axis=0) / network_snr
        waveform = whiten(waveform, whitening_psds, low_frequency_cutoff)
        waveforms.append(waveform)

    noises = np.stack(noises, axis=0)
    waveforms = np.stack(waveforms, axis=0)
    return GeneratedDataset(noises=noises, waveforms=waveforms)


# =============================================================================
# Entry point
# =============================================================================
def main():
    parser = ArgumentParser(
        description="Generate a two-detector BBH training dataset for the "
                    "baseline transformer of the GW machine-learning chapter.")

    parser.add_argument("-o", "--output-file", type=str, required=True,
                        help="Path to the HDF5 file where datasets will be stored.")
    parser.add_argument("-a", "--approximant", type=str, default="IMRPhenomD",
                        choices=["IMRPhenomD", "IMRPhenomXPHM"],
                        help="Waveform model. IMRPhenomD: aligned-spin, dominant "
                             "mode. IMRPhenomXPHM: precession + higher-order modes. "
                             "Default: IMRPhenomD.")
    parser.add_argument("-d", "--detectors", type=int, default=2, choices=[2],
                        help="Number of detectors. The chapter uses the H1/L1 pair.")
    parser.add_argument("-s", "--sample-length", type=int, default=2048,
                        help="Length of each sample in samples, at 2048 Hz. "
                             "Default: 2048 (1 s).")
    parser.add_argument("--training-samples", type=int, nargs=2,
                        default=[10000, 10000], metavar=("N_INJ", "N_NOISE"),
                        help="Number of training injections and pure-noise "
                             "samples. Default: 10000 10000.")
    parser.add_argument("--validation-samples", type=int, nargs=2,
                        default=[2000, 2000], metavar=("N_INJ", "N_NOISE"),
                        help="Number of validation injections and pure-noise "
                             "samples. Default: 2000 2000.")
    parser.add_argument("--psd-file", type=str, default=None,
                        help="HDF5 file of measured PSDs for Gaussian noise "
                             "generation. Default: analytic aLIGOZeroDetHighPower.")
    parser.add_argument("--real-noise-file", type=str, default=None,
                        help="HDF5 file of real (pre-whitened) detector noise. "
                             "Overrides --psd-file.")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Random seed. Default: 2026.")
    parser.add_argument("--verbose", action="store_true", help="Print progress.")
    parser.add_argument("--debug", action="store_true", help="Print debug messages.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite the output file if it already exists.")

    args = parser.parse_args()

    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARN
    logging.basicConfig(format="%(levelname)s | %(asctime)s: %(message)s",
                        level=log_level, datefmt="%d-%m-%Y %H:%M:%S")

    if os.path.isfile(args.output_file) and not args.force:
        raise RuntimeError(
            f"Output file '{args.output_file}' exists. Use --force to overwrite.")
    filemode = "w" if args.force else "w-"

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)

    detectors_abbr = ("H1", "L1", "V1", "K1")[:args.detectors]
    detectors = [pycbc.detector.Detector(abbr) for abbr in detectors_abbr]

    colored_sample_length = args.sample_length + MAX_FILTER_LENGTH

    if args.real_noise_file is None:
        noise_getter = GaussianNoiseGenerator(
            detectors=detectors,
            low_frequency_cutoff=LOW_FREQUENCY_CUTOFF,
            sample_length=colored_sample_length,
            fpath=args.psd_file,
        )
    else:
        noise_getter = RealNoiseGenerator(args.real_noise_file, detectors=detectors)
    noise_getter_iter = iter(noise_getter)

    logging.info("Generating training dataset.")
    train_ds = generate_dataset(
        args.training_samples, detectors, noise_getter_iter, rng,
        approximant=args.approximant, sample_length=args.sample_length,
        verbose=args.verbose)

    with h5py.File(args.output_file, filemode) as outfile:
        train_ds.save(outfile, "training")
        # Provenance: everything needed to regenerate this file exactly.
        outfile.attrs["schema_version"] = 1
        outfile.attrs["generator"] = "generate_dataset.py"
        outfile.attrs["seed"] = args.seed
        outfile.attrs["detectors"] = detectors_abbr
        outfile.attrs["sample_rate"] = SAMPLE_RATE
        outfile.attrs["low_frequency_cutoff"] = LOW_FREQUENCY_CUTOFF
        outfile.attrs["approximant"] = args.approximant
        outfile.attrs["mass_min_solar"] = MASS_MIN
        outfile.attrs["mass_max_solar"] = MASS_MAX
        outfile.attrs["noise_type"] = (
            "real" if args.real_noise_file is not None else "gaussian")
        outfile.attrs["waveform_normalisation"] = "network_snr_1"
        outfile.attrs["merger_placement_seconds"] = (
            f"[{args.sample_length * DELTA_T - MERGER_WINDOW_START:.2f}, "
            f"{args.sample_length * DELTA_T - MERGER_WINDOW_END:.2f}]")
        outfile.attrs["generation_config_json"] = canonical_json(vars(args))
        for label, path in (("psd", args.psd_file),
                            ("real_noise", args.real_noise_file)):
            if path is not None:
                outfile.attrs[f"{label}_sha256"] = sha256_file(path)
                outfile.attrs[f"{label}_path"] = os.path.abspath(path)

    logging.info("Generating validation dataset.")
    valid_ds = generate_dataset(
        args.validation_samples, detectors, noise_getter_iter, rng,
        approximant=args.approximant, sample_length=args.sample_length,
        verbose=args.verbose)
    with h5py.File(args.output_file, "a") as outfile:
        valid_ds.save(outfile, "validation")

    logging.info("Wrote %s", args.output_file)


if __name__ == "__main__":
    main()
