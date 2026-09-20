#!/usr/bin/env python
"""
Generate a held-out TEST set of binary-black-hole signals in detector noise.

This writes a single group, ``testing``, and nothing else. It exists as a
separate script from generate_dataset.py so that a test set cannot be confused
with, or accidentally drawn from, the data a model was fitted on: the file it
produces has no ``training`` group to reach for by mistake.

The signal simulation, projection, SNR normalisation and whitening are imported
from generate_dataset.py rather than reimplemented, so the test data is built
by exactly the same code as the training data. Only the bookkeeping differs.

------------------------------------------------------------------------------
WHAT MAKES A TEST SET HELD OUT
------------------------------------------------------------------------------
Passing a different --seed is necessary but not sufficient, and what it buys
depends on the noise:

  * Gaussian noise. A different seed gives independent noise realisations and
    independent source parameters, because both are drawn afresh from the same
    analytic PSD. That is genuinely held out.

  * Real noise. A different seed does NOT help, because the segments come from
    a file. Point --real-noise-file at the TEST half produced by
    prepare_real_noise.py split, never at the half used for training. Slices
    within one half overlap by 50%, so sharing a half means sharing data.

------------------------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------------------------
Requires: pycbc, numpy, h5py, tqdm, and generate_dataset.py alongside.

    # Gaussian noise: a different seed from the training run is enough
    python generate_test_data.py -o data_test.h5 --seed 7 \
        --samples 5000 5000 --verbose

    # Real noise: use the half that prepare_real_noise.py set aside
    python prepare_real_noise.py split real_noise_file.hdf \
        train_noise.hdf test_noise.hdf --duration 1e6
    python prepare_real_noise.py slice test_noise.hdf -o sliced_test.hdf
    python generate_test_data.py -o data_o3_test.h5 --seed 7 \
        --real-noise-file sliced_test.hdf --verbose

    # Waveforms with precession and higher-order modes, to test beyond the
    # family the baseline was trained on
    python generate_test_data.py -o data_test_xphm.h5 -a IMRPhenomXPHM --seed 7

Then use it:

    python plot_roc.py -w runs/baseline/best_state_dict.pt -d data_test.h5 \
        -o figures/roc
    python plot_dataset.py data_test.h5 --group testing

------------------------------------------------------------------------------
Copyright 2026 Chayan Chatterjee. Shares its machinery with
generate_dataset.py, adapted from the MLGWSC-1 generation code of
Ondrej Zelenka (https://github.com/ondrzel/ml-gw-search), Apache License 2.0.
------------------------------------------------------------------------------
"""

from argparse import ArgumentParser
import logging
import os
import os.path

import h5py

import pycbc.detector

# Imported, not duplicated: the test set must be built by the same code as the
# training set, or a difference in preprocessing is indistinguishable from a
# difference in the model.
from generate_dataset import (
    DELTA_T,
    GaussianNoiseGenerator,
    LOW_FREQUENCY_CUTOFF,
    MASS_MAX,
    MASS_MIN,
    MAX_FILTER_LENGTH,
    MERGER_WINDOW_END,
    MERGER_WINDOW_START,
    RealNoiseGenerator,
    SAMPLE_RATE,
    canonical_json,
    generate_dataset,
    seed_everything,
    sha256_file,
)

import numpy as np

TEST_GROUP = "testing"


def main():
    parser = ArgumentParser(
        description="Generate a held-out test set. Writes a single 'testing' "
                    "group, so it cannot be mistaken for training data.")

    parser.add_argument("-o", "--output-file", type=str, required=True,
                        help="Path to the HDF5 file to write.")
    parser.add_argument("--samples", type=int, nargs=2, default=[5000, 5000],
                        metavar=("N_INJ", "N_NOISE"),
                        help="Number of injections and pure-noise samples. "
                             "Default: 5000 5000.")
    parser.add_argument("-a", "--approximant", type=str, default="IMRPhenomD",
                        choices=["IMRPhenomD", "IMRPhenomXPHM"],
                        help="Waveform model. Use IMRPhenomXPHM to test "
                             "against precession and higher-order modes the "
                             "baseline was not trained on. Default: "
                             "IMRPhenomD.")
    parser.add_argument("-d", "--detectors", type=int, default=2, choices=[2],
                        help="Number of detectors. The chapter uses H1/L1.")
    parser.add_argument("-s", "--sample-length", type=int, default=2048,
                        help="Length of each sample in samples, at 2048 Hz. "
                             "Must match the training data. Default: 2048.")
    parser.add_argument("--psd-file", type=str, default=None,
                        help="HDF5 file of measured PSDs for Gaussian noise. "
                             "Default: analytic aLIGOZeroDetHighPower.")
    parser.add_argument("--real-noise-file", type=str, default=None,
                        help="HDF5 file of real, pre-whitened noise. Use the "
                             "TEST half from prepare_real_noise.py split, "
                             "never the training half. Overrides --psd-file.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed. MUST differ from the seed used for "
                             "the training data. Default: 7.")
    parser.add_argument("--verbose", action="store_true", help="Print progress.")
    parser.add_argument("--debug", action="store_true", help="Debug messages.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite the output file if it exists.")

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
            f"Output file '{args.output_file}' exists. Use --force to "
            f"overwrite.")

    # The seed is the only thing separating a Gaussian test set from the
    # training set, so a collision with the training default is worth flagging.
    if args.real_noise_file is None and args.seed == 2026:
        logging.warning(
            "Seed 2026 is the default used by generate_dataset.py. If the "
            "training data was generated with it, this test set will "
            "reproduce the same noise realisations and source parameters. "
            "Pass a different --seed.")
    if args.real_noise_file is not None:
        logging.info(
            "Using real noise from %s. Confirm this is the TEST half from "
            "prepare_real_noise.py split: a different seed does not make "
            "shared strain independent.", args.real_noise_file)

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
        noise_getter = RealNoiseGenerator(args.real_noise_file,
                                          detectors=detectors)
    noise_getter_iter = iter(noise_getter)

    logging.info("Generating test dataset.")
    test_ds = generate_dataset(
        args.samples, detectors, noise_getter_iter, rng,
        approximant=args.approximant, sample_length=args.sample_length,
        verbose=args.verbose)

    with h5py.File(args.output_file, "w" if args.force else "w-") as outfile:
        test_ds.save(outfile, TEST_GROUP)
        outfile.attrs["schema_version"] = 1
        outfile.attrs["generator"] = "generate_test_data.py"
        outfile.attrs["purpose"] = "held-out test set"
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

    n_injections, n_noise = args.samples
    logging.info("Wrote %s: group '%s' with %i injections and %i pure-noise "
                 "samples.", args.output_file, TEST_GROUP, n_injections,
                 n_noise)
    print(f"Wrote {args.output_file}")


if __name__ == "__main__":
    main()
