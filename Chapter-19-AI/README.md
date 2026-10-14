# Machine learning for gravitational-wave searches — companion code

Code and figures for Chapter 19. The chapter builds a transformer-based search for binary black hole
mergers, shows where it fails, and fixes it in three steps. This repository reproduces every result it reports.

---

## Installation

The stack needs Python 3.9 or newer.

```bash
git clone https://github.com/physicish/GWR/tree/main/Chapter-19-AI

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
```

**Install PyTorch first** if you need a particular CUDA build, because
`ml4gw` depends on it and pip will otherwise choose for you. Pick the command
for your platform from <https://pytorch.org/get-started/locally/>, for example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Then install everything else and the shared package:

```bash
pip install -r requirements.txt
pip install -e .
```

Check that it worked:

```bash
python -c "import torch, pycbc, ml4gw, h5py; print('ok')"
python -c "from gwsearch import CoherentTransformer; \
           print(sum(p.numel() for p in CoherentTransformer().parameters()), 'parameters')"
```

The second command should print `664522`, matching the chapter.

### If the install fails

`pycbc` pulls in `lalsuite`, which ships large binary wheels and is the usual
source of trouble. On Linux and macOS the wheels normally resolve; on Windows
they do not, and WSL is the practical route. If pip cannot solve the
environment, installing the gravitational-wave stack through conda first and
then pip-installing the rest works:

```bash
conda create -n gwsearch python=3.11
conda activate gwsearch
conda install -c conda-forge pycbc lalsuite
pip install -r requirements.txt && pip install -e .
```

`ml4gw` is needed only by stage 05. Every other stage runs without it, so a
failed `ml4gw` install is not a reason to stop.

---

## Hardware and running time

Every result in the chapter was produced on a single NVIDIA A100 80 GB.

| Stage | What it does | Time |
|---|---|---|
| 01 | Generate Gaussian training data | ~1 h, CPU |
| 02 | Train on Gaussian noise, 250 epochs | ~30 min |
| 03 | Evaluate on Dataset 4 | ~18 min per file |
| 04 | Prepare real noise, retrain | ~2 h |
| 05 | Mixed ml4gw training | ~5 h |
| 06 | Train and evaluate the coincident model | ~5 h |
| 07 | 100 time slides and validation plot | ~30 min, CPU |

The MLGWSC-1 real-noise file is a 94 GB download. Stages 01 to 03 need only
simulated data and are the cheapest place to start.

---

## Repository layout

```
gwsearch/                   Shared package: models, scorer, loss-curve plotting
01_generate_gaussian_data/  Simulated training, validation and test sets
02_train_coherent_gaussian/ Train the coherent transformer   -> Figure 1.1
03_evaluate_dataset4/       Run it as a search on Dataset 4  -> Figure 1.2(a)
04_train_on_real_noise/     Retrain on real O3a noise        -> Figure 1.2(b)
05_mixed_ml4gw_training/    Add on-the-fly generation        -> Figure 1.2(c)
06_coincident_model/        Per-detector scoring             -> Figure 1.2(d)
07_background_validation/   Time slides and FAR validation   -> Figure 1.3
supplementary/              The experiment behind Box 2
```

Each directory has its own README with the exact commands and the numbers that
command should reproduce.

---

## External data

Stages 03 onward need files this repository does not contain.

**MLGWSC-1 Dataset 4** — foreground and background, generated with the
challenge's own tools:

```bash
git clone https://github.com/gwastro/ml-mock-data-challenge-1
cd ml-mock-data-challenge-1
python generate_data.py -d 4 -i ds4_fg.hdf -b ds4_bg.hdf --duration 2592000
```

**Real O3a noise for training** — the same repository supplies the download and
segment selection. Stage 04's README covers the preparation; see also
`segments.csv`, which the challenge's `generate_data.py` fetches automatically.
