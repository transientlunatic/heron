# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Heron is a Gaussian Process Regression (GPR) toolkit for gravitational waveform modelling. It produces waveform surrogates with built-in uncertainty estimates. The key innovation is returning full covariance matrices alongside mean predictions, enabling uncertainty-aware parameter estimation.

The current active model is `ExactGPSurrogate` (Matérn kernel + chirp-time warping) in `heron/models/gp/exact.py`. The older `HeronNonSpinningApproximantMatern` in `heron/models/gpytorch.py` is a legacy class kept for checkpoint backward-compatibility.

## Commands

**Run all tests:**
```bash
pytest tests/ -v
```

**Run a single test file:**
```bash
pytest tests/test_gp_exact.py -v
```

**Lint:**
```bash
flake8 heron tests
```

**Install in development mode:**
```bash
pip install -e ".[dev]"
pip install -e ".[lal]"   # for LAL waveform approximants
```

**Training a model:**
```bash
heron train --settings config.yaml
```

**Evaluating a trained model:**
```bash
heron evaluate --settings eval.yaml
```

## Repository Layout

```
heron/                      Main Python package
  types.py                  Waveform, WaveformDict dataclasses (no heavy deps)
  train.py                  Training CLI (heron train)
  evaluate.py               Evaluation CLI (heron evaluate)
  cli.py                    Click entry-point wiring train + evaluate
  utils.py                  load_yaml and other helpers
  models/
    base.py                 WaveformSurrogate ABC
    gp/
      exact.py              ExactGPSurrogate — the primary model
      mean.py               NewtonianInspiralMean, TaylorT2Mean (PN mean functions)
      sparse.py             SparseGPSurrogate (inducing-point approximation)
    warping.py              ChirpTimeWarping, SimpleWarping, get_warping()
    lalsimulation.py        IMRPhenomPv2, SEOBNRv3 — reference LAL approximants
    testing.py              SineGaussianWaveform stub for tests
    gpytorch.py             LEGACY: HeronNonSpinningApproximantMatern (v1 checkpoint compat)
  training/
    dataset.py              TrainingSet dataclass (HDF5 save/load)
    sampling.py             sobol_sample, latin_hypercube_sample
    active.py               active_learning_loop (uncertainty-guided refinement)
    data.py                 LEGACY: DataWrapper001 (old HDF5 format)
  evaluation/
    mismatch.py             MismatchEvaluator
    calibration.py          CalibrationEvaluator
    report.py               EvaluationReport (summary + plots)
  asimov/                   Asimov pipeline integration (in progress — no __init__.py yet)
tests/
  test_gp_exact.py          ExactGPSurrogate — train, predict, save/load roundtrip
  test_types.py             Waveform and WaveformDict
  test_training.py          TrainingSet, sobol_sample, latin_hypercube_sample
  test_warping.py           ChirpTimeWarping, SimpleWarping
  test_mean_functions.py    PN mean functions
  test_evaluation.py        Evaluation pipeline
  test_evaluators.py        Individual evaluators
  models/test_lalsimulation.py  LAL approximant interface
  training-data/            Integration tests for training data generation
checkpoints/                Pre-trained checkpoint files (.pt)
scripts/                    One-off analysis scripts (not part of the package)
injections/                 Asimov blueprints and injection configs (work in progress)
```

## Architecture

### Data Types (`heron/types.py`)

- `Waveform`: numpy-only dataclass — `data`, `times`, `covariance`, `variance`, `std`
- `WaveformDict`: container for `plus`/`cross` polarisations; keyed by polarisation name

### Model Interface (`heron/models/base.py`)

`WaveformSurrogate` ABC enforces:
- `predict(parameters: dict) -> WaveformDict` — must return covariance, not just mean
- `save(path)` / `load(path)` classmethod for checkpoint round-trip
- `parameter_names: list[str]` and `parameter_bounds: dict[str, tuple]`

### ExactGPSurrogate (`heron/models/gp/exact.py`)

The primary production model. Architecture:
- Separate `_ExactGPModel` (GPyTorch `ExactGP`) for plus and cross polarisations
- Product Matérn kernel over each input dimension with `ScaleKernel`
- Time column warped via `ChirpTimeWarping` (power-law, default `alpha=0.375`)
- Output scaled by `output_scale=1e27` for numerical stability
- Checkpoint format is versioned: v2 saves unwarped training data; `_load_v1` handles old checkpoints from `HeronNonSpinningApproximantMatern`

### Time Warping (`heron/models/warping.py`)

- `ChirpTimeWarping(alpha, t_ref)`: `t_warp = sign(t) * |t/t_ref|^alpha * t_ref`
  - Best empirical config: `alpha=0.625`; default `alpha=0.375` (Newtonian)
- `SimpleWarping(scale)`: linear compression of `t < 0`
- `get_warping(type, **kwargs)` factory — use this everywhere, not constructors directly

### Training (`heron/train.py`, `heron/training/`)

Three training modes (set `training.mode` in YAML):
- `fixed`: grid of mass ratios, sampled uniformly in warped time
- `active`: iterative uncertainty-guided refinement (`active_learning_loop`)
- `data`: load pre-existing HDF5 `TrainingSet`

`TrainingSet` (`heron/training/dataset.py`):
- Stores `x` (N, D), `y_plus` (N,), `y_cross` (N,); last column of x is always time
- HDF5 save/load; `append()` for incremental active learning

### Evaluation (`heron/evaluate.py`, `heron/evaluation/`)

- `MismatchEvaluator`: computes overlap mismatch between surrogate and reference
- `CalibrationEvaluator`: checks GP uncertainty calibration against reference
- `EvaluationReport`: collects both, writes summary + optional matplotlib plots

## Training YAML Schema

```yaml
training:
  mode: fixed           # fixed | active | data
  approximant: IMRPhenomPv2
  mass_ratios: [0.3, 0.5, 0.7, 1.0]
  total_mass: 60.0      # solar masses
  distance: 100.0       # Mpc
  n_samples: 200        # time samples per mass ratio
  warping:
    type: chirp
    alpha: 0.625
  model: exact          # exact | sparse
  nu: 2.5
  output_scale: 1.0e27
  iterations: 400
  checkpoint: checkpoints/model.pt
  # optional:
  device: cpu           # or cuda
  save_training_data: checkpoints/training.h5
  mean_function:
    type: newtonian     # zero | newtonian | taylort2
```

## Known Issues

- `heron/models/gpytorch.py` — `HeronNonSpinningApproximant` (RBF class) references `self.warping` never set. Only `HeronNonSpinningApproximantMatern` is usable, and only for loading v1 checkpoints.
- `heron/training/data.py` — `DataWrapper001.add_waveform()` has a stray `print()` on line 393. Legacy code; new code uses `TrainingSet`.
- `heron/asimov/` — directory exists but has no `__init__.py`; Asimov integration not yet functional.
- `heron/models/warping.py` — `PiecewiseWarping.unwarp()` raises `NotImplementedError`.
- `heron/models/gp/sparse.py` — `SparseGPSurrogate` exists and is referenced in train.py but may not be fully implemented.

## Environment

A local virtualenv is at `./environment/`. Core dependencies: `torch`, `gpytorch`, `numpy>=2.2`, `scipy`, `h5py`, `pyyaml`, `click`. Optional: `lalsuite`, `astropy` (for LAL approximants), `matplotlib` (for plots).
