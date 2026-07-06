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
- Time column warped via `ChirpTimeWarping` (power-law, best `alpha=0.625`)
- Output scaled by `output_scale=1e27` for numerical stability
- Checkpoint format is versioned: v2 saves unwarped training data; `_load_v1` handles old checkpoints from `HeronNonSpinningApproximantMatern`

**Critical hyperparameter constraints** — without these, L-BFGS collapses lengthscales to ~0.001 and the GP treats each training point independently:
- `ls_min_q` (default 0.0005): set to the q-spacing of your training grid. E.g., 5 mass ratios at spacing 0.2 → `ls_min_q=0.2`. Without it, the GP cannot interpolate between mass ratios and reverts to the prior at any untrained q.
- `ls_min_time` (default 0.0005): set to ~2× the warped-time training spacing. E.g., 200 samples → spacing 0.015 s → `ls_min_time=0.030`. Without it, the GP reverts to the prior between training times.
- `noise_floor_rel` (default 1e-6): set to 1e-3 when `ls_min > training_spacing` (highly correlated training points → ill-conditioned kernel matrix → CG NaN). This regularises K+σ²I without inflating the surrogate uncertainty in the likelihood.

**Training uses Cholesky, not CG** (enforced via `gpytorch.settings.max_cholesky_size(cholesky_size)`, default 2000, inside `_train()` and `predict()`). The "N ≤ ~2000" figure was a CPU-only assumption — on GPU, raising `cholesky_size` (e.g. to 6000) trains cleanly at much higher N; see `examples/train_phenomd_hf_dense30.yaml`. `SparseGPSurrogate` is still available for N beyond what GPU Cholesky can handle, but is no longer the first choice for N in the low thousands.

**Use the latent GP covariance in the likelihood** — `model(x)`, not `model.likelihood(model(x))`. LALSuite training data is noiseless; the trained σ²_noise is a regularisation artefact. The predictive (latent + noise) covariance inflates K unnecessarily. `predict()` already uses the latent distribution.

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

### GW Likelihood (`heron/gw_likelihood.py`, `heron/noise.py`, `heron/detector.py`)

The GP-marginalised GW likelihood `p(d|θ) = N(d | μ(θ), C + K(θ))`:
- `GWLikelihood`: wraps the surrogate; computes `log p(d|θ)` for one detector
- `use_waveform_uncertainty=True/False` switches GP-marginalised vs. standard matched-filter
- Data and signal are HP-filtered at `f_low`; noise covariance C from `noise_covariance(times, psd_fn, f_low)`
- K is approximated as **diagonal** (HP filtering creates negative eigenvalues in the full projected K — diagonal stays positive-definite)
- GP predict casts to `float64` before dividing by `output_scale²` — float32 underflows at 1e-42

**The log-determinant bias**: the likelihood has two competing terms — a data-fit term (peaks at true θ) and a log-det term `−½ log|C+K(θ)|` (penalises high-K regions, i.e. pulls posteriors toward training points). When K/C is large *and* varies strongly with θ (sparse training grid), the log-det term dominates and the posterior is biased toward training parameters. Rule of thumb:
- K_prior/C ≈ SNR²/N_samples (e.g., SNR=163, N=256 → K_prior/C ≈ 144)
- K_mid/C < 1 requires q-spacing ≤ 0.03 (30 qs); unbiased posterior requires q-spacing ≤ 0.02 (45 qs)
- At high SNR (>30), a training grid of 5–10 mass ratios is insufficient for unbiased q PE

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
    alpha: 0.625        # best empirical value; 0.375 is Newtonian
  model: exact          # exact | sparse
  nu: 2.5
  output_scale: 1.0e27
  optimizer: lbfgs      # lbfgs (default) | adam
  iterations: 80        # L-BFGS outer steps; ~80 usually enough
  checkpoint: checkpoints/model.pt
  # Lengthscale lower bounds — MUST be set; see ExactGPSurrogate notes above
  ls_min_q: 0.2         # set to q-spacing of mass_ratios grid
  ls_min_time: 0.030    # set to ~2× warped-time training spacing
  noise_floor_rel: 1.0e-3  # raise from 1e-6 when ls_min > training spacing
  # optional:
  device: cpu           # or cuda
  save_training_data: checkpoints/training.h5
  mean_function:
    type: newtonian     # zero | newtonian | taylort2
```

A worked example: `examples/train_phenomd_hf.yaml` (5 mass ratios, N=1000, ls_min_q=0.2, ls_min_time=0.030).

## Known Issues

- `heron/models/gpytorch.py` — `HeronNonSpinningApproximant` (RBF class) references `self.warping` never set. Only `HeronNonSpinningApproximantMatern` is usable, and only for loading v1 checkpoints.
- `heron/training/data.py` — `DataWrapper001.add_waveform()` has a stray `print()` on line 393. Legacy code; new code uses `TrainingSet`.
- `heron/asimov/` — directory exists but has no `__init__.py`; Asimov integration not yet functional.
- `heron/models/warping.py` — `PiecewiseWarping.unwarp()` raises `NotImplementedError`.
- `heron/models/gp/sparse.py` — `SparseGPSurrogate` has test coverage (`tests/test_gp_sparse.py`) and several past bugs are fixed (kwarg routing in `train.py`, missing `ls_min_q`/`ls_min_time` floors, `predict()` used predictive instead of latent covariance, float32 underflow when dividing by `output_scale**2`, lengthscale init anchored at `data_range/4` instead of `ls_min` on dense grids). A real fit was recovered on the 30-mass-ratio/N=6000 dense grid using `NaturalVariationalDistribution` + `gpytorch.optim.NGD`, but **only after discovering the ExactGPSurrogate-standard `alpha=0.625` chirp warping is specifically bad for this SVGP setup** (r≈0.08 with own training data) vs `alpha=0.375`, `ChirpTimeWarping`'s default (r≈0.74) — `ls_min_time` must be rescaled for whichever alpha is used, since the warped-time range/spacing itself depends on alpha. See `examples/train_phenomd_hf_sparse.yaml`. Tuning across all three available levers — `n_inducing` (300→1000), training iterations (150/400/1200), and `ngd_lr` (0.03/0.1/0.3) — plateaued at K/C mean≈176-194, still ~8x worse than the exact-GP dense10 grid (K/C=22); iteration counts above ~600 actively degrade the fit (NGD drifts past the optimum, confirmed via mean-fit correlation dropping from r=0.78 to r=0.71). **Superseded by GPU exact-GP Cholesky training at the same density** — see below. Kept for reference; not the recommended path when a GPU is available.
- `heron/gw_likelihood.py` — K is approximated as diagonal (HP filtering produces negative eigenvalues in the full projected covariance). A regularised full-matrix solve would be more correct.
- **Log-det bias — greatly reduced, not fully eliminated, at 30-mass-ratio density on GPU.** The "safe to N≈2000" Cholesky ceiling was a CPU-only assumption, never stress-tested — `ExactGPSurrogate` now exposes `cholesky_size` (see `heron/models/gp/exact.py`) instead of a hardcoded 2000, and training the same 30-mass-ratio/N=6000 grid that `SparseGPSurrogate` struggled with (see above) works cleanly on GPU: `examples/train_phenomd_hf_dense30.yaml`, checkpoint `checkpoints/phenomd_nonspinning_dense30.pt`. Result: K/C mean≈1.5-2, essentially flat across the whole trained q range (`scripts/kc_profile_scan.py`), vs 22 for the 10-mass-ratio grid and 176 for the best `SparseGPSurrogate` tuning — by far the best model produced so far. Not a complete fix: 0% of q-points have mean K/C strictly below 1, and the posterior-width bias signature (with-K posterior narrower than without-K, the opposite of the physically-expected direction) reappears mildly at low SNR / near the edge of the trained q range (σ_q width ratio ×0.9 at q=0.2, SNR≈79) even though it's clearly sane at higher SNR (×1.2-2.3 at q=0.5/0.8). The original diagnosis stands — the log-det term `−½ log|C+K(θ)|` competing with the data-fit term — just at a much-reduced magnitude. Next lever: the full 45-mass-ratio/N=9000/spacing≤0.02 grid (CLAUDE.md's condition for complete unbiasing) is untested but plausibly tractable given GPU-Cholesky now works cleanly at N=6000.

## Environment

A local virtualenv is at `./environment/`. Core dependencies: `torch`, `gpytorch`, `numpy>=2.2`, `scipy`, `h5py`, `pyyaml`, `click`. Optional: `lalsuite`, `astropy` (for LAL approximants), `matplotlib` (for plots).
