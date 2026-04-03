# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Heron is a Gaussian Process Regression (GPR) toolkit for gravitational waveform modelling. It produces waveform surrogates with built-in uncertainty estimates, which are used directly during Bayesian parameter estimation (PE) for compact binary coalescences.

The active paper work focuses on two test scenarios:
1. **GPR model** (`HeronNonSpinningApproximantMatern`) with Matérn kernel + chirp-time warping
2. **Fake uncertainty model** (`IMRPhenomPv2_FakeUncertainty`) — LAL waveform with synthetic covariance

## Commands

**Run all tests:**
```bash
pytest tests/
```

**Run a single test file:**
```bash
pytest tests/test_td_likelihood.py -v
```

**Run a single test:**
```bash
pytest tests/test_td_likelihood.py::TestTDLikelihood::test_likelihood -v
```

**Lint:**
```bash
flake8 heron tests
```

**Install in development mode:**
```bash
pip install -e ".[dev]"
```

**CLI commands:**
```bash
heron inference --settings <config.yaml>   # Run PE inference
heron injection --settings <config.yaml>   # Create injection frames
heron train --settings <config.yaml>       # Train a GPR model checkpoint
```

**Asimov pipeline (HTCondor job submission):**
```bash
asimov apply --file injections/asimov_blueprints.yaml
asimov build  # build DAGs
asimov submit # submit to condor
```

## Architecture

### Core Data Flow

Training data → `HeronNonSpinningApproximantMatern` (GPR) → checkpoint `.pt` file → loaded in `heron_inference` → `TimeDomainLikelihoodModelUncertainty` → `NessaiSampler` → posterior samples

### Key Modules

**`heron/models/`** — Waveform models
- `gpytorch.py`: GPR surrogate models. `HeronNonSpinningApproximantMatern` is the active model (Matérn kernel). `HeronNonSpinningApproximant` is the older RBF-based class (partially broken — `self.warping` uninitialized). Both subclass `WaveformSurrogate` from `models/__init__.py`.
- `lalsimulation.py`: LAL-based approximants (`IMRPhenomPv2`, `SEOBNRv3`, `IMRPhenomPv2_FakeUncertainty`). Base class `LALSimulationApproximant` handles parameter conversion (mass ratio, solar mass units, distance).
- `warping.py`: Time coordinate warping. `ChirpTimeWarping(alpha, t_ref)` with power-law transformation is the production warping. Best config: `alpha=0.625`.
- `testing.py`: Stub waveforms (`SineGaussianWaveform`, `FlatPSD`) used in tests.

**`heron/likelihood.py`** — Likelihood functions
- `NumericallyScaled`: Wraps covariance matrices with a preconditioning scale to avoid numerical overflow.
- `TimeDomainLikelihood`: Noise-only likelihood.
- `TimeDomainLikelihoodModelUncertainty`: Noise + GP model covariance combined. The key innovation.
- `MultiDetector`: Sums log-likelihoods across detectors.

**`heron/sampling.py`** — `NessaiSampler` wraps `nessai.model.Model` to interface Heron likelihoods with the nessai normalising flow sampler. Note: `allow_vectorised = True` but the implementation loops; no actual vectorization speedup.

**`heron/inference.py`** — End-to-end PE run. Reads YAML config, instantiates detectors/PSD/waveform/likelihood/sampler. `KNOWN_WAVEFORMS` dict maps string names to classes (`"HeronGPR"` → `HeronNonSpinningApproximantMatern`). When a `waveform.checkpoint` key is present, calls `from_checkpoint()` instead of training from scratch. **Known issue**: `import otter` at module level will crash if `otter-report` is not installed.

**`heron/train.py`** — Training pipeline. `generate_training_data()` calls a LAL approximant at multiple mass ratios, applies chirp-time warping, then trains GPR. Saves via `save_checkpoint(path)`.

**`heron/injection.py`** — Creates synthetic strain data (frame files) for injection studies. `make_injection_zero_noise` is broken (references undefined `settings` variable, line ~122).

**`heron/asimov/__init__.py`** — Three Asimov pipeline classes registered as entry points:
- `Pipeline` ("heron") — runs PE via `heron inference`
- `InjectionPipeline` ("heron injection") — creates injections
- `TrainingPipeline` ("heron training") — trains GPR model
- All submit HTCondor jobs via `MetaPipeline.build_dag()`.

**`heron/datatypes.py`** — Extends GWPy `TimeSeries` with variance/covariance support and projection utilities. `Waveform` and `WaveformDict` are the main return types from model calls.

**`heron/detector.py`** — `KNOWN_IFOS` dict mapping detector names (e.g. `"AdvancedLIGOLivingston"`, `"H1"`) to detector classes.

### Asimov Integration

Blueprints (`injections/asimov_blueprints.yaml`) define `kind: event` and `kind: analysis` documents. Each analysis specifies a `pipeline:` name, which Asimov maps to the registered entry points. The Jinja2 templates (`heron/asimov/heron_template.yml`, `heron_training_template.yml`) render YAML configs for individual runs.

### Known Issues to Be Aware Of

- `heron/injection.py:122`: `make_injection_zero_noise` references undefined `settings` — not currently used by asimov.
- `heron/models/gpytorch.py`: `HeronNonSpinningApproximant` (RBF class) references `self.warping` which is never set — use `HeronNonSpinningApproximantMatern` instead.
- `heron/sampling.py:81-90`: `allow_vectorised = True` but loops over samples — no actual batching.
- `heron/inference.py:31`: `import otter` at module level — crashes without `otter-report` installed.

## Environment

A local virtualenv is at `./environment/`. The project uses `lalsuite`, `gpytorch`, `torch`, `bilby`, `nessai`, and `asimov` as primary dependencies. GPU (CUDA) is used automatically if available; the code falls back to CPU.
