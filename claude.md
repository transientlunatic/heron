# Heron Project Notes - Claude Code Review

## Project Summary

Heron is a GP-based gravitational waveform surrogate for parameter estimation.
The current paper introduces waveform models with model uncertainty during PE.

Two key test scenarios:
1. **GPR model** (Matérn kernel + chirp warping) against injections
2. **Fake uncertainty model** (`IMRPhenomPv2_FakeUncertainty`) against injections

Asimov blueprints at `injections/asimov_blueprints.yaml` with 4 test cases
(q=0.3, 0.5, 0.7 fake uncertainty + q=0.5 standard baseline).

Old incorrect file `injections/asimov_ledger.yaml` used wrong format (`kind: ledger`
doesn't exist in asimov). Replaced with proper `kind: event` and `kind: analysis`
blueprints separated by `---`.

---

## Fixes Applied (2026-02-06)

### 1. Removed debug print statements from uncertainty likelihood
**File:** `heron/likelihood.py`
Removed 5 `print()` calls that would dump covariance matrices every likelihood call.

### 2. Fixed hardcoded `.cuda()` calls
**File:** `heron/models/gpytorch.py`
Changed all `.cuda()` to `.to(device)` or `.to(self.device)` so code works on
CPU-only systems. Affected both approximant classes and their evaluation methods.

### 3. Fixed `parameters.pop("time")` mutation
**File:** `heron/models/gpytorch.py`
Both `HeronNonSpinningApproximant.time_domain()` and
`HeronNonSpinningApproximantMatern.time_domain()` were mutating the caller's
parameters dict. Now creates a filtered copy instead.

### 4. Fixed fake covariance computation
**File:** `heron/models/lalsimulation.py`
- Added explicit `lengthscale` parameter (default 0.001s, appropriate for GW timescales)
- Uses relative times (`times - times[0]`) instead of GPS times
- Caches covariance matrix between calls (was recomputing N×N matrix every call)

### 5. Rewrote asimov blueprints
**File:** `injections/asimov_blueprints.yaml` (new, replaces `asimov_ledger.yaml`)
- Separate `kind: event` and `kind: analysis` documents (correct asimov format)
- Uses `needs:` for dependencies (not `dependencies:`)
- Added q=0.7 test case (best-performing mass ratio at 3.6% mismatch)
- 4 events, 8 analyses (4 injection + 4 inference)

---

## Known Issues (Not Yet Fixed)

### `make_injection_zero_noise` is broken
**File:** `heron/injection.py:91-138`
References undefined variable `settings` (line 122). Not currently used by
the asimov pipeline so not blocking, but should be fixed eventually.

### `HeronNonSpinningApproximant` references `self.warping` but never initializes it
**File:** `heron/models/gpytorch.py`
The non-Matérn class uses `self.warping` in evaluation methods but `__init__`
uses old manual warping. Only matters if someone uses the RBF-based class
(the Matérn class is the one being used for the paper).

### Vectorized likelihood is just a loop
**File:** `heron/sampling.py:81-90`
`allow_vectorised = True` but implementation loops over individual samples.
No actual speedup. Could be improved with batched GP evaluation.

### `otter` dependency in inference.py
**File:** `heron/inference.py:31`
`import otter` at module level - will crash if not installed. Should be
a lazy import or optional dependency.

### Asimov template may need updates
**File:** `heron/asimov/heron_template.yml`
The template renders `production.meta` into heron YAML config. Need to verify
that the blueprint fields flow correctly through the template, especially:
- `fixed_parameters` (not in current template)
- `data` section (populated by `InjectionPipeline.collect_assets()`)
- Interferometer names (`H1`/`L1` vs `AdvancedLIGOHanford`/`AdvancedLIGOLivingston`)

### GPR model not yet wired into asimov — RESOLVED
~~The blueprints only test `IMRPhenomPv2_FakeUncertainty`. To test the GPR model:~~
- ~~Need a way to load a trained GP model in the inference pipeline~~
- ~~Need `HeronNonSpinningApproximantMatern` registered in `KNOWN_WAVEFORMS`~~
- ~~Need an asimov blueprint that references it~~

All three done: checkpoint save/load added, registered as `HeronGPR` in
`KNOWN_WAVEFORMS`, inference.py handles `waveform.checkpoint` setting,
and GPR blueprints added (training → injection → inference).

---

## Additional Fixes Applied (2026-02-06, session 2)

### 6. GPR checkpoint save/load
**File:** `heron/models/gpytorch.py`
Added `save_checkpoint(path)` and `from_checkpoint(path)` classmethod to
`HeronNonSpinningApproximantMatern`. Saves model state_dicts, training data,
warping config, and metadata via `torch.save`. `from_checkpoint` reconstructs
the model without retraining.

### 7. Registered GPR in inference pipeline
**File:** `heron/inference.py`
- Added `HeronGPR` → `HeronNonSpinningApproximantMatern` to `KNOWN_WAVEFORMS`
- Waveform instantiation now checks for `waveform.checkpoint` setting and
  calls `from_checkpoint()` if present

### 8. Training pipeline
**Files:** `heron/train.py` (new), `heron/asimov/heron_training_template.yml` (new),
`heron/asimov/__init__.py`, `heron/cli.py`, `pyproject.toml`
- `heron train --settings <file>` CLI command
- Generates training data from reference approximant with chirp warping
- Trains `HeronNonSpinningApproximantMatern` and saves checkpoint
- `TrainingPipeline` registered as `"heron training"` asimov pipeline
- Entry point added to pyproject.toml
- GPR blueprints updated with training → injection → inference dependency chain

### 9. Fixed package configuration
**File:** `pyproject.toml`
- `packages` now lists all subpackages (`heron.asimov`, `heron.models`,
  `heron.training`) so templates and submodules are included

---

## Architecture Notes

- Likelihood classes: `TimeDomainLikelihood` (noise only) and
  `TimeDomainLikelihoodModelUncertainty` (noise + model covariance)
- Both use `NumericallyScaled` for numerical stability
- Multi-detector handled by `MultiDetector` (sums log-likelihoods)
- Sampling via nessai `FlowSampler` through `NessaiSampler` wrapper
- Asimov pipelines:
  - `InjectionPipeline` (name="heron injection") — creates frame files
  - `Pipeline` (name="heron") — runs PE inference
  - `TrainingPipeline` (name="heron training") — trains GPR model
- GP models use GPyTorch with Matérn or RBF kernels
- Time warping via `ChirpTimeWarping` with power-law transformation
- Best GP config: Matérn(ν=2.5) + ChirpTimeWarping(α=0.625)

## Key File Locations

- Likelihood: `heron/likelihood.py`
- Injection: `heron/injection.py`
- Inference: `heron/inference.py`
- Training: `heron/train.py`
- Sampling: `heron/sampling.py`
- GP models: `heron/models/gpytorch.py`
- Time warping: `heron/models/warping.py`
- LAL waveforms: `heron/models/lalsimulation.py`
- Asimov pipeline: `heron/asimov/__init__.py`
- Asimov inference template: `heron/asimov/heron_template.yml`
- Asimov training template: `heron/asimov/heron_training_template.yml`
- Asimov blueprints: `injections/asimov_blueprints.yaml`
- Test configs: `injections/fake_uncertainty_test_*.yaml`
