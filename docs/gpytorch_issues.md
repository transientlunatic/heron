# GPyTorch Model Issues and Improvements

This document tracks known issues, performance problems, and potential improvements for the GPyTorch-based surrogate models in Heron.

**Generated:** 2026-01-12
**File:** `heron/models/gpytorch.py`

---

## A. Bugs / Mistakes

### 1. Hard-coded CUDA calls (HIGH PRIORITY)
**Location:** [gpytorch.py:120-121](heron/models/gpytorch.py#L120-L121), [130](heron/models/gpytorch.py#L130), [146](heron/models/gpytorch.py#L146)

Multiple `.cuda()` calls ignore the global `device` variable and will crash if CUDA is unavailable:
```python
self.train_y_plus = train_y_plus.cuda() * self.output_scale  # Should use .to(self.device)
self.train_y_cross = train_y_cross.cuda() * self.output_scale
self.models["plus"].likelihood.cuda()
test_mass_ratio = torch.linspace(...).cuda()
```

**Impact:** Code crashes on CPU-only systems
**Fix:** Replace all `.cuda()` with `.to(self.device)`

---

### 2. Mutable default argument (MEDIUM PRIORITY)
**Location:** [gpytorch.py:20](heron/models/gpytorch.py#L20), [44](heron/models/gpytorch.py#L44)

Creates shared likelihood instance across all model instantiations:
```python
def __init__(self, train_x, train_y, likelihood=gpytorch.likelihoods.GaussianLikelihood()):
```

**Impact:** Multiple model instances share the same likelihood object, causing unexpected behavior
**Fix:** Use `likelihood=None` with instantiation inside `__init__`

---

### 3. Dictionary key inconsistency (LOW PRIORITY)
**Location:** [gpytorch.py:163](heron/models/gpytorch.py#L163)

Uses `"total mass"` (with space) while everywhere else uses `"total_mass"` (with underscore).

**Impact:** Parameter may not be found, causing silent failures
**Fix:** Standardize on `"total_mass"`

---

### 4. Missing parameter handling (MEDIUM PRIORITY)
**Location:** [gpytorch.py:229](heron/models/gpytorch.py#L229)

`parameters.pop("time")` will crash if "time" doesn't exist (which happens when `times` argument is provided).

**Impact:** Crashes when calling `time_domain()` with explicit `times` argument
**Fix:** Use `parameters.pop("time", None)` or check existence first

---

### 5. Inconsistent training data storage (LOW PRIORITY)
**Location:** [gpytorch.py:22](heron/models/gpytorch.py#L22) vs [47-48](heron/models/gpytorch.py#L47-L48), [80](heron/models/gpytorch.py#L80)

`ExactGPModel` doesn't store `train_x`/`train_y` as attributes, but `ExactGPModelKeOps` does. The `train()` method at line 80 accesses `model.train_x`, which won't exist for `ExactGPModel`.

**Impact:** Training will crash if using `ExactGPModel` instead of `ExactGPModelKeOps`
**Fix:** Add `self.train_x` and `self.train_y` to `ExactGPModel.__init__`

---

## B. Performance Issues

### 1. No training progress monitoring (MEDIUM PRIORITY)
**Location:** [gpytorch.py:78-83](heron/models/gpytorch.py#L78-L83)

Training loop runs silently with no loss logging.

**Impact:** Impossible to diagnose convergence issues or determine if more iterations are needed
**Improvement:** Add optional logging of loss values, possibly with tqdm progress bar

---

### 2. Redundant GPU transfers (LOW PRIORITY)
**Location:** [gpytorch.py:172](heron/models/gpytorch.py#L172), [238](heron/models/gpytorch.py#L238)

Time axis unwarping modifies tensors in-place on GPU after copying points:
```python
points[points[:, 1] < 0, 1] = points[points[:, 1] < 0, 1] * self.warp_scale
```

**Impact:** Creates unnecessary boolean masks on GPU
**Improvement:** Pre-compute unwarping indices or use more efficient indexing

---

### 3. Inefficient tensor creation (LOW PRIORITY)
**Location:** [gpytorch.py:209-224](heron/models/gpytorch.py#L209-L224)

Creates separate tensors then vstacks instead of using `torch.cartesian_prod` like in `_make_evaluation_manifold`.

**Impact:** Slower and uses more memory
**Improvement:** Use `torch.cartesian_prod` consistently

---

### 4. No GP hyperparameter caching (MEDIUM PRIORITY)
**Location:** Entire class

Trained models don't support serialization.

**Impact:** Must retrain from scratch every time, wasting computation
**Improvement:** Implement `save_model()` and `load_model()` methods

---

### 5. Fixed learning rate (MEDIUM PRIORITY)
**Location:** [gpytorch.py:75](heron/models/gpytorch.py#L75)

`lr=0.05` is high and may cause instability.

**Impact:** Training may oscillate or fail to converge
**Improvement:** Use learning rate scheduling or adaptive optimizers (e.g., AdamW with scheduler)

---

### 6. Covariance matrix always computed (LOW PRIORITY)
**Location:** [gpytorch.py:168](heron/models/gpytorch.py#L168), [233-235](heron/models/gpytorch.py#L233-L235)

Even with `fast_pred_var()`, full covariance matrix is stored and transferred lazily.

**Impact:** Memory overhead when only mean predictions are needed
**Improvement:** Add option to skip covariance computation entirely

---

### 7. Missing batch prediction (LOW PRIORITY)
**Location:** `time_domain()` method

Could vectorize predictions across multiple parameter points.

**Impact:** Slower when evaluating many parameter combinations
**Improvement:** Support batch evaluation

---

## C. Potential Improvements

### 0. Kernel Architecture Review (MEDIUM-HIGH PRIORITY)

**See detailed analysis**: [gpytorch_kernel_analysis.md](gpytorch_kernel_analysis.md)

The current kernel is a product of two RBF kernels (one for mass ratio, one for time). This works for basic interpolation but has several limitations:

**Key Issues:**
- Assumes stationarity in time (but CBC waveforms have distinct inspiral/merger/ringdown phases)
- Zero mean function (ignores known physics of waveform envelope)
- No periodicity modeling (inefficient for oscillatory signals)
- Independent plus/cross models (ignores physical correlations)
- Fixed lengthscale constraints (not adaptive to different parameter spaces)

**Recommended improvements:**
1. Use Matérn or Spectral Mixture kernels for time dimension
2. Add physics-informed mean function (PN approximation + ringdown)
3. Multi-output GP to capture plus/cross correlations
4. Add tests for phase accuracy and frequency content

**Benefit:** Better extrapolation, more efficient representation, physics-informed uncertainties

---

### 1. Add model checkpointing (HIGH PRIORITY)
Save/load trained models to avoid retraining:
```python
def save_model(self, filepath):
    torch.save({'models': self.models, 'hyperparams': ...}, filepath)
```

**Benefit:** Dramatically reduces workflow time for repeated analyses

---

### 2. Implement early stopping (MEDIUM PRIORITY)
Monitor validation loss and stop when convergence plateaus.

**Benefit:** Faster training, prevents overfitting

---

### 3. Add support for sparse GPs (LOW PRIORITY)
For larger datasets, use inducing points (VariationalGPs) to reduce O(n³) complexity.

**Benefit:** Scale to much larger training datasets

---

### 4. Parameterize kernel constraints (MEDIUM PRIORITY)
**Location:** [gpytorch.py:28](heron/models/gpytorch.py#L28), [32](heron/models/gpytorch.py#L32), [57-59](heron/models/gpytorch.py#L57-L59)

Hard-coded lengthscale constraints should be configurable based on parameter ranges.

**Benefit:** Better fit quality across different parameter spaces

---

### 5. Add uncertainty quantification utilities (LOW PRIORITY)
Currently returns covariance but no helper methods for confidence intervals or prediction bands.

**Benefit:** Easier to use uncertainty information in downstream analyses

---

### 6. Support multi-output GPs (LOW PRIORITY)
Instead of training separate models for plus/cross polarizations, use a multi-task GP.

**Benefit:** Can capture correlations between polarizations, potentially better predictions

---

### 7. Add adaptive time warping (LOW PRIORITY)
The `warp_scale` is fixed but could be learned as a GP hyperparameter.

**Benefit:** More flexible, data-driven time scaling

---

### 8. Better mean function (LOW PRIORITY)
Uses `ZeroMean()` but could use a physically-motivated mean (e.g., post-Newtonian approximation).

**Benefit:** Better predictions, especially for extrapolation

---

### 9. Add automatic relevance determination (ARD) (MEDIUM PRIORITY)
Different lengthscales per dimension could improve fit quality.

**Benefit:** Model can learn which dimensions are most important

---

## D. Testing Gaps

### 1. No unit tests for individual methods (HIGH PRIORITY)
Only integration tests exist in [test_gpytorch.py](tests/models/test_gpytorch.py).

**Missing tests:**
- `_make_evaluation_manifold()`
- Time warping/unwarping logic
- Mass/distance scaling factors

---

### 2. No error handling tests (HIGH PRIORITY)
**Missing scenarios:**
- Invalid parameters passed
- CUDA unavailable but code calls `.cuda()`
- Training fails to converge
- Parameters missing from dictionary

---

### 3. No numerical stability tests (MEDIUM PRIORITY)
**Missing scenarios:**
- Extreme mass ratios (near 0 or 1)
- Very long/short time series
- Numerical overflow with `output_scale=1e27`

---

### 4. No performance benchmarks (LOW PRIORITY)
**Missing metrics:**
- Training time vs dataset size
- Prediction time vs number of test points
- Memory usage scaling

---

### 5. No hyperparameter sensitivity tests (MEDIUM PRIORITY)
**Missing scenarios:**
- Different learning rates
- Sensitivity to kernel constraints
- Number of training iterations required

---

### 6. No serialization tests (MEDIUM PRIORITY)
Can models be saved and loaded correctly?

---

### 7. No uncertainty calibration tests (HIGH PRIORITY)
Tests check overlap but not whether uncertainties are calibrated (e.g., do 95% confidence intervals contain the true value 95% of the time?).

---

### 8. Edge case for lazy covariance transfer (LOW PRIORITY)
No test for GPU-to-CPU transfer mechanism in [types.py:145-170](heron/types.py#L145-L170).

---

### 9. No test for mixed parameter specifications (LOW PRIORITY)
Code handles both `"mass_ratio"` and `"mass ratio"` - needs explicit testing.

---

### 10. Missing test for global disable_cuda flag (LOW PRIORITY)
**Location:** [gpytorch.py:11](heron/models/gpytorch.py#L11)

No way to test CPU fallback behavior.

---

## Priority Summary

### Critical (Fix First)
- Hard-coded CUDA calls
- Error handling tests
- Uncertainty calibration tests

### High Priority
- Missing parameter handling
- Model checkpointing
- Unit tests for individual methods

### Medium Priority
- Mutable default argument
- Training progress monitoring
- No hyperparameter caching
- Fixed learning rate
- Parameterize kernel constraints

### Low Priority
- Dictionary key inconsistency
- Inconsistent training data storage
- Various performance optimizations
- Additional features (sparse GPs, multi-output, etc.)
