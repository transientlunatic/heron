# KeOps Improvement Opportunities for Heron

**Date:** 2026-01-13
**Status:** To be revisited

## Context

The heron codebase already uses KeOps in `ExactGPModelKeOps` for memory-efficient GPU kernel computations during GP training. This document outlines additional opportunities to leverage KeOps for further performance improvements.

## Current KeOps Usage

- **Location**: [heron/models/gpytorch.py:42-66](../heron/models/gpytorch.py#L42-L66)
- **Purpose**: Memory-efficient RBF kernel computations in GPyTorch models
- **Benefit**: O(n) vs O(n²) memory for kernel matrices

## Potential Improvements

### 1. KeOps in Likelihood Computations (High Impact)

**Current state**: Likelihood functions in [heron/likelihood.py](../heron/likelihood.py) use standard PyTorch operations for quadratic forms `residual.T @ C^{-1} @ residual`.

**Opportunity**: Use KeOps LazyTensors for covariance operations to reduce memory by 50-70% for N > 2048 samples.

```python
from pykeops.torch import LazyTensor

# Create lazy representation of covariance
C_keops = LazyTensor(C_scaled)
# Compute weighted residual without materializing full matrix
weighted_residual = (residual * C_keops.solve(residual)).sum()
```

**Expected benefit**: 50-70% memory reduction for large N

---

### 2. Multi-Output GP with KeOps (Medium Impact)

**Current state**: Plus/cross polarizations trained independently (see [gpytorch_kernel_analysis.md](gpytorch_kernel_analysis.md#L169-L185)).

**Opportunity**: Use KeOps for efficient multi-output kernel computations to capture physical correlations between polarizations.

**Expected benefit**: Better uncertainty quantification, minimal memory overhead

---

### 3. Custom Physics-Informed Kernels (High Impact)

**Current state**: Product of RBF kernels (see [gpytorch_kernel_analysis.md](gpytorch_kernel_analysis.md#L52-L61) for limitations).

**Opportunity**: Implement custom KeOps kernels that exploit CBC waveform structure:
- Quasi-periodic kernels for chirping behavior
- Phase-sensitive kernels (GW astronomy requires <0.1 radian accuracy)
- Regime-specific kernels for inspiral/merger/ringdown

**Example: Custom chirping kernel**
```python
def chirping_kernel(x, y, params):
    """Custom kernel for chirping waveforms"""
    # Frequency increases with time: f(t) ∝ t^(-3/8) for inspiral
    mass_diff = ((x[..., 0] - y[..., 0]) / params[0])**2
    time_diff = x[..., 1] - y[..., 1]
    phase_term = (time_diff / params[1]) * (1 + params[2] * time_diff)
    return (-0.5 * (mass_diff + phase_term**2)).exp()
```

**Expected benefit**: Better waveform interpolation, especially near merger

---

### 4. Efficient Log-Determinant Computation (Medium Impact)

**Current state**: Standard `np.linalg.slogdet(K)` in [likelihood.py:31-32](../heron/likelihood.py#L31-L32).

**Opportunity**: For structured kernels, use GPyTorch's lazy evaluation with KeOps backend to compute log-determinants without materializing matrices.

```python
from gpytorch.lazy import LazyTensor
cov_lazy = covar_module(x)  # Lazy representation
logdet = cov_lazy.logdet()  # Computed efficiently via Lanczos
```

**Expected benefit**: Faster computation for N > 2048

---

### 5. Batched Parameter Evaluation (High Impact)

**Current state**: Likelihood evaluations during parameter estimation (e.g., nested sampling) are sequential.

**Opportunity**: Leverage KeOps for batched operations to evaluate GP at multiple parameter values simultaneously.

```python
# Evaluate GP at multiple parameter values simultaneously
params_batch = torch.stack([torch.tensor([q1, t]), torch.tensor([q2, t]), ...])
with gpytorch.settings.fast_pred_var():
    predictions = model(params_batch)  # Batched KeOps kernel evaluation
```

**Expected benefit**: 5-10x speedup when evaluating 100+ parameter samples

---

### 6. Sparse/Inducing Point Methods (Medium Impact)

**Current state**: Exact GP inference for all training data.

**Opportunity**: For very large training sets (N > 10,000), use sparse GP methods with KeOps to scale to 10x more training data.

```python
from gpytorch.models import ApproximateGP
from gpytorch.variational import VariationalStrategy

class SparseKeOpsGP(ApproximateGP):
    def __init__(self, inducing_points):
        # Use KeOps kernels with inducing points
        self.covar_module = gpytorch.kernels.keops.RBFKernel()
```

**Expected benefit**: -80% memory, enables much larger training sets

---

## Performance Summary

| Improvement | Memory Impact | Speed Impact | Implementation Effort |
|-------------|---------------|--------------|----------------------|
| 1. KeOps likelihood | -50% | +20% | Medium |
| 2. Multi-output GP | -30% | +40% | High |
| 3. Custom CBC kernels | 0% | +100%* | High |
| 4. Lazy log-det | -20% | +30% | Low |
| 5. Batched evaluation | 0% | +5-10x | Low |
| 6. Sparse methods | -80% | -20%† | High |

*Better accuracy, not just speed
†Slower per sample but enables much larger training sets

---

## Recommendations by Priority

### Immediate (Low-hanging fruit)
1. **Batched likelihood evaluation** - Add batch support to `__call__` method
2. **Lazy log-determinant** - Use GPyTorch's lazy tensors with KeOps backend

### Medium-term (High impact)
3. **Multi-output KeOps GP** - Train plus/cross together with correlations
4. **Custom chirping kernel** - Physics-informed kernel for better extrapolation

### Long-term (Research)
5. **Sparse GP with KeOps** - Scale to very large training sets
6. **Deep kernel learning** - Let NN learn optimal warping instead of manual `warp_scale`

---

## Key Insight

The codebase is already well-optimized with KeOps for **training**. The biggest gains would come from:

1. Extending KeOps to the **likelihood evaluation** (not just waveform generation)
2. **Batched operations** for parameter estimation
3. **Physics-informed kernels** that exploit CBC waveform structure

---

## Related Documentation

- [GPU Implementation Summary](gpu_implementation_summary.md)
- [GPyTorch Kernel Analysis](gpytorch_kernel_analysis.md)
- [Mean Function Comparison](mean_function_comparison.md)
- [GPU Migration Guide](GPU_MIGRATION_GUIDE.md)

---

## References

- KeOps documentation: https://www.kernel-operations.io/
- GPyTorch KeOps integration: https://docs.gpytorch.ai/en/stable/examples/02_Scalable_Exact_GPs/index.html
