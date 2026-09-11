# GPU Migration Guide

Quick guide for enabling GPU acceleration in your heron likelihood computations.

## TL;DR

Replace:
```python
from heron.likelihood import TimeDomainLikelihood
likelihood = TimeDomainLikelihood(data, psd, waveform, detector)
```

With:
```python
from heron.likelihood import TimeDomainLikelihoodGPU
likelihood = TimeDomainLikelihoodGPU(data, psd, waveform, detector)
```

That's it! Automatically uses GPU if available, falls back to CPU otherwise.

## What You Get

- **2.5-7x faster** likelihood evaluations (N≥1024 samples)
- **Automatic** GPU↔CPU management
- **No changes** to your analysis code
- **Same numerical results** (within machine precision)

## Before You Start

### Check GPU Availability
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Install PyTorch with CUDA
If you don't have PyTorch with CUDA:
```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Migration Steps

### 1. Update Likelihood Class

**Before:**
```python
from heron.likelihood import (
    TimeDomainLikelihood,
    TimeDomainLikelihoodModelUncertainty
)

likelihood = TimeDomainLikelihood(data, psd, waveform, detector)
likelihood_uncertain = TimeDomainLikelihoodModelUncertainty(data, psd, waveform, detector)
```

**After:**
```python
from heron.likelihood import (
    TimeDomainLikelihoodGPU,
    TimeDomainLikelihoodModelUncertaintyGPU
)

likelihood = TimeDomainLikelihoodGPU(data, psd, waveform, detector)
likelihood_uncertain = TimeDomainLikelihoodModelUncertaintyGPU(data, psd, waveform, detector)
```

### 2. (Optional) Runtime GPU Detection

For code that runs on both GPU and CPU systems:

```python
import torch
from heron.likelihood import (
    TimeDomainLikelihood,
    TimeDomainLikelihoodGPU
)

# Use GPU if available, otherwise CPU
LikelihoodClass = (
    TimeDomainLikelihoodGPU
    if torch.cuda.is_available()
    else TimeDomainLikelihood
)

likelihood = LikelihoodClass(data, psd, waveform, detector)
```

### 3. No Other Changes Needed!

Everything else stays the same:
```python
# Same parameter estimation workflow
result = bilby.run_sampler(
    likelihood=likelihood.log_likelihood,
    priors=priors,
    sampler='dynesty'
)

# Same likelihood evaluation
log_l = likelihood(parameters)

# Same SNR calculation
snr = likelihood.snr(waveform)
```

## What's Happening Under the Hood

### 1. Waveform Generation (GPyTorch)
```
Parameters → GPyTorch Model (GPU) → Waveform (data on CPU, covariance on GPU)
                                      ↓
                               Covariance stays on GPU (lazy loading)
```

### 2. Likelihood Evaluation
```
Waveform data → GPU
Detector data → GPU
                ↓
        Cholesky solve (GPU, cached)
                ↓
        Result → CPU (scalar)
```

### 3. Lazy Covariance
```
if using TimeDomainLikelihood:
    Covariance never transferred (not needed)  ← 2.5% speedup!

if using TimeDomainLikelihoodModelUncertainty:
    Covariance transferred on first access
```

## Verification

### Check GPU Usage
```python
likelihood = TimeDomainLikelihoodGPU(data, psd)

print(f"Device: {likelihood.device}")  # Should be "cuda"
print(f"Using GPU Cholesky: {likelihood._use_torch_cholesky}")  # Should be True
```

### Verify Numerical Accuracy
```python
import numpy as np

# Create both versions
likelihood_cpu = TimeDomainLikelihood(data, psd, waveform, detector)
likelihood_gpu = TimeDomainLikelihoodGPU(data, psd, waveform, detector)

# Compare results
params = {...}  # Your parameters
ll_cpu = likelihood_cpu(params)
ll_gpu = likelihood_gpu(params)

# Should match within numerical precision
print(f"CPU: {ll_cpu}")
print(f"GPU: {ll_gpu}")
print(f"Difference: {abs(ll_cpu - ll_gpu)}")  # Should be < 1e-10
```

### Benchmark Performance
```python
import time

# Warm up
for _ in range(5):
    _ = likelihood_gpu(params)

# Time GPU
start = time.time()
for _ in range(100):
    _ = likelihood_gpu(params)
gpu_time = (time.time() - start) / 100

# Time CPU (for comparison)
likelihood_cpu = TimeDomainLikelihood(data, psd, waveform, detector)
start = time.time()
for _ in range(100):
    _ = likelihood_cpu(params)
cpu_time = (time.time() - start) / 100

print(f"GPU: {gpu_time*1000:.2f} ms")
print(f"CPU: {cpu_time*1000:.2f} ms")
print(f"Speedup: {cpu_time/gpu_time:.1f}x")
```

## Troubleshooting

### "CUDA out of memory"
Your data is too large for GPU memory. Options:
1. Use smaller data segments
2. Use CPU version for this analysis
3. Use a GPU with more memory

```python
# Fallback to CPU
import heron.likelihood as likelihood_module
likelihood_module.disable_cuda = True
likelihood = TimeDomainLikelihoodGPU(data, psd)  # Will use CPU
```

### "RuntimeError: No CUDA GPUs available"
CUDA not detected. Check:
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.version.cuda)  # CUDA version
```

If False, reinstall PyTorch with CUDA support.

### "Results don't match CPU version"
Numerical differences beyond 1e-10 may indicate an issue. Please report:
1. Sample rate and data length
2. CPU vs GPU results
3. Waveform model used

```python
# Increase verbosity
import logging
logging.basicConfig(level=logging.DEBUG)
```

### GPU Slower Than Expected
For small data (N<512), GPU overhead may dominate. This is normal.

Check if GPU is being used:
```python
import torch
print(f"GPU utilization: {torch.cuda.utilization()}%")
```

## Expected Performance

| Data Size | Speedup | Recommendation |
|-----------|---------|----------------|
| N < 512 | 0.5-1x | Use CPU |
| N = 1024 | ~3x | Use GPU |
| N = 2048 | ~7x | Use GPU ✅ |
| N = 4096 | ~15x | Use GPU ✅ |

## Rollback Plan

If you encounter issues, simply revert to CPU version:

```python
# Change back to original
from heron.likelihood import TimeDomainLikelihood
likelihood = TimeDomainLikelihood(data, psd, waveform, detector)
```

All your analysis code remains unchanged!

## Best Practices

### 1. Always Verify Results
When first migrating, run a small test comparing CPU vs GPU:
```python
assert abs(ll_cpu - ll_gpu) < 1e-10, "Results don't match!"
```

### 2. Profile Your Workload
Time one likelihood evaluation to ensure GPU benefit:
```python
%timeit likelihood_gpu(params)  # In Jupyter
# or
import cProfile
cProfile.run('likelihood_gpu(params)')
```

### 3. Monitor GPU Memory
For production runs:
```python
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### 4. Batch Similar Analyses
GPU shines with repeated evaluations (MCMC/nested sampling). Single evaluations may not show much benefit.

## Support

- **Documentation:** See `docs/gpu_implementation_summary.md`
- **Performance Analysis:** See `docs/gpu_cpu_transfer_optimization.md`
- **Tests:** Run `pytest tests/test_gpu_likelihood.py -v`
- **Issues:** Report at https://github.com/transientlunatic/heron/issues

## Summary

GPU acceleration in heron is designed to be:
- **Easy:** One line change
- **Safe:** Automatic fallback, same results
- **Fast:** 2.5-7x speedup for typical GW data

Just change `TimeDomainLikelihood` → `TimeDomainLikelihoodGPU` and enjoy the speedup!
