# GPyTorch Kernel Analysis for CBC Waveform Surrogates

**Date:** 2026-01-12
**Models:** [ExactGPModel](heron/models/gpytorch.py#L18-L39), [ExactGPModelKeOps](heron/models/gpytorch.py#L42-L66)

## Current Kernel Structure

Both models use the same kernel structure:

```python
covar_module = gpytorch.kernels.ScaleKernel(
    gpytorch.kernels.RBFKernel(active_dims=[0])  # mass_ratio dimension
    * gpytorch.kernels.RBFKernel(active_dims=[1])  # time dimension
)
```

This is a **product of two RBF (Radial Basis Function) kernels** with a scale factor, operating on:
- Dimension 0: Mass ratio (q ∈ [0.1, 1.0])
- Dimension 1: Time (t, with warping applied to negative times)

### Mathematical Form

The kernel is:
```
k(x, x') = σ² · k_mass(q, q') · k_time(t, t')
```

Where each RBF kernel has the form:
```
k_RBF(x, x') = exp(-||x - x'||² / (2ℓ²))
```

This creates a **stationary, isotropic** kernel in each dimension, meaning:
1. **Stationarity**: Covariance depends only on the distance |x - x'|, not absolute positions
2. **Smoothness**: Infinitely differentiable (very smooth functions)
3. **Isotropy**: Same in all directions (spherically symmetric)

## Is This Appropriate for CBC Waveforms?

### ✅ What Works Well

1. **Smoothness in mass ratio**: CBC waveforms vary smoothly with mass ratio, making RBF reasonable for this dimension

2. **Product structure captures independence**: The product kernel assumes mass ratio and time effects are independent, which is physically reasonable after the time warping

3. **Computational efficiency**: RBF kernels have nice properties for GP inference (especially with KeOps for large datasets)

4. **Time warping helps**: The manual time warping (dividing negative times by `warp_scale`) addresses non-stationarity in the time dimension

### ❌ Potential Issues

#### 1. **Time Dimension Non-Stationarity**
CBC waveforms have distinct phases with different behaviors:
- **Inspiral** (t < 0): Slowly increasing amplitude and frequency (adiabatic)
- **Merger** (t ≈ 0): Rapid, highly nonlinear dynamics
- **Ringdown** (t > 0): Exponentially damped oscillations

**Problem**: The RBF kernel assumes the same smoothness everywhere in time. The merger happens much faster than the inspiral, violating stationarity.

**Current mitigation**: Time warping compresses the inspiral, helping somewhat, but it's a crude approximation.

#### 2. **Zero Mean Function**
```python
self.mean_module = gpytorch.means.ZeroMean()
```

**Problem**: CBC waveforms don't oscillate around zero uniformly - they have:
- Near-zero amplitude during early inspiral
- Large amplitude near merger
- Exponential decay during ringdown

**Better approach**: Use a physics-informed mean function (e.g., post-Newtonian approximation for inspiral + exponential decay for ringdown)

#### 3. **No Periodicity Modeling**
CBC waveforms are quasi-periodic (oscillating with increasing frequency).

**Problem**: RBF kernels can't efficiently capture periodic structure. They require many training points to represent oscillations.

**Better approach**: Use a spectral mixture kernel or add a periodic component:
```python
gpytorch.kernels.SpectralMixtureKernel()  # Can learn quasi-periodicity
```

#### 4. **No Multi-Output Structure**
Plus and cross polarizations are trained as **independent models**.

**Problem**: The polarizations are physically related:
```
h₊(t) ∝ (1 + cos²(ι)) F₊ ...
h×(t) ∝ 2cos(ι) F× ...
```

They share the same time evolution and differ mainly in amplitude/phase.

**Better approach**: Use a multi-output GP (e.g., `gpytorch.models.IndependentMultitaskGPModel` or `MultitaskGP`) to capture correlations.

#### 5. **Fixed Lengthscale Constraints**
```python
lengthscale_constraint=gpytorch.constraints.GreaterThan(0.05)  # mass_ratio
lengthscale_constraint=gpytorch.constraints.GreaterThan(0.001)  # time
```

**Problem**: These are hard-coded and may not be optimal across different parameter spaces or waveform families.

**Better approach**: Learn constraints from data or use more flexible priors.

#### 6. **No Handling of Discontinuities**
The transition from inspiral to merger is nearly discontinuous in derivative space (though the waveform itself is continuous).

**Problem**: RBF kernels assume smooth derivatives everywhere.

**Better approach**: Use a kernel that allows for changes in smoothness (e.g., Matérn kernel with learnable smoothness parameter ν).

## Recommended Improvements

### Priority 1: Better Time Modeling

Replace the simple RBF in time with a more sophisticated kernel:

```python
# Option A: Matérn kernel (allows variable smoothness)
time_kernel = gpytorch.kernels.MaternKernel(
    nu=2.5,  # Or make it learnable
    active_dims=[1]
)

# Option B: Spectral Mixture (captures quasi-periodicity)
time_kernel = gpytorch.kernels.SpectralMixtureKernel(
    num_mixtures=3,
    active_dims=[1]
)

# Option C: Additive structure for different phases
time_kernel = (
    gpytorch.kernels.RBFKernel(active_dims=[1]) +  # Smooth envelope
    gpytorch.kernels.SpectralMixtureKernel(num_mixtures=2, active_dims=[1])  # Oscillations
)
```

### Priority 2: Physics-Informed Mean Function

**See detailed analysis**: [mean_function_comparison.md](mean_function_comparison.md)

**Recommendation**: Use an analytical waveform approximant (e.g., IMRPhenomPv2) as the mean function, not a simple PN+ringdown model.

```python
class ApproximantMeanFunction(gpytorch.means.Mean):
    def __init__(self, approximant, fixed_params):
        super().__init__()
        self.approximant = approximant  # e.g., IMRPhenomPv2
        self.fixed_params = fixed_params

    def forward(self, x):
        # x has shape [n_points, 2] with [mass_ratio, time]
        # Evaluate the analytical approximant at these parameter values
        # and return the waveform amplitudes
        # (see mean_function_comparison.md for full implementation)
```

**Why this is better:**
- GP learns corrections to a fast-but-approximate model (standard surrogate approach)
- Captures all physics (inspiral/merger/ringdown) correctly
- Smaller residuals = fewer training points needed
- IMRPhenomPv2 already implemented in Heron and fast (~1-5ms)
- This is literally the point of surrogate modeling!

### Priority 3: Multi-Output GP

```python
class MultiOutputCBCModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y_multi, likelihood):
        # train_y_multi has shape [n_data, 2] for plus and cross
        super().__init__(train_x, train_y_multi, likelihood)

        # Shared base kernel
        self.base_covar = gpytorch.kernels.ScaleKernel(...)

        # Task covariance (plus vs cross)
        self.task_covar = gpytorch.kernels.IndexKernel(
            num_tasks=2,
            rank=1
        )

        self.covar_module = self.base_covar * self.task_covar
```

### Priority 4: Adaptive Lengthscale Constraints

Instead of hard-coded values, learn from the data range:

```python
# Compute statistics from training data
mass_ratio_range = train_x[:, 0].max() - train_x[:, 0].min()
time_range = train_x[:, 1].max() - train_x[:, 1].min()

# Set constraints as fraction of data range
mass_lengthscale_min = 0.05 * mass_ratio_range
time_lengthscale_min = 0.001 * time_range
```

## Alternative Kernel Architectures

### Option 1: Deep Kernel Learning
Use a neural network to learn a warping of the input space, then apply GP:

```python
class DeepKernelGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)

        # Neural network feature extractor
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(2, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 4)
        )

        # GP on learned features
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=4)
        )
```

This could automatically learn the appropriate warping instead of manual time warping.

### Option 2: Additive Structure
Model different waveform phases separately:

```python
covar_module = (
    gpytorch.kernels.RBFKernel() +  # Smooth baseline
    gpytorch.kernels.PeriodicKernel() +  # Oscillations
    gpytorch.kernels.LocalPeriodicKernel()  # Chirping (increasing frequency)
)
```

### Option 3: Sparse Spectrum GP
For computational efficiency with many training points:

```python
class SparseSpectrumGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_inducing=100):
        super().__init__(train_x, train_y, likelihood)

        # Use inducing points to reduce O(n³) to O(nm²)
        inducing_points = train_x[:num_inducing]

        self.base_covar = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.SpectralMixtureKernel(num_mixtures=5)
        )

        self.covar_module = gpytorch.kernels.InducingPointKernel(
            self.base_covar,
            inducing_points=inducing_points,
            likelihood=likelihood
        )
```

## Testing Recommendations

To validate kernel choice, add tests that check:

1. **Frequency content**: Does the GP capture the chirping behavior?
   ```python
   # Compare power spectral density of GP vs true waveform
   psd_true = np.abs(np.fft.fft(true_waveform))**2
   psd_gp = np.abs(np.fft.fft(gp_waveform))**2
   ```

2. **Phase accuracy**: For GW astronomy, phase errors are critical (must be < 0.1 radians)
   ```python
   # Compute phase difference
   phase_error = np.angle(true_waveform) - np.angle(gp_waveform)
   assert np.max(np.abs(phase_error)) < 0.1
   ```

3. **Uncertainty calibration**: Are 95% confidence intervals actually 95% coverage?
   ```python
   # Generate test waveforms not in training set
   # Check if they fall within predicted uncertainties
   ```

4. **Different waveform regimes**: Test on:
   - High mass ratio (q → 1): Less chirping
   - Low mass ratio (q → 0): Long inspiral
   - Different total masses: Affects frequency content

## Conclusion

**Current kernel is adequate but suboptimal.**

The product-of-RBFs kernel will work for interpolation within the training range, but:

1. **Extrapolation will be poor**: RBF kernels revert to the mean (zero) outside training data
2. **Efficiency could be better**: Doesn't exploit the periodic/quasi-periodic structure
3. **Physics is ignored**: No incorporation of known waveform properties
4. **Uncertainty may be miscalibrated**: Especially near merger where things change rapidly

**Recommended next steps:**
1. Add tests for frequency/phase accuracy (Priority 1)
2. Experiment with Matérn kernel for time dimension (Priority 2)
3. Implement physics-informed mean function (Priority 3)
4. Consider multi-output GP for plus/cross together (Priority 4)

The current approach is reasonable for a proof-of-concept but would benefit significantly from physics-informed kernel design for production use.
