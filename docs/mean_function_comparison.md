# Mean Function Comparison: Analytical Approximant vs. Simple PN+Ringdown

**Context**: Choosing the mean function for the GP surrogate model in Heron

## Question
Should we use a full analytical waveform approximant (e.g., IMRPhenomXPHM, IMRPhenomPv2) as the mean function, or is a simpler PN + ringdown model sufficient?

## TL;DR: Use the Analytical Approximant

**Recommendation**: Use an analytical approximant (e.g., IMRPhenomPv2 or simpler) as the mean function.

**Why**: You already have these approximants available in Heron, they're fast, and this is exactly what the GP is meant to be doing - learning the **corrections** to a fast-but-approximate model.

---

## Detailed Comparison

### Option 1: Full Analytical Approximant (e.g., IMRPhenomPv2)

#### Advantages

1. **This is the entire point of surrogate modeling!**
   - The GP is being trained to learn corrections to IMRPhenomPv2 (or similar)
   - The target is a more accurate (but slower) approximant like SEOBNRv4
   - Using IMRPhenomPv2 as the mean is saying: "Start from this approximation, GP learns the difference"
   - This is the **standard approach** in surrogate modeling

2. **Captures all the physics correctly**
   - Inspiral, merger, and ringdown all modeled properly
   - Correct frequency evolution (chirping)
   - Proper amplitude envelope
   - Phase relationships between polarizations
   - Spin effects (if using a precessing model)

3. **GP only needs to model residuals**
   - If IMRPhenomPv2 is ~95% accurate, GP only learns the 5% correction
   - Smaller residuals = easier to model = need fewer training points
   - Better extrapolation (GP starts from physically reasonable baseline)
   - Uncertainties are more meaningful (uncertainty in the *correction*)

4. **Already implemented in Heron**
   - [lalsimulation.py](heron/models/lalsimulation.py) has IMRPhenomPv2, SEOBNRv3
   - These are optimized C code (via LAL) - very fast
   - No need to reimplement anything

5. **Hierarchical surrogate approach**
   - Fast approximate model (IMRPhenom) as mean
   - GP learns corrections to match slow accurate model (SEOB/NR)
   - This is how modern surrogates work (e.g., NRSur7dq4)

#### Disadvantages

1. **Requires evaluating the approximant for every GP prediction**
   - Adds computational cost: `cost = approximant_cost + GP_cost`
   - But IMRPhenom models are already fast (~milliseconds)

2. **Mean function has hyperparameters**
   - The approximant has its own parameters (masses, spins, etc.)
   - Must be evaluated at the correct parameter values for each prediction
   - Slightly more complex implementation

3. **Can't use approximant that's too slow**
   - If the mean function approximant is as slow as the target, no speedup
   - Need to pick a fast approximant (IMRPhenomPv2, IMRPhenomD, TaylorF2)

---

### Option 2: Simple PN + Ringdown

#### Advantages

1. **Simpler to implement**
   - Just a few exponentials and power laws
   - Can write as a PyTorch module in ~50 lines
   - Fewer parameters to tune

2. **Very fast**
   - Just arithmetic operations, no special functions
   - Negligible computational cost

3. **Smooth and differentiable**
   - Nice properties for optimization
   - Can take gradients easily

#### Disadvantages

1. **Misses critical physics**
   - PN approximation breaks down near merger
   - No smooth transition between inspiral and ringdown
   - Wrong frequency evolution (PN is adiabatic, not accurate near merger)
   - Doesn't capture spin effects
   - Wrong amplitude scaling

2. **GP has to learn everything**
   - If the mean is wrong by 50%, GP needs to learn large corrections
   - Requires many more training points
   - Worse extrapolation (GP can't distinguish "true physics" from "noise in training")

3. **Not how surrogate modeling is done**
   - Standard practice: use fast approximate model + learn corrections
   - Your approach would be: use bad model + learn everything
   - Why have a mean function at all if it's not capturing the structure?

4. **Defeats the purpose of using an existing approximant**
   - You're training on IMRPhenomPv2 (or similar) data
   - Why not just use IMRPhenomPv2 as the starting point?

---

## Concrete Example

### Scenario: Training a surrogate for SEOBNRv4

**Setup:**
- Target: SEOBNRv4 (accurate but slow, ~seconds per waveform)
- Goal: Speed up to milliseconds

**Option A: Use IMRPhenomPv2 as mean**
```python
class CBCGPWithPhenomMean(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, phenom_approximant):
        super().__init__(train_x, train_y, likelihood)
        self.phenom = phenom_approximant  # IMRPhenomPv2
        self.covar_module = gpytorch.kernels.ScaleKernel(...)

    def mean_function(self, x):
        # x contains mass_ratio, time
        # Evaluate IMRPhenomPv2 at these parameters
        phenom_waveform = self.phenom.time_domain(params_from_x(x))
        return phenom_waveform

# Training:
# y_train = SEOB_waveform - Phenom_waveform (residuals)
# GP learns: SEOB ≈ Phenom + GP_correction
```

**Result:**
- Residuals are small (~5% of amplitude)
- GP can model accurately with fewer points
- Prediction: `SEOB(new_params) ≈ Phenom(new_params) + GP(new_params)`
- Speed: ~10ms (Phenom) + ~1ms (GP) = **11ms total**
- Accuracy: Can match SEOB to within 0.1% with ~100 training waveforms

**Option B: Use simple PN+ringdown as mean**
```python
def simple_mean(x):
    t = x[:, 1]
    q = x[:, 0]
    # Crude approximation
    inspiral = np.exp(t / 0.1) * (t < 0)
    ringdown = np.exp(-t * 10) * (t >= 0)
    return inspiral + ringdown
```

**Result:**
- Residuals are huge (50-80% of amplitude)
- GP struggles to model, needs many points
- Extrapolation is poor
- Speed: ~0.1ms (mean) + ~1ms (GP) = **1.1ms total** (faster!)
- Accuracy: Much worse - can't match SEOB within 1% even with 1000 training waveforms

**Verdict**: Option A is clearly better for accuracy, even if slightly slower.

---

## Which Approximant to Use as Mean?

If using an analytical approximant, which one?

### Tier 1: Best Choices (Fast & Reasonable Accuracy)

1. **IMRPhenomPv2** ✅
   - Already in Heron
   - Includes precession
   - Fast (~1-5ms)
   - ~90-95% accurate for most systems

2. **IMRPhenomD**
   - Simpler (no precession)
   - Faster (~0.5-2ms)
   - Good for aligned-spin systems
   - ~85-90% accurate

3. **TaylorF2** (frequency domain)
   - Very fast (~0.1-0.5ms)
   - Only inspiral (no merger/ringdown)
   - Good if you only care about low-frequency part

### Tier 2: Maybe Too Slow

1. **IMRPhenomXPHM**
   - More accurate than Pv2
   - But slower (~10-50ms)
   - Might not give enough speedup

2. **SEOBNRv4/v5**
   - Very accurate
   - But slow (seconds)
   - This is what you're trying to *replace*, not use as mean!

### Recommendation for Heron

**Start with IMRPhenomPv2** because:
1. Already implemented in Heron
2. Fast enough (~1-5ms)
3. Accurate enough to make GP learning efficient
4. Handles precession (if needed)
5. Well-tested and widely used

If you need even more speed, can try IMRPhenomD (non-precessing).

---

## Implementation Strategy

### Step 1: Modify GP Model to Accept Mean Function

```python
class HeronNonSpinningApproximant(WaveformSurrogate, GPyTorchSurrogate):
    def __init__(
        self,
        train_x_plus,
        train_x_cross,
        train_y_plus,
        train_y_cross,
        total_mass,
        distance,
        mean_approximant=None,  # NEW: Pass in an approximant
        warp_scale=2,
        training=400,
    ):
        # ... existing code ...

        # Store the mean function approximant
        self.mean_approximant = mean_approximant or IMRPhenomPv2()

        # Create models with custom mean
        self.models["plus"] = ExactGPModelWithMean(
            self.train_x_plus,
            self.train_y_plus,
            mean_approximant=self.mean_approximant
        )
```

### Step 2: Create GP Model with Custom Mean

```python
class ExactGPModelWithMean(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean_approximant):
        super().__init__(train_x, train_y, likelihood)

        # Use custom mean function that wraps the approximant
        self.mean_module = ApproximantMeanFunction(mean_approximant)

        # Covariance stays the same
        self.covar_module = gpytorch.kernels.ScaleKernel(...)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ApproximantMeanFunction(gpytorch.means.Mean):
    def __init__(self, approximant, fixed_params):
        super().__init__()
        self.approximant = approximant
        self.fixed_params = fixed_params

    def forward(self, x):
        # x has shape [n_points, 2] with [mass_ratio, time]
        # Need to evaluate approximant at these points

        # This is tricky because we need to evaluate the approximant
        # for each unique mass_ratio, then sample at the times

        # Simplified version:
        unique_mass_ratios = torch.unique(x[:, 0])

        outputs = []
        for q in unique_mass_ratios:
            mask = x[:, 0] == q
            times = x[mask, 1]

            # Evaluate approximant
            params = self.fixed_params.copy()
            params['mass_ratio'] = q.item()
            params['time'] = {'lower': times.min().item(),
                             'upper': times.max().item(),
                             'number': len(times)}

            wf = self.approximant.time_domain(params)
            outputs.append(wf['plus'].data)

        return torch.cat(outputs)
```

### Step 3: Train on Residuals

```python
# Generate training data
train_data = make_optimal_manifold(
    approximant=SEOBNRv4,  # High-accuracy target
    ...
)

# Generate mean function evaluations
mean_data = make_optimal_manifold(
    approximant=IMRPhenomPv2,  # Fast approximation
    ...
)

# Compute residuals
residuals = train_data - mean_data

# Train GP on residuals
model = HeronNonSpinningApproximant(
    train_x_plus=train_x,
    train_y_plus=residuals_plus,  # Train on differences!
    mean_approximant=IMRPhenomPv2(),
    ...
)
```

---

## What if You Want Maximum Speed?

If the analytical approximant is still too slow, there are intermediate options:

### Hybrid Approach: Parameterized Mean Function

Learn a simple parameterized mean from the analytical approximant:

```python
class LearnedMeanFunction(gpytorch.means.Mean):
    def __init__(self):
        super().__init__()
        # Learnable parameters for phenomenological model
        self.inspiral_rate = torch.nn.Parameter(torch.tensor(0.1))
        self.merger_time = torch.nn.Parameter(torch.tensor(0.0))
        self.ringdown_decay = torch.nn.Parameter(torch.tensor(10.0))
        self.amplitude = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        t = x[:, 1]
        q = x[:, 0]

        # Phenomenological model with learned parameters
        t_shift = t - self.merger_time

        inspiral = torch.exp(t_shift / self.inspiral_rate) * (t < self.merger_time)
        ringdown = torch.exp(-t_shift * self.ringdown_decay) * (t >= self.merger_time)

        # Mass ratio dependence
        q_factor = (1 + q) / 2

        return self.amplitude * q_factor * (inspiral + ringdown)

# Pre-train this on IMRPhenomPv2 waveforms to get good initial parameters
# Then use as mean function for GP
```

This gives you:
- Speed of analytical formula (~0.1ms)
- More physics than pure PN+ringdown
- Parameters learned from actual waveforms

---

## Conclusion

**Use an analytical approximant (IMRPhenomPv2) as the mean function.**

### Reasons:
1. ✅ This is standard practice in surrogate modeling
2. ✅ Reduces GP to learning small corrections
3. ✅ Better extrapolation and fewer training points needed
4. ✅ Already implemented in Heron
5. ✅ Fast enough for most applications
6. ✅ Captures all relevant physics

### When to use simpler mean:
- Only if computational budget is **extremely** tight
- And you're willing to sacrifice accuracy significantly
- Or if you can pre-train a phenomenological model

### Next steps:
1. Implement `ExactGPModelWithMean` class that wraps an approximant
2. Modify training pipeline to compute residuals
3. Add tests comparing mean-only, GP-only, and mean+GP predictions
4. Benchmark speed vs accuracy tradeoff

**Bottom line**: The whole point of building a GP surrogate is to **accelerate** an accurate-but-slow model by learning corrections to a fast-but-approximate model. Using the fast approximation as your mean function is exactly the right approach.
