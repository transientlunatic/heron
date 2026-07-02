# Deep Kernel Learning for CBC Waveform Surrogates

**Context**: Exploring Deep Kernel Learning (DKL) as an alternative to manual feature engineering (time warping) for GP-based CBC waveform surrogates.

## What is Deep Kernel Learning?

Deep Kernel Learning combines:
1. **Neural Network**: Learns a non-linear feature transformation of the input
2. **Gaussian Process**: Models uncertainty and correlations in the learned feature space

Instead of manually designing features (like time warping), the neural network **learns the right representation** from data.

### Mathematical Framework

**Standard GP**:
```
f(x) ~ GP(m(x), k(x, x'))
```
where `k(x, x')` is computed directly on raw inputs `x`

**Deep Kernel Learning**:
```
z = φ(x; θ)              # Neural network feature extractor
f(x) ~ GP(m(z), k(z, z'))  # GP on learned features
```

The key insight: Let the neural network `φ` learn the warping/transformation that makes the GP's job easier.

---

## Why DKL for CBC Waveforms?

### Problem with Current Approach

Current Heron implementation uses **manual time warping**:
```python
# Hard-coded transformation
times[times < 0] = times[times < 0] / warp_scale  # warp_scale = 2
```

**Issues:**
1. Fixed transformation (not data-driven)
2. Only handles time dimension
3. Single warp parameter (can't adapt to different mass ratios)
4. Linear transformation (doesn't capture non-linear structure)

### What DKL Could Learn

A neural network could automatically learn:

1. **Adaptive time warping** based on mass ratio
   - High q (equal mass): Less warping needed
   - Low q (asymmetric): More aggressive warping

2. **Non-linear transformations**
   - Map merger region to give it more "weight"
   - Compress early inspiral where variation is slow

3. **Joint transformations** of time AND mass ratio
   - Learn that certain (q, t) combinations are "similar"
   - Discover latent structure in the waveform manifold

4. **Physics-aware features**
   - Could learn to represent data in "chirp time" coordinates
   - Or in PN-inspired feature spaces

---

## Architecture Options

### Option 1: Simple Feature Extractor (Recommended Starting Point)

```python
class SimpleDKL(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_dim=4):
        super().__init__(train_x, train_y, likelihood)

        # Input: [mass_ratio, time] -> shape [n, 2]
        # Output: learned features -> shape [n, feature_dim]

        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(2, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, feature_dim)  # Typically 2-8 features
        )

        # Mean function (could be zero or approximant-based)
        self.mean_module = gpytorch.means.ZeroMean()

        # GP kernel operates on learned features
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=feature_dim  # Different lengthscale per feature
            )
        )

    def forward(self, x):
        # Transform inputs through neural network
        features = self.feature_extractor(x)

        # Standard GP operations on learned features
        mean_x = self.mean_module(features)
        covar_x = self.covar_module(features)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
```

**Architecture Details:**
- Input dimension: 2 (mass_ratio, time)
- Hidden layers: 32 neurons (small network - avoid overfitting)
- Output dimension: 4 learned features
- Total parameters: ~1,100 (very lightweight)

**What it learns:**
- The network learns `φ: R² → R⁴` that makes the waveform "straighter" in feature space
- RBF kernel then models smooth correlations in this learned space

---

### Option 2: Spectral Features (Physics-Inspired)

```python
class SpectralDKL(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)

        # Learn to extract frequency-domain features
        self.feature_extractor = torch.nn.Sequential(
            # First layer: Learn time-frequency transforms
            torch.nn.Linear(2, 64),
            torch.nn.Tanh(),  # Smooth activation for oscillations

            # Second layer: Learn spectral features
            torch.nn.Linear(64, 32),
            torch.nn.Tanh(),

            # Output: Compressed spectral representation
            torch.nn.Linear(32, 8)
        )

        self.mean_module = gpytorch.means.ZeroMean()

        # Use a spectral mixture kernel on learned features
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(
            num_mixtures=3,
            ard_num_dims=8
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        mean_x = self.mean_module(features)
        covar_x = self.covar_module(features)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
```

**What this could learn:**
- Chirp mass-like variables
- Instantaneous frequency representations
- PN expansion coordinates

---

### Option 3: Residual Network (For Deeper Models)

```python
class ResidualBlock(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.ReLU(),
            torch.nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.net(x)  # Residual connection

class ResNetDKL(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, hidden_dim=32, n_blocks=3):
        super().__init__(train_x, train_y, likelihood)

        # Initial projection
        self.input_layer = torch.nn.Linear(2, hidden_dim)

        # Residual blocks for deep feature learning
        self.residual_blocks = torch.nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(n_blocks)]
        )

        # Final projection to feature space
        self.output_layer = torch.nn.Linear(hidden_dim, 6)

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=6)
        )

    def forward(self, x):
        features = self.input_layer(x)
        features = torch.relu(features)
        features = self.residual_blocks(features)
        features = self.output_layer(features)

        mean_x = self.mean_module(features)
        covar_x = self.covar_module(features)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
```

**When to use:**
- Larger training datasets (>1000 waveforms)
- More complex parameter spaces (spin, eccentricity)
- Residual connections help with gradient flow in deeper networks

---

### Option 4: Convolutional Features (For Time Series Structure)

```python
class ConvDKL(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)

        # Treat time as a sequence dimension
        # Input will need reshaping: [batch, 1, time_steps]
        self.conv_net = torch.nn.Sequential(
            torch.nn.Conv1d(1, 16, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),

            torch.nn.Conv1d(16, 32, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),

            torch.nn.Flatten(),
            torch.nn.Linear(32 * (time_steps // 4), 8)
        )

        # Also process mass_ratio separately
        self.mass_net = torch.nn.Linear(1, 4)

        # Concatenate features
        # Total features: 8 (from conv) + 4 (from mass) = 12

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=12)
        )

    def forward(self, x):
        # x[:, 0] = mass_ratio, x[:, 1] = time
        mass_features = self.mass_net(x[:, 0:1])

        # Reshape time for conv: needs [batch, channels, time]
        time_reshaped = x[:, 1:2].unsqueeze(1)  # [batch, 1, 1]
        time_features = self.conv_net(time_reshaped)

        features = torch.cat([mass_features, time_features], dim=-1)

        mean_x = self.mean_module(features)
        covar_x = self.covar_module(features)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
```

**Note**: This is more complex and may be overkill for current Heron use case, but could be useful if expanding to multi-dimensional parameter spaces.

---

## Training Strategy

### Joint Training (Recommended)

Train the neural network and GP hyperparameters **together** by maximizing the marginal likelihood:

```python
def train_dkl_model(model, likelihood, train_x, train_y, n_iter=500):
    model.train()
    likelihood.train()

    # Optimizer for ALL parameters (NN weights + GP hyperparams)
    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': 0.01},  # NN params
        {'params': model.covar_module.parameters(), 'lr': 0.1},         # GP kernel params
        {'params': likelihood.parameters(), 'lr': 0.1}                  # Likelihood params
    ])

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f'Iter {i}/{n_iter} - Loss: {loss.item():.3f}')

    model.eval()
    likelihood.eval()
```

**Key points:**
- Use different learning rates for NN (0.01) vs GP params (0.1)
- Neural network learns slower to avoid overfitting
- Total training time: ~2-5 minutes for 100 training waveforms

### Two-Stage Training (Alternative)

1. **Stage 1**: Pre-train feature extractor with supervised loss
2. **Stage 2**: Freeze features, train GP
3. **Stage 3**: Fine-tune end-to-end

This can be more stable but less flexible.

---

## Advantages of DKL for Heron

### 1. **Learns Optimal Warping Automatically**
Instead of `warp_scale = 2`, the network learns:
```python
# Pseudo-code for what the network might learn:
def learned_warp(mass_ratio, time):
    if time < -0.3:  # Early inspiral
        warp = 5.0 * (1 + mass_ratio)
    elif time < 0:  # Late inspiral
        warp = 2.0
    else:  # Merger + ringdown
        warp = 1.0
    return time / warp
```

### 2. **Handles Multiple Parameters Gracefully**
When you add spin parameters later:
```python
# Input: [mass_ratio, chi1, chi2, time]
# Network learns how spin affects time evolution automatically
```

No need to manually design warping for each new parameter!

### 3. **Better Extrapolation**
By learning a meaningful feature space, the GP can extrapolate better than on raw parameters.

### 4. **Uncertainty Quantification**
Still get full GP uncertainties, but in a better-learned space.

### 5. **Interpretability** (with analysis)
Can visualize learned features to understand what the network discovered:
```python
# After training, project all training data to feature space
features = model.feature_extractor(train_x)

# Visualize in 2D using PCA or t-SNE
from sklearn.manifold import TSNE
features_2d = TSNE(n_components=2).fit_transform(features.detach().cpu())

# Color by mass ratio or waveform phase
plt.scatter(features_2d[:, 0], features_2d[:, 1], c=train_x[:, 0])
plt.colorbar(label='Mass Ratio')
plt.title('Learned Feature Space')
```

---

## Disadvantages / Challenges

### 1. **More Parameters to Train**
- Simple DKL: ~1,100 NN parameters + GP hyperparameters
- Risk of overfitting with small datasets (<100 waveforms)
- **Mitigation**: Use regularization (dropout, weight decay)

### 2. **Less Interpretable**
- Manual warping: "We compress inspiral by factor 2"
- DKL: "The network learned something... but what?"
- **Mitigation**: Analyze learned features post-hoc

### 3. **Slower Training**
- Must train both NN and GP jointly
- Current: ~1 min for GP only
- DKL: ~5 min for GP + small NN
- **Mitigation**: Still much faster than generating training waveforms

### 4. **Hyperparameter Tuning**
- Must choose: network depth, width, feature dimension, learning rates
- **Mitigation**: Start with simple architecture (Option 1), tune only if needed

### 5. **Not Obviously Better for Simple Cases**
- For 2D parameter space (mass_ratio, time), manual warping works fine
- DKL shines when you have:
  - Higher dimensional inputs (4+ parameters)
  - Complex non-linear structure
  - Large training datasets

---

## When to Use DKL vs Manual Warping

### Use Manual Warping If:
- ✅ Low-dimensional parameter space (2-3 params)
- ✅ You understand the physics well enough to design good features
- ✅ Small training dataset (<100 waveforms)
- ✅ Need maximum interpretability
- ✅ Computational budget is tight

### Use DKL If:
- ✅ High-dimensional parameter space (4+ params: masses, spins, eccentricity)
- ✅ Don't know optimal warping/transformation
- ✅ Large training dataset (>200 waveforms)
- ✅ Willing to sacrifice some interpretability for performance
- ✅ Want the model to discover structure automatically

---

## Practical Implementation for Heron

### Recommended Approach: Start Simple, Then Add Complexity

#### Phase 1: Baseline (Current)
```python
# Manual warping + RBF kernel
warp_scale = 2
# Works but suboptimal
```

#### Phase 2: Add Analytical Mean
```python
# IMRPhenomPv2 mean + RBF kernel
mean_function = ApproximantMeanFunction(IMRPhenomPv2())
# Big improvement, still interpretable
```

#### Phase 3: Replace Warping with DKL
```python
# IMRPhenomPv2 mean + DKL feature learning + RBF kernel
model = SimpleDKL(train_x, train_y_residuals, likelihood, feature_dim=4)
# Learn optimal warping from data
```

#### Phase 4: Full DKL + Spectral Kernel
```python
# Approximant mean + Spectral DKL
model = SpectralDKL(train_x, train_y_residuals, likelihood)
# Capture both warping and periodicity
```

### Concrete Example Code

```python
class HeronDKLSurrogate(WaveformSurrogate, GPyTorchSurrogate):
    def __init__(
        self,
        train_x_plus,
        train_x_cross,
        train_y_plus,
        train_y_cross,
        total_mass,
        distance,
        mean_approximant=None,
        use_dkl=True,
        feature_dim=4,
        training=400,
    ):
        self.device = device
        self.use_dkl = use_dkl

        # ... (existing setup code) ...

        # Create DKL models instead of standard GP
        if use_dkl:
            self.models["plus"] = SimpleDKL(
                self.train_x_plus,
                self.train_y_plus,
                gpytorch.likelihoods.GaussianLikelihood(),
                feature_dim=feature_dim
            ).to(self.device)

            self.models["cross"] = SimpleDKL(
                self.train_x_cross,
                self.train_y_cross,
                gpytorch.likelihoods.GaussianLikelihood(),
                feature_dim=feature_dim
            ).to(self.device)
        else:
            # Fall back to standard GP
            self.models["plus"] = ExactGPModelKeOps(...)
            self.models["cross"] = ExactGPModelKeOps(...)

        self.train(training)
```

---

## Visualization: What DKL Might Learn

Imagine the original parameter space:

```
Time Axis (linear)
|
|   Inspiral --------- very slow changes
|   Inspiral --------- slow changes
|   Inspiral --------- moderate changes
|   Late Inspiral ---- FAST changes
|   Merger ----------- VERY FAST changes (lots of structure)
|   Ringdown --------- exponential decay
|
+---> Mass Ratio Axis
```

After DKL transformation:

```
Learned Feature Space
|
|   All waveform "phases" are more evenly spaced
|   Inspiral compressed (was 90% of time, now 50%)
|   Merger expanded (was 5% of time, now 25%)
|   Ringdown compressed
|
|   Mass ratio might be transformed to chirp-mass-like variable
|
+---> Learned features capture what GP needs to model smoothly
```

The GP kernel can now use a simple RBF in this space instead of needing complex kernels in the original space.

---

## Testing DKL Performance

Add tests to compare:

1. **Accuracy**: Overlap with true waveforms
2. **Uncertainty calibration**: Are confidence intervals correct?
3. **Training time**: How much longer?
4. **Inference speed**: Should be similar to standard GP
5. **Data efficiency**: How many training waveforms needed?

```python
def test_dkl_vs_manual_warping():
    # Train both models on same data
    model_manual = HeronNonSpinningApproximant(train_x, train_y, warp_scale=2)
    model_dkl = HeronDKLSurrogate(train_x, train_y, use_dkl=True)

    # Test on validation waveforms
    test_overlaps_manual = []
    test_overlaps_dkl = []

    for params in validation_parameters:
        wf_true = SEOBNRv4().time_domain(params)
        wf_manual = model_manual.time_domain(params)
        wf_dkl = model_dkl.time_domain(params)

        test_overlaps_manual.append(overlap(wf_true, wf_manual))
        test_overlaps_dkl.append(overlap(wf_true, wf_dkl))

    print(f"Manual warping mean overlap: {np.mean(test_overlaps_manual):.4f}")
    print(f"DKL mean overlap: {np.mean(test_overlaps_dkl):.4f}")
```

---

## Conclusion

**Deep Kernel Learning is a natural evolution** for Heron's GP surrogates:

### Recommended Roadmap

1. ✅ **First**: Implement analytical approximant mean function
   - Biggest bang for buck
   - Still interpretable
   - Standard practice

2. ✅ **Second**: Experiment with DKL on current 2D problem
   - Use SimpleDKL architecture (Option 1)
   - Compare against manual warping
   - If similar or better → proceed

3. ✅ **Third**: Apply to higher-dimensional problems
   - Add spin parameters
   - DKL will shine here
   - Manual feature engineering becomes intractable

4. ✅ **Fourth**: Combine DKL + Spectral kernels
   - Let NN learn warping
   - Let spectral kernel capture periodicity
   - Best of both worlds

### Bottom Line

For **current Heron** (2D parameter space):
- Manual warping is fine, DKL is optional
- **Priority 1**: Add analytical mean function
- **Priority 2**: Try simple DKL, see if it helps

For **future Heron** (4+ dimensional):
- DKL becomes essential
- Too many dimensions for manual feature engineering
- Let the network discover structure

**Start simple (analytical mean), add complexity (DKL) only when needed.**
