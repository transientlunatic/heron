# Heron Examples

This directory contains example scripts demonstrating how to use heron with various tools and frameworks.

## Bilby Integration Example

**File:** `bilby_integration_example.py`

This example shows how to use heron's Gaussian process-based waveform models with the bilby gravitational wave inference library. The key features are:

- Using `HeronWaveformGenerator` to create bilby-compatible waveforms from heron models
- Using `HeronGravitationalWaveTransient` likelihood that properly handles waveform uncertainty
- Incorporating model uncertainty from Gaussian process regression into parameter estimation

### Running the Example

The example requires both heron and bilby to be installed:

```bash
pip install heron-model bilby
```

Then you can run the example:

```bash
python bilby_integration_example.py
```

Note: You'll need to uncomment the `main()` or `compare_with_without_uncertainty()` function call at the bottom of the script to actually run an analysis.

### Key Concepts

The example demonstrates two main components:

1. **HeronWaveformGenerator**: A bilby-compatible waveform generator that wraps heron waveform models and propagates uncertainty information.

2. **HeronGravitationalWaveTransient**: A bilby likelihood that incorporates waveform model uncertainty into the likelihood calculation by adding the model covariance to the detector noise covariance.

### Model Uncertainty

When `include_model_uncertainty=True` (the default), the likelihood accounts for uncertainty in the waveform predictions that comes from the Gaussian process regression. This leads to more conservative parameter estimates that properly reflect the model's confidence in different regions of parameter space.

The uncertainty is propagated through:
- Antenna response projection (F+ and Fx)
- FFT operations for frequency domain analysis
- Matched filtering calculations

### Comparison Example

The `compare_with_without_uncertainty()` function shows how to compare results with and without model uncertainty, demonstrating the impact of including this information in the analysis.
