# Bilby Integration for Heron

## Overview

This implementation adds full bilby compatibility to heron, allowing heron's Gaussian process-based waveform models with uncertainty to be used in standard bilby gravitational wave parameter estimation analyses.

## Components Implemented

### 1. HeronWaveformGenerator

**File:** `heron/bilby.py`

A bilby-compatible waveform generator that wraps heron waveform models.

**Key Features:**
- Inherits from `bilby.gw.waveform_generator.WaveformGenerator`
- Automatically handles time-domain waveform generation from heron models
- Propagates uncertainty information (covariance and variance) from the Gaussian process
- Compatible with all bilby interferometer and analysis tools

**Usage:**
```python
from heron.bilby import HeronWaveformGenerator
from heron.models.gpytorch import HeronNonSpinningApproximant

heron_model = HeronNonSpinningApproximant()
wfg = HeronWaveformGenerator(
    duration=4.0,
    sampling_frequency=2048,
    heron_waveform=heron_model
)
```

### 2. HeronGravitationalWaveTransient

**File:** `heron/bilby.py`

A bilby likelihood that properly incorporates waveform model uncertainty.

**Key Features:**
- Inherits from `bilby.gw.likelihood.GravitationalWaveTransient`
- Includes model uncertainty in the likelihood calculation
- Handles FFT operations with proper normalization
- Integrates antenna response functions correctly
- Can be toggled between uncertainty-aware and standard modes

**Scientific Approach:**

The likelihood with uncertainty is computed as:

```
log L = -1/2 * (d - h)^T (C_n + C_h)^{-1} (d - h) - 1/2 * log|C_n + C_h| - N/2 * log(2π)
```

Where:
- `d` is the observed data
- `h` is the predicted waveform
- `C_n` is the detector noise covariance (from PSD)
- `C_h` is the waveform model uncertainty covariance (from GP)
- `N` is the number of data points

For computational efficiency, the implementation uses an approximation that treats the model uncertainty as an effective increase in noise power, avoiding full matrix inversion for large covariance matrices.

**Usage:**
```python
from heron.bilby import HeronGravitationalWaveTransient

likelihood = HeronGravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=wfg,
    include_model_uncertainty=True  # Enable uncertainty handling
)
```

## Uncertainty Handling

### Waveform Uncertainty Propagation

The implementation correctly propagates uncertainty through several steps:

1. **GP Prediction**: The heron waveform model provides both mean waveform and uncertainty (covariance matrix)

2. **Antenna Response Projection**: When projecting waveforms onto detectors:
   ```
   Var(F+ * h+ + Fx * hx) = F+^2 * Var(h+) + Fx^2 * Var(hx)
   ```

3. **FFT Operations**: Proper normalization is maintained through Fourier transforms

4. **Likelihood Calculation**: Model uncertainty is added to detector noise in the likelihood

### Approximations and Limitations

- The current implementation uses an approximate treatment of model uncertainty for computational efficiency
- Full covariance inversion is avoided by treating uncertainty as diagonal in frequency domain
- For very large uncertainty, results should be validated against full time-domain calculation
- The approximation is most accurate when model uncertainty is small compared to detector noise

## Testing

### Unit Tests

**File:** `tests/test_bilby.py`

Comprehensive test suite including:

1. **Waveform Generation Tests**
   - Initialization and configuration
   - Time-domain waveform generation
   - Uncertainty propagation
   - Waveform shape validation

2. **Scientific Accuracy Tests**
   - Waveform normalization
   - Finite values check
   - Positive semi-definite covariance validation

3. **Placeholder Tests**
   - Full likelihood evaluation (requires bilby installation)
   - SNR calculations
   - Comparison with/without uncertainty

**Running Tests:**
```bash
python -m unittest tests.test_bilby
```

## Examples

### Basic Example

**File:** `examples/bilby_integration_example.py`

Demonstrates:
- Setting up a complete parameter estimation with heron waveforms
- Configuring priors
- Running sampler with uncertainty
- Comparing results with/without model uncertainty

**Running:**
```bash
python examples/bilby_integration_example.py
```

### Documentation

**File:** `examples/README.md`

Explains:
- How to use the bilby integration
- Key concepts in model uncertainty
- Comparison examples

## Documentation Standards

All code follows numpydoc documentation standards with:

- Complete parameter descriptions
- Return value documentation  
- Detailed notes on scientific methods
- Usage examples
- References to relevant literature

Example from `HeronGravitationalWaveTransient`:
```python
def __init__(self, interferometers, waveform_generator, ...):
    """
    Initialize the heron likelihood.
    
    Parameters
    ----------
    interferometers : list
        List of bilby Interferometer objects containing the data.
    waveform_generator : HeronWaveformGenerator
        A heron waveform generator instance.
    ...
    
    Notes
    -----
    The likelihood is computed as:
    ...
    
    Examples
    --------
    >>> likelihood = HeronGravitationalWaveTransient(...)
    """
```

## Integration Points

### With Bilby

The implementation is designed to be a drop-in replacement for bilby's standard likelihood:

```python
# Standard bilby
from bilby.gw.likelihood import GravitationalWaveTransient
likelihood = GravitationalWaveTransient(ifos, wfg)

# Heron with uncertainty
from heron.bilby import HeronGravitationalWaveTransient
likelihood = HeronGravitationalWaveTransient(ifos, wfg, include_model_uncertainty=True)
```

### With Heron

Works with any heron waveform model:
- `HeronNonSpinningApproximant` (GPyTorch-based)
- Custom trained models
- Testing models (for validation)

## Future Improvements

Potential enhancements for future development:

1. **Full Covariance Treatment**: Implement exact likelihood with full (C_n + C_h)^{-1} inversion using efficient algorithms (e.g., Woodbury identity, Cholesky decomposition)

2. **Frequency-Domain Waveforms**: Add support for frequency-domain heron models

3. **Advanced Marginalization**: Implement uncertainty-aware distance and phase marginalization

4. **Performance Optimization**: GPU acceleration for uncertainty calculations

5. **Additional Tests**: Full integration tests with trained heron models

## References

- **Matched Filtering**: Finn & Chernoff (1993), Cutler & Flanagan (1994)
- **Heron Method**: Williams et al. (2020), Phys. Rev. D 101, 063011
- **Bilby**: Ashton et al. (2019), ApJS 241, 27

## Maintenance

The implementation includes:
- Graceful handling when bilby is not installed
- Clear error messages for missing parameters
- Backward compatibility with existing heron code
- No breaking changes to existing functionality

## Contact

For questions or issues related to the bilby integration, please open an issue on the heron GitHub repository.
