# Heron Testing Framework

## Philosophy

This framework implements a **progressive complexity approach** to model validation:

1. **Level 0**: Single waveform, single point in time
2. **Level 1**: Single waveform, full time series
3. **Level 2**: Multiple waveforms at fixed mass ratios
4. **Level 3**: Full parameter space coverage
5. **Level 4**: Systematic comparison across models

Each level must pass before proceeding to the next. This ensures we catch problems early and understand exactly where models fail.

## Structure

```
tests/test_framework/
├── README.md                  # This file
├── __init__.py               # Framework exports
├── base.py                   # Base test classes
├── metrics.py                # Validation metrics (mismatch, overlap, etc.)
├── fixtures.py               # Common test fixtures
├── progressive/              # Progressive test suites
│   ├── level0_single_point.py
│   ├── level1_time_series.py
│   ├── level2_multi_waveform.py
│   ├── level3_parameter_space.py
│   └── level4_model_comparison.py
└── reports/                  # Test result reports (gitignored)
```

## Usage

### Run a specific level

```bash
pytest tests/test_framework/progressive/level1_time_series.py -v
```

### Run all levels sequentially

```bash
pytest tests/test_framework/progressive/ -v --tb=short
```

### Generate a validation report

```python
from tests.test_framework import ValidationReport

report = ValidationReport("my_model")
report.run_progressive_tests()
report.save("validation_report.html")
```

## Metrics

All tests use standardized metrics:

- **Mismatch**: 1 - overlap, target < 1e-3
- **Overlap**: Noise-weighted inner product, target > 0.999
- **Amplitude Ratio**: GPR_amp / Reference_amp, target ≈ 1.0
- **Correlation**: Pearson correlation, target > 0.99
- **Phase Error**: Max phase difference (degrees), target < 5°

## Test-Driven Development

Before implementing a model change:

1. Write a test at the appropriate level
2. Run the test (it should fail)
3. Implement the change
4. Run the test (it should pass)
5. Document the change in `CHANGELOG.md`

## Future Extensions

This framework is designed to be extended for:

- Spinning waveforms (add spin parameters)
- Higher-order modes (add mode indices)
- Precessing systems (add precession angles)
- NR-trained models (swap out training data source)
