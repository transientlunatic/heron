# Heron Testing Framework

## Philosophy

This framework implements a **progressive complexity approach** to model validation:

1. **Level 0**: Single waveform, single point in time
2. **Level 1**: Single waveform, full time series
3. **Level 2**: Multiple waveforms at fixed mass ratios
4. **Level 3**: Full parameter space coverage
5. **Level 4**: Systematic comparison across models

Each level must pass before proceeding to the next. This ensures we catch problems early and understand exactly where models fail.

## Status

Implemented: levels 0-2 plus the harness self-tests. Level 3 (full parameter
space, mismatch heatmaps) and level 4 (model comparison) are not written yet;
the held-out-q check in level 2 is the seed of level 3.

## Structure

```
tests/test_framework/
├── base.py          Harness (surrogate + oracle + grids + per-level limits),
│                    Prediction, WaveformSurrogateAdapter (wraps heron models)
├── metrics.py       mismatch, correlation, amplitude ratio, phase error,
│                    Thresholds, compare_waveforms -> MetricReport (numpy/scipy only)
├── fixtures.py      analytic chirp oracle, OraclePlusError (injectable defects),
│                    make_analytic_harness, make_gp_harness (needs torch)
├── conftest.py      `harness` fixture (analytic + gp) and level gating
├── test_harness_selftest.py   injected defects MUST be caught
├── progressive/
│   ├── test_level0_single_point.py
│   ├── test_level1_time_series.py
│   └── test_level2_multi_waveform.py
└── reports/         (gitignored)
```

## Usage

```bash
pytest tests/test_framework -v                 # everything
pytest tests/test_framework -k analytic        # no torch needed
pytest tests/test_framework -m "level(1)"      # one level
```

**Gating**: tests are marked `@pytest.mark.level(n)`. If any test at level
`< n` failed for a harness, level `n` tests for that harness are skipped with
the reason, so a broken model stops at the first failing level.

**Adding a model**: write a `Harness` (an adapter exposing
`predict(q, times) -> Prediction`, an oracle `(q, times) -> (plus, cross)`, and
`Thresholds` per level) and add it to the `harness` fixture params. Models
that are less accurate relax their `Thresholds` explicitly; the tests do not
change.

**Ratchets**: thresholds for known-weak behaviour (e.g. the GP harness's
held-out-q limit) record today's measured baseline, not an aspiration.
Tighten them as the model improves.

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
