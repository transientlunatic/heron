==================================
Model Verification and Evaluation
==================================

Heron includes a built-in evaluation framework for assessing surrogate quality
using standard gravitational wave metrics and statistical calibration tests.

The evaluation module lives in ``heron.evaluation`` and provides three components:

1. **Mismatch evaluation** — noise-weighted overlap against reference waveforms
2. **Uncertainty calibration** — z-scores, coverage fractions, Q-Q plots
3. **Reporting** — text summaries and matplotlib diagnostic plots

All evaluators work without ``lalsuite`` if you use
``SineGaussianWaveform`` as the reference (useful for testing).
For production evaluation against LAL approximants, install ``lalsuite``.


CLI Usage
=========

The fastest way to evaluate a trained model is via the CLI:

.. code:: bash

   heron evaluate --settings evaluate.yaml

Chain it after training in one command:

.. code:: bash

   heron train --settings train.yaml && heron evaluate --settings eval.yaml

The evaluation config is a YAML file with an ``evaluation:`` block:

.. code:: yaml

   logging:
     level: info

   evaluation:
     checkpoint: checkpoints/exact_gp.pt
     model: exact                      # exact | sparse
     reference: IMRPhenomPv2           # IMRPhenomPv2 | SEOBNRv3 | SineGaussian
     name: MyModel_v1

     parameter_bounds:
       mass_ratio: [0.1, 1.0]

     time:
       lower: -0.5
       upper: 0.02
       number: 512

     mismatch: true                    # run mismatch evaluation
     calibration: true                 # run calibration evaluation
     n_mismatch: 200                   # held-out points for mismatch
     n_calibration: 100                # held-out points for calibration
     seed: 42
     device: cpu

     output_dir: evaluation            # directory for report + plots
     plots: true                       # generate matplotlib plots

This produces:

- ``evaluation/report.txt`` — text summary of all results
- ``evaluation/plots/`` — diagnostic plots (if ``plots: true`` and matplotlib installed)

See ``examples/evaluate.yaml`` for a complete example.

Configuration reference
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Key
     - Default
     - Description
   * - ``checkpoint``
     - (required)
     - Path to trained model checkpoint
   * - ``model``
     - ``exact``
     - Model type: ``exact`` or ``sparse``
   * - ``reference``
     - ``IMRPhenomPv2``
     - Reference approximant: ``IMRPhenomPv2``, ``SEOBNRv3``, or ``SineGaussian``
   * - ``name``
     - checkpoint filename
     - Model name for plot titles and report
   * - ``parameter_bounds``
     - from model
     - Parameter bounds for Sobol sampling (overrides model's own bounds)
   * - ``time``
     - ``{lower: -0.5, upper: 0.02, number: 512}``
     - Time grid for waveform comparison
   * - ``mismatch``
     - ``true``
     - Run mismatch evaluation
   * - ``calibration``
     - ``true``
     - Run calibration evaluation
   * - ``n_mismatch``
     - ``200``
     - Number of held-out mismatch evaluation points
   * - ``n_calibration``
     - ``100``
     - Number of held-out calibration evaluation points
   * - ``seed``
     - —
     - Random seed for reproducibility
   * - ``output_dir``
     - ``evaluation``
     - Output directory
   * - ``plots``
     - ``true``
     - Generate diagnostic plots


Mismatch Evaluation
===================

The mismatch (1 − overlap) is the standard metric for waveform faithfulness.

Low-level functions
-------------------

For comparing two individual waveforms:

.. code:: python

   from heron.evaluation import compute_overlap, compute_mismatch
   import numpy as np

   n = 4096
   dt = 1.0 / 4096
   t = np.arange(n) * dt

   h1 = np.sin(2 * np.pi * 100 * t) * np.exp(-50 * t)
   h2 = np.sin(2 * np.pi * 101 * t) * np.exp(-50 * t)

   overlap = compute_overlap(h1, h2, dt)        # normalised inner product
   mismatch = compute_mismatch(h1, h2, dt)      # 1 - overlap

   # With a power spectral density for noise-weighted overlap:
   freqs = np.fft.rfftfreq(n, d=dt)
   psd = np.ones_like(freqs)
   overlap_weighted = compute_overlap(h1, h2, dt, psd=psd)

The overlap is computed in the frequency domain:

.. math::

   \langle h_1 | h_2 \rangle = 4 \, \mathrm{Re} \int_0^{\infty}
   \frac{\tilde{h}_1(f) \, \tilde{h}_2^*(f)}{S_n(f)} \, df

and normalised so that ``overlap(h, h) = 1``.


MismatchEvaluator
-----------------

For systematic evaluation across parameter space, ``MismatchEvaluator``
draws Sobol-sampled held-out points and computes the mismatch at each:

.. code:: python

   from heron.evaluation import MismatchEvaluator
   from heron.models.gp.exact import ExactGPSurrogate

   model = ExactGPSurrogate.load("checkpoint.pt")
   reference = IMRPhenomPv2()   # or any waveform generator

   evaluator = MismatchEvaluator(model, reference)
   result = evaluator.evaluate(
       n_points=200,
       parameter_bounds={"mass_ratio": (0.1, 1.0)},
       time_config={"lower": -0.5, "upper": 0.02, "number": 512},
       seed=42,
   )

   print(result.summary())
   # Mismatch evaluation (200 points):
   #   Median mismatch:  2.31e-04
   #   Worst mismatch:   8.74e-03
   #   Worst at:         {'mass_ratio': 0.12}
   #   < 1e-3 (detect):  85.0%
   #   < 1e-2 (PE):      100.0%

The ``MismatchResult`` object contains:

- ``mismatches``: array of mismatch values at each held-out point
- ``parameters``: dict of parameter arrays (for plotting mismatch vs parameter)
- ``worst_mismatch``, ``median_mismatch``: summary statistics
- ``fraction_below_1e3``, ``fraction_below_1e2``: fraction meeting detection/PE thresholds
- ``worst_parameters``: parameter values at the worst mismatch point

Quality thresholds:

- **Detection-grade**: mismatch < 10⁻³ everywhere
- **PE-grade**: mismatch < 10⁻² everywhere


Uncertainty Calibration
=======================

A good surrogate should not just predict accurate waveforms — it should
*know where it is uncertain*. The calibration evaluator tests whether
the model's predicted uncertainty is honest.

.. code:: python

   from heron.evaluation import CalibrationEvaluator

   evaluator = CalibrationEvaluator(model, reference)
   result = evaluator.evaluate(
       n_points=100,
       parameter_bounds={"mass_ratio": (0.1, 1.0)},
       time_config={"lower": -0.5, "upper": 0.02, "number": 256},
       seed=42,
   )

   print(result.summary())

At each held-out parameter point, the evaluator:

1. Predicts waveform + covariance from the surrogate
2. Generates the "true" waveform from the reference
3. Computes z-scores: ``z = (truth - mean) / std``
4. Computes log predictive density under the full multivariate normal

Z-scores
--------

If the uncertainty is perfectly calibrated, the z-scores should follow
a standard normal distribution N(0, 1). The evaluator runs a
Kolmogorov-Smirnov (KS) test against N(0, 1) and reports the test statistic
and p-value.

Coverage fractions
------------------

At each credible level (50%, 68%, 90%, 95%, 99%), the evaluator checks
what fraction of true waveform samples fall within the predicted credible
interval. For a well-calibrated model, the observed coverage should match
the nominal level.

Q-Q plot data
-------------

The ``CalibrationResult.qq_data`` property returns ``(theoretical, observed)``
quantile arrays for constructing a Q-Q plot. Departures from the diagonal
indicate miscalibration.

Log predictive density
----------------------

The log predictive density evaluates the *full* multivariate normal likelihood
of the true waveform under the predicted distribution. This tests both
mean accuracy and covariance calibration simultaneously. Higher values
indicate better calibration.

The ``CalibrationResult`` object contains:

- ``z_scores``: array of shape ``(n_points, n_times)``
- ``coverage``: dict mapping ``"90%"`` → observed fraction
- ``ks_statistic``, ``ks_pvalue``: KS test results
- ``log_predictive_densities``: array of per-point log densities
- ``mean_log_pred_density``: mean across all points
- ``qq_data``: property returning ``(theoretical, observed)`` quantile arrays


Evaluation Reports
==================

The ``EvaluationReport`` class combines mismatch and calibration results
into a single report with text summaries and optional diagnostic plots.

.. code:: python

   from heron.evaluation import EvaluationReport

   report = EvaluationReport(
       mismatch=mismatch_result,
       calibration=calibration_result,
       name="my_surrogate_v1",
   )

   # Text summary
   print(report.summary())

   # Save text to file
   report.save_summary("evaluation/report.txt")

   # Generate diagnostic plots (requires matplotlib)
   report.plot_all("evaluation/plots/")

Generated plots
---------------

When ``matplotlib`` is available, ``report.plot_all()`` generates:

- **mismatch_histogram.png** — histogram of log10(mismatch) with detection/PE threshold lines
- **mismatch_vs_{parameter}.png** — scatter plot of mismatch vs each parameter (identifies systematic weak spots)
- **qq_plot.png** — Q-Q plot of observed z-scores vs N(0,1)
- **zscore_histogram.png** — histogram of z-scores overlaid with the N(0,1) PDF
- **coverage.png** — observed vs nominal coverage fractions (diagonal = perfect calibration)

If ``matplotlib`` is not installed, ``plot_all()`` returns an empty list
without raising an error.


Complete Evaluation Example
===========================

.. code:: python

   from heron.models.gp.exact import ExactGPSurrogate
   from heron.models.lalsimulation import IMRPhenomPv2
   from heron.evaluation import (
       MismatchEvaluator,
       CalibrationEvaluator,
       EvaluationReport,
   )

   # Load trained surrogate
   model = ExactGPSurrogate.load("checkpoints/exact_gp.pt")
   reference = IMRPhenomPv2()

   bounds = {"mass_ratio": (0.1, 1.0)}
   time_cfg = {"lower": -0.5, "upper": 0.02, "number": 512}

   # Mismatch evaluation
   mm = MismatchEvaluator(model, reference)
   mm_result = mm.evaluate(n_points=200, parameter_bounds=bounds,
                           time_config=time_cfg, seed=42)

   # Calibration evaluation
   cal = CalibrationEvaluator(model, reference)
   cal_result = cal.evaluate(n_points=100, parameter_bounds=bounds,
                             time_config=time_cfg, seed=42)

   # Generate report
   report = EvaluationReport(
       mismatch=mm_result,
       calibration=cal_result,
       name="ExactGP_IMRPhenomPv2",
   )
   print(report.summary())
   report.save_summary("evaluation/report.txt")
   report.plot_all("evaluation/plots/")


Testing Without LAL
===================

All evaluators work without ``lalsuite`` by using ``SineGaussianWaveform``
as the reference:

.. code:: python

   from heron.models.testing import SineGaussianWaveform

   reference = SineGaussianWaveform()
   evaluator = MismatchEvaluator(my_surrogate, reference)
   # ... same interface as above

This is how the unit tests in ``tests/test_evaluators.py`` work —
they wrap ``SineGaussianWaveform`` as both surrogate and reference
to verify the evaluation machinery without external dependencies.
