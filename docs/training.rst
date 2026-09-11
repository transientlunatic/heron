Training Waveform Surrogates
++++++++++++++++++++++++++++

Heron trains Gaussian Process surrogate models that interpolate gravitational waveforms
across parameter space, producing both mean predictions and full covariance matrices.

Quick Start
===========

The simplest way to train a model is via the CLI with a YAML config file:

.. code:: bash

   heron train --settings config.yaml

Or programmatically:

.. code:: python

   from heron.train import heron_train

   model = heron_train("config.yaml")
   wf = model.predict({
       "mass_ratio": 0.5,
       "time": {"lower": -0.5, "upper": 0.02, "number": 500},
   })

   # wf["plus"].data      — strain array
   # wf["plus"].covariance — full covariance matrix


Training Modes
==============

Heron supports three training modes, selected via the ``mode`` key in the config.

Fixed Grid (``mode: fixed``)
----------------------------

Generates waveforms at explicitly listed mass ratios from a reference approximant.
This is the simplest mode and closest to the original heron workflow.

Requires ``lalsuite`` (install with ``pip install heron[lal]``).

.. code:: yaml

   training:
     mode: fixed
     model: exact
     approximant: IMRPhenomPv2
     total_mass: 60.0              # solar masses
     distance: 100.0               # Mpc
     mass_ratios: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
     n_samples: 200                # time samples per waveform
     nu: 2.5
     output_scale: 1.0e+27
     iterations: 400
     warping:
       type: chirp
       alpha: 0.625
     checkpoint: checkpoints/exact_gp.pt

Active Learning (``mode: active``)
----------------------------------

Iteratively refines the training set by adding waveforms where the model is most uncertain.
This is the recommended mode for production surrogates — it converges faster with fewer total
waveforms than a fixed grid.

Requires ``lalsuite``.

The loop:

1. Generate an initial training set using Sobol quasi-random sampling
2. Train the surrogate model
3. Evaluate predictive variance at candidate parameter points
4. Generate new waveforms at the highest-uncertainty locations
5. Append to the training set and repeat

.. code:: yaml

   training:
     mode: active
     model: sparse                 # sparse GP recommended for active learning
     approximant: IMRPhenomPv2
     total_mass: 60.0
     distance: 100.0
     parameter_space:
       mass_ratio: [0.1, 1.0]     # bounds for Sobol sampling
     initial_samples: 50           # Sobol points to start
     active_iterations: 5          # refinement rounds
     points_per_iteration: 20      # new waveforms per round
     n_samples: 200                # time samples per waveform
     seed: 42
     n_inducing: 200               # sparse GP inducing points
     nu: 2.5
     output_scale: 1.0e+27
     iterations: 400
     warping:
       type: chirp
       alpha: 0.625
     mean_function:
       type: newtonian             # PN mean — GP learns residual
     checkpoint: checkpoints/sparse_active.pt
     save_training_data: checkpoints/training_data.h5

From Pre-existing Data (``mode: data``)
---------------------------------------

Loads a ``TrainingSet`` HDF5 file and trains directly. Use this for NR catalogues or
when retraining from previously saved training data. Does not require ``lalsuite``.

.. code:: yaml

   training:
     mode: data
     model: exact
     data_path: checkpoints/training_data.h5
     total_mass: 60.0
     distance: 100.0
     nu: 2.5
     output_scale: 1.0e+27
     iterations: 400
     warping:
       type: chirp
       alpha: 0.625
     checkpoint: checkpoints/from_data.pt

You can also create and save training sets programmatically:

.. code:: python

   from heron.training.dataset import TrainingSet
   import torch

   training_set = TrainingSet(
       x=torch.randn(1000, 2),        # (mass_ratio, time)
       y_plus=torch.randn(1000),
       y_cross=torch.randn(1000),
       parameter_names=["mass_ratio"],
       metadata={"source": "my_nr_catalogue"},
   )
   training_set.save("my_training_data.h5")


Model Types
===========

Exact GP (``model: exact``)
---------------------------

Full Gaussian Process with O(N³) training cost. Best for small training sets
(up to ~5000 points). Produces exact posterior covariance.

Sparse Variational GP (``model: sparse``)
------------------------------------------

Uses M inducing points to reduce cost to O(NM²). With M=200 and N=4000,
this is ~400× faster than exact GP. Same kernel, same warping, same interface.
Recommended for larger training sets and active learning.

Sparse-specific config:

- ``n_inducing``: number of inducing points (default 200). Inducing points are
  initialised via k-means clustering on the warped training data.


Mean Functions
==============

By default the GP uses a zero mean function and learns the entire waveform from scratch.
Specifying a Post-Newtonian (PN) mean function lets the GP learn only the residual
to the inspiral — the merger and ringdown corrections. This reduces the GP's workload
and concentrates uncertainty where it matters most (near merger).

Available mean functions:

- ``zero`` (default): GP learns the full waveform.
- ``newtonian``: Leading-order Newtonian inspiral. No LAL dependency, GPU-compatible.
- ``taylort2``: TaylorT2 at 1PN order. Adds phase corrections to Newtonian.

.. code:: yaml

   training:
     mean_function:
       type: newtonian    # or taylort2, or zero


Time Warping
============

Chirp-time warping compresses the long inspiral and expands the short merger, giving
the GP a more uniform view of the waveform. The ``alpha`` parameter controls the
strength of the warping (default 0.625, tuned for non-spinning BBH).

.. code:: yaml

   training:
     warping:
       type: chirp
       alpha: 0.625


Configuration Reference
========================

All keys live under the top-level ``training:`` block.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Key
     - Default
     - Description
   * - ``mode``
     - ``fixed``
     - Training mode: ``fixed``, ``active``, or ``data``
   * - ``model``
     - ``exact``
     - Surrogate type: ``exact`` or ``sparse``
   * - ``approximant``
     - ``IMRPhenomPv2``
     - LAL approximant name (fixed/active modes)
   * - ``total_mass``
     - ``60.0``
     - Reference total mass in solar masses
   * - ``distance``
     - ``100.0``
     - Reference luminosity distance in Mpc
   * - ``mass_ratios``
     - —
     - List of mass ratios (fixed mode only)
   * - ``parameter_space``
     - —
     - Parameter bounds dict (active mode only)
   * - ``data_path``
     - —
     - Path to HDF5 training set (data mode only)
   * - ``n_samples``
     - ``200``
     - Time samples per waveform
   * - ``nu``
     - ``2.5``
     - Matérn kernel smoothness (1.5 or 2.5)
   * - ``output_scale``
     - ``1e27``
     - Numerical rescaling factor
   * - ``iterations``
     - ``400``
     - Training optimisation steps
   * - ``device``
     - ``cpu``
     - Torch device (``cpu`` or ``cuda``)
   * - ``n_inducing``
     - ``200``
     - Inducing points (sparse model only)
   * - ``initial_samples``
     - ``50``
     - Initial Sobol samples (active mode)
   * - ``active_iterations``
     - ``5``
     - Refinement rounds (active mode)
   * - ``points_per_iteration``
     - ``20``
     - New waveforms per round (active mode)
   * - ``seed``
     - —
     - Random seed (active mode)
   * - ``warping.type``
     - ``chirp``
     - Warping type: ``chirp`` or ``simple``
   * - ``warping.alpha``
     - ``0.625``
     - Chirp warping exponent
   * - ``mean_function.type``
     - ``zero``
     - Mean function: ``zero``, ``newtonian``, or ``taylort2``
   * - ``checkpoint``
     - ``heron_checkpoint.pt``
     - Output checkpoint path
   * - ``save_training_data``
     - —
     - If set, save TrainingSet to this HDF5 path

A top-level ``logging:`` block with ``level: info`` (or ``debug``, ``warning``) controls
log verbosity.


Using Trained Models
====================

Load a checkpoint and generate waveforms:

.. code:: python

   from heron.models.gp.exact import ExactGPSurrogate
   # or: from heron.models.gp.sparse import SparseGPSurrogate

   model = ExactGPSurrogate.load("checkpoints/exact_gp.pt")

   wf = model.predict({
       "mass_ratio": 0.5,
       "time": {"lower": -0.5, "upper": 0.02, "number": 500},
   })

   strain = wf["plus"].data            # shape (500,)
   times = wf["plus"].times            # shape (500,)
   covariance = wf["plus"].covariance  # shape (500, 500)
   variance = wf["plus"].variance      # shape (500,) — diagonal

You can also rescale to different total masses and distances at prediction time:

.. code:: python

   wf = model.predict({
       "mass_ratio": 0.5,
       "total_mass": 80.0,             # different from training
       "luminosity_distance": 200.0,   # different from training
       "time": {"lower": -0.5, "upper": 0.02, "number": 500},
   })


Evaluating Surrogates
=====================

Heron includes built-in evaluation tools for assessing surrogate quality.

Mismatch
--------

Compute the noise-weighted overlap and mismatch between waveforms:

.. code:: python

   from heron.evaluation.mismatch import compute_overlap, compute_mismatch

   overlap = compute_overlap(h1, h2, dt)      # normalised inner product
   mismatch = compute_mismatch(h1, h2, dt)    # 1 - overlap

Uncertainty Calibration
-----------------------

Test whether the GP's predictive uncertainty is well-calibrated:

.. code:: python

   from heron.evaluation.calibration import CalibrationEvaluator

   evaluator = CalibrationEvaluator(surrogate, reference_approximant)
   results = evaluator.evaluate(n_points=200, parameter_ranges={...})

   # results.coverage       — dict of {level: fraction} (e.g. {0.9: 0.88})
   # results.ks_statistic   — Kolmogorov-Smirnov test on z-scores
   # results.qq_data        — data for Q-Q plots


Memory and Scaling
==================

Exact GP memory usage scales as O(N²), where N is the number of training points:

- 2,000 points: ~200 MB
- 5,000 points: ~1 GB
- 10,000 points: ~4 GB

For larger training sets, use the sparse GP (``model: sparse``), which scales as O(NM)
where M is the number of inducing points (typically 100–500).


Troubleshooting
===============

**Import errors for lalsuite:**
LAL is only needed for generating training data from approximants. If you're training
from pre-existing data (``mode: data``), lalsuite is not required.
Install with ``pip install heron[lal]``.

**Training loss not decreasing:**
Try adjusting the learning rate (default 0.05 for exact, 0.01 for sparse),
increasing iterations, or changing the output scale.

**Out of memory:**
Switch to ``model: sparse`` with fewer inducing points, or reduce ``n_samples``.
