.. heron documentation

Heron : The Waveform Emulator
=============================

`Heron` is a Python package for producing surrogate models of gravitational waveforms
using Gaussian Process Regression. It emulates waveforms across parameter space with
built-in uncertainty estimates (full covariance matrices), enabling both fast waveform
generation and honest error propagation.

Key features:

- **Exact and Sparse GP surrogates** — O(N³) exact GP for small datasets, O(NM²) sparse variational GP for scaling up
- **Uncertainty quantified** — full covariance matrices from GP posterior, not just point predictions
- **PN mean functions** — GP learns the residual to a Post-Newtonian inspiral, concentrating uncertainty at merger
- **Active learning** — iteratively adds training data where the model is most uncertain
- **Chirp-time warping** — physical coordinate transformation for better GP interpolation
- **Built-in evaluation** — mismatch distributions and uncertainty calibration metrics

.. warning::
   This documentation is being updated for the modernised heron architecture.
   Some older pages may reference modules that have been removed.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   readme
   installation
   getting-started

.. toctree::
   :maxdepth: 2
   :caption: Training and Usage

   training
   usage

.. toctree::
   :maxdepth: 2
   :caption: Theory

   theory

.. toctree::
   :maxdepth: 2
   :caption: Models

   model-interface

.. toctree::
   :maxdepth: 2
   :caption: Verification

   verification

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   authors
   history

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
