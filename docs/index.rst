.. heron documentation master file, created by
   sphinx-quickstart on Tue Jul  9 22:26:36 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to heron's documentation!
======================================

`Heron` is a Python package for producing surrogate models for
computationally intensive functions, such as numerical relativity
waveforms.

Tutorials
=========

.. toctree::
   :maxdepth: 2

   usage

		
API documentation
=================

This is the low-level documentation for how each of the functions and
classes in the package actually work. If you're looking to get started
using `heron` you might be better-off looking at some of the
tutorials.

Surrogate Modelling
-------------------

.. autosummary::
   :toctree: _autosummary

   heron.data
   heron.priors
   heron.kernels
   heron.regression
   heron.training

Matched Filtering
-----------------
.. autosummary::
   :toctree: _autosummary
	     
   heron.filtering

Bayesian optimisation
---------------------

.. autosummary::
   :toctree: _autosummary

   heron.acquisition


The Heron Code repository
=========================

.. toctree::
   :maxdepth: 2

   readme
   usage
   

   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

