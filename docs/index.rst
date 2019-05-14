.. heron documentation master file, created by
   sphinx-quickstart on Tue Jul  9 22:26:36 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to heron's documentation!
======================================

`Heron` is a Python package for producing surrogate models for
computationally intensive functions, such as numerical relativity
waveforms.

This version of heron is implemented slightly differently to older versions,
and should allow for a greater degree of flexibility for using different GP libraries for the modelling,
and to fit into analysis pipelines better than the earlier development versions.



Tutorials
=========

.. toctree::
   :maxdepth: 2

   usage

		
API documentation
=================

.. currentmodule:: heron

This is the low-level documentation for how each of the functions and classes in the package actually work.
If you're looking to get started using `heron` you might be better-off looking at some of the tutorials.

Waveform Models
---------------

The heron pacakge contains a plethora of different pre-constructed waveform models.

.. autosummary::
   :toctree: _autosummary
   heron.models.georgebased

Old (simple) interface
----------------------

This is the documentation for the "waveform catalogue"-like interface.
If you're starting a new project this is probably not the right way to approach the use of this library.

.. autosummary::
   :toctree: _autosummary
   heron.waveform


Matched Filtering
-----------------
.. autosummary::
   :toctree: _autosummary	     
   heron.filtering

..
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

