.. heron documentation 

Heron : The Waveform Emulator
=============================

`Heron` is a Python package for producing surrogate models for computationally intensive functions, such as numerical relativity
waveforms.

This version of heron is implemented slightly differently to older versions, and should allow for a greater degree of flexibility for using different GP libraries for the modelling, and to fit into analysis pipelines better than the earlier development versions.

.. warning::
   This documentation is still being written, and you may find a few places either where documentation is missing, or where the formatting isn't good.
   If you spot something which has clearly been missed in the documentation please open an issue on the git repository.
   

.. toctree::
   :maxdepth: 2
   :caption: Basics
	      
   readme
   installation
   getting-started
   authors
   
Tutorials and Examples
======================
.. toctree::
   :maxdepth: 2
   :caption: Tutorials
	     
   usage
   theory
   training
   verification-tutorial
	likelihood-tutorial
   

Pre-supplied Models
===================

This package contains a number of different pre-baked waveform models, and the data which is required to reproduce them.
It's also fairly easy to use the existing framework to implement a new model, using the same training data as pre-supplied models, or using new training data.

.. toctree::
   :maxdepth: 2
   :caption: Models

   george
   gpytorch

   model-interface
   

Inference
=========

Using waveform models which have uncertainty requires a new likelihood function compared to models without.

.. toctree::
   :maxdepth: 2
   :caption: Inference

   likelihood
   
Testing and verifying models
============================

.. toctree::
   :maxdepth: 2
   :caption: Verification

   verification

Theory
======

.. toctree::
   :maxdepth: 1
   :caption: Theory

   theory
   likelihood
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

