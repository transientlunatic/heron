=====
Heron
=====

.. image:: https://badge.fury.io/py/heron-model.svg
    :target: https://badge.fury.io/py/heron-model

.. image:: https://travis-ci.org/transientlunatic/heron.svg?branch=master
    :target: https://travis-ci.org/transientlunatic/heron

The ``heron`` package is a python library for using Gaussian Process Regression (GPR) to emulate functions which are expensive to 

It was originally built for producing a surrogate model for numerical
relativity waveforms from binary black hole coalesences, but the code
should be sufficiently general to allow other surrogate models to be
built.

In order to handle very large models, ``heron`` can use the `george`_
python package to generate the underlying Gaussian Process, which can
handle very large models thanks to its use of a hierarchical matrix
inverter.

..

Features
--------

* Single-valued function surrogate production from multivalued inputs
* Handling very large datasets.

.. _george: http://dan.iel.fm/george/
.. _emcee: http://dan.iel.fm/emcee/
