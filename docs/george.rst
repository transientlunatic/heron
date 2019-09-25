===================
George-based models
===================

A number of models implemented in `Heron` make use of the `George` Gaussian process library which implements a number of simplifications to make the inversion of the covariance matrix required for GPR predictions more tractable.

The main model produced this way is `HeronHODLR`, which implements a fully-spinning BBH waveform model which is trained on waveform data from the Georgia Tech waveform catalogue.

All of the george-based models are contained in the `heron.models.georgebased` module.

HeronHODLR: A spinning, NR-trained waveform model
-------------------------------------------------

Ther `HeronHODLR` model implements a surrogate model for gravitational waveforms form binary black hole events with arbitrary spin parameters between a mass ratio of 1 and 8.

.. autoclass:: heron.models.georgebased.HeronHodlr


Heron2DHodlrIMR
---------------

This model is a 2D prototype waveform model trained on phenomenological sample waveforms.
In contrast to the full `HeronHODLR` model, this model models only non-spinning waveforms between mass ratios of 1 and 10.


.. autoclass:: heron.models.georgebased.Heron2dHodlrIMR
   
