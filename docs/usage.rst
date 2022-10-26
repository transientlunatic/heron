===============
Getting started
===============

In order to run one of Heron's built-in models you'll need to import that model as well as Heron itself, for example: ::

  import heron
  from heron.models.georgebased import HeronHodlr

  import numpy as np

We also imported numpy for convenience.

This will load latest version of the NR-trained 7-dimensional Heron model.
We now need to set the generator up to produce waveforms.
Currently we need to tell the model the total mass of the system, but this ::

  generator = HeronHodlr()
  
Two different types of waveform can be requested from the model: the mean waveform (and its variance), or individual waveform samples.

In order to produce a mean waveform we need to provide the model with the instrinsic properties of the system, that is, the mass ratio, and the spin parameters. If any parameters are omitted from the dictionary they're set to zero. ::

  waveform = generator.mean(times=np.linspace(-0.02, 0.02), p={"mass ratio": 0.3})

The output of the ``mean`` method is two waveforms (one each for the plus and cross polarisations).

The ``data`` attribute of each waveform contains the mean strain data, while the ``variance`` attribute contains the variance on this mean waveform.

Alternatively, Heron can return individual function draws. These may not look especially similar to what you would expect out of a normal waveform model, but used collectively they can allow the calculation of various statistics. ::

  samples = generator.distribution(samples=100, times=np.linspace(-0.02, 0.02), p={"mass ratio": 3})


This produces 100 waveform samples drawn from the model, at the same model configuration as the previous mean waveform.
