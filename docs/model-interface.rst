=============================
The Heron Modelling Interface
=============================

All models implemented in the `heron` package are built on top of the `Model` class, which provides a number of useful methods to assist in creating a Gaussian process model.

.. autoclass:: heron.models.Model

All of the methods which are provided by this class are intended to be treated as *private* methods, which other classes build upon, and which aren't intended to be accessed directly.

To build a new model we can begin by inheriting the `Model` class. ::
  
  class NewModel(Model):
     pass

This will give your new model access to the various methods which are needed to produce waveform outputs from a model.

As a minimum your new model must contain three methods so that `heron` can interact with it properly.

``distribution()``
------------------
This method should return the parameters of the waveform distribution at a given location in parameter space, i.e. a vector representing the mean and variance at each requested location.
The function should have the following signature ::

  distribution(self, p, times, *args, **kwargs)

With `p` being a dictionary of coordinates in the parameter space, and `times` being a list or array of times at which the waveform distribution should be produced.


``mean()``
----------
This method should return only the mean waveform at a given location in the parameter space.
It should have the signature ::

  mean(p, times, *args, **kwargs)

Where ``p`` is a dictionary of coordinates in the parameter space, and ``times`` is a list or array of times at which the waveform distribution should be produed.

``train()``
------------
This method defines the correct way to "train" the model, in order to determine the optimal values of the hyperparameters for the model (using an empirical Bayesian approach).
This method should, at the minimum, set ``self.training`` to ``True`` for the model, and ``self.evaluate`` to ``False``, in order to mark the model as being in a training state, and not suitable for evaluation.

If this method conducts the entire training process you may then switch these flags to set ``self.evaluate`` to ``True`` and ``self.training`` to ``False`` before the method completes its execution.
Alternatively you should define an ``eval()`` method to do this, and any other cleaning-up which should be done after training.
If ``heron`` encounters a model in training state when attempting to evaluate it, it will first attempt to run the ``eval()`` method on the model.


In addition to the ``Model`` class, a number of additional helper classes exist within ``heron``, mainly to help with the construction of gravitational wave models.

Gravitational wave-specific classes
-----------------------------------

The ``heron.models.gw.BBHSurrogate`` should be inherited in a class if it is designed to emulate binary black hole waveforms.
This class provides metadata related to the intrinsic parameters of these systems.
For simpler models which don't include spin effects you should use the ``heron.models.gw.BBHNonSpinSurrogate`` class instead.

For time-domain strain models you should have your model class inherit the ``heron.models.gw.HofTSurrogate`` class.
This provides interfaces to the waveform model which are particular to a time-domain model.


Frequency domain transforms
---------------------------

``Heron`` is capable of performing the Fourier transform of the outputs of its models.
Unlike a conventional waveform model, where the conversion from time-domain to frequency-domain requires applying the DFT to the time-domain waveform, ``heron`` must compute the transformation on the multidimensional normal distribution which the model outputs.

Fortunately, this process is straight-forward.
A ``Heron`` model provides a mean waveform, and the associated covariance matrix, meaning that a waveform, $h$ drawn from the model can be represented as a series of draws of the form

.. math::

   h(t) ~ \mathcal{N}(\mu, \Sigma)

for :math:`\mu` and :math:`\Sigma` respectively the mean and covariance at the appropriate point in parameter space.

Given the Gaussian function has the property that :math:`\forall A \in \mathcal{C}`

.. math::

   Ax ~ \mathcal(A \mu, A \Sigma A^{\mathrm{T}})

It is possible to represent the discrete Fourier transform as a matrix operator (the DFT Matrix), so for

.. math::

   \tilde{x}(f) = \mathcal{F}[x(t)] = Fx ~ \mathcal{N}(F\mu, F \Sigma F^{\mathrm{T}})

meaning that a draw of a frequency domain waveform can be calculated by drawing from the normal distribution with a mean of ``fft(mean_t)`` and covariance ``fft2(covariance_t)``.


The ``frequency_domain_waveform`` method can be used on models which support the conversion to provide frequency domain waveforms.
