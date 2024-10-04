=====================================================
Waveform likelihood with waveform model uncertainties
=====================================================


Using the ``heron`` likelihood function
---------------------------------------

In order to allow Bayesian inference to be conducted using the ``heron`` model a new likelihood function is required which is able to incorporate the waveform uncertainty in the likelihood estimate.

The ``heron.likelihood`` function contains suitable likelihood functions for incorporation into sampling engines.

CUDALikelihood
++++++++++++++

The CUDA Likelihood function allows the GPU on a system to be leveraged to perform likelihood calculations, and should be used with models which use the ``heron.models.torchbased.CUDAModel`` base class.
This likelihood function allows considerable acceleration compared to a CPU-based likelihood function, since the waveform is both generated and correlated on the GPU.

.. autoclass:: heron.likelihood.CUDALikelihood

Theory
------

This section details the derivation and the behaviour of the matched-filtering likelihood function for heron waveform models.

The approach taken to derive this is closely related to the approach described in :cite:`gpr:waveform:Moore+15`,
however we do not take the approach of adding a statistical waveform difference to an existing approximant,
and instead use a statistical model which produces the waveform as a statistical distribution.

Given measured data :math:`d(f)` which is composed of some signal :math:`s(f)` and stationary Gaussian noise :math:`n(f)` which has a power spectral density :math:`S_n(f)` :cite:`sensitivity:Moore+15`, that is

.. math::

   d(f) = s(f) + n(f),

then we can perform matched filtering to analyse the signal :math:`s(f)` using some waveform model :math:`h(f,\vec{\lambda})`.

For convenience from this point forward we define :math:`s \gets s(f)`, and :math:`h(\vec{\lambda}) \gets h(f, \vec{\lambda})`.

From Bayes Theorem

.. math::

   p(\vec{\lambda} | s) = \frac{ p'(s | \vec{\lambda}) p(\vec{\lambda}) }{ p(s) }

where :math:`p'(s|\vec{\lambda})` is the likelihood that the signal :math:`s` is a realisation of the model :math:`h(\lambda)`.

This likelihood is then

.. math::

   p'(s | \vec{\lambda}) \propto \exp \left( - \frac{1}{2} \left\langle s - h(\vec{\lambda}) | s - h(\vec{\lambda}) \right\rangle \right)

which introduces the noise-weighted inner product of two vectors,

.. math::

   \langle x | y \rangle = 4 \mathfrak{R} \left\{ \sum_{\kappa=1}^M \frac{x(f_{\kappa}) y^*(f_\kappa)}{S_n(f_{\kappa})} \delta f \right\}

with :math:`\kappa` labelling the :math:`M` frequency bins witha  resolution :math:`\delta f`.

The models we use for the gravitational waveform are known to be imperfect, however if we imagine a perfect waveform, we can define a likelihood function :math:`p(s|\vec{\lambda})` which represents this model.
For a good approximate model then :math:`p(s|\vec{\lambda}) \approx p'(s|\vec{\lambda})`.

The conventional approach to improving this agreement is to seek ever better approximate models.
The approach outlined in :cite:`gpr:waveform:Moore+15` works by modelling the difference between the "true" likelihood and the approximate one using Gaussian process regression.

Here we take a third approach.
Each waveform drawn from the ``Heron`` model is a draw from a probability distribution; given the probabilistic nature of the waveform it is necessary to include the probability of the waveform in the likelihood function, and then marginalise this out as a nuisance parameter, that is

.. math::

   p(s | \vec{\lambda}) \propto \int p(h(\vec{\lambda})) \exp\left(-\frac{1}{2} \langle s-h(\vec{\lambda}) | s-h(\vec{\lambda}) \rangle \right) \mathrm{d}h(\vec{\lambda})

The expression for :math:`p(h(\vec{\lambda})` for the ``heron`` model is analytic, by virtue of it being a Gaussian process.
Letting :math:`h \gets h(\lambda)`,

.. math::

   p(h(\vec{\lambda})) &= \frac{1}{ \sqrt{(2 \pi)^{k} |K|} }
   \exp \left( - \frac{1}{2} (h - \mu)^T K^{-1} (h-\mu) \right)

   &=  \frac{1}{ \sqrt{(2 \pi)^{k} |K|} }
   \exp \left( - \frac{1}{2} \sum_{i,j}^{M} [K^{-1}]_{i,j} (h-\mu)(f_i) (h-\mu)^*(f_j) \right)

   &= \frac{1}{ \sqrt{(2 \pi)^{k} |K|} }  \exp \left( - \frac{1}{2} (h-\mu | h-\mu) \right)

with :math:`\mu \gets \mu(\vec{\lambda})` and :math:`K \gets K(\vec{\lambda})` respectively the mean and the covariance matrix of the Gaussian process evaluated at :math:`\vec{\lambda}` for a set of frequencies :math:`f_1 \cdots f_M`.
For convenience we can introduce the notation :math:`(x|y)` for the inner product weighted by the model variance.

The full likelihood expression is then the integral of the product of Gaussians, which is analytical, giving

.. math::

    p(s | \vec{\lambda}) \propto \frac{1}{1+ \prod_{\kappa=1}^{M} \sigma^2(f_{\kappa}, \vec{\lambda}) / S_n(f_{\kappa})}
    \exp\left( - \frac{1}{2} \cdot 4 \mathfrak{R} \left\{
    \sum^M_{\kappa=1} \frac{ (s(f_\kappa)-\mu(f_\kappa, \vec{\lambda}))(s(f_\kappa)-\mu(f_\kappa, \vec{\lambda}))^* }{S_n(f_{\kappa}) + \sigma^2(f_\kappa, \vec{\lambda})} \delta f \right\} \right)


Implementation
--------------


    
.. bibliography:: heron.bib


The production of the likelihood for the heron model goes through a number of steps, given that the waveform model itself works in the time-domain, but the standard method of calculating the likelihood uses a frequency-domain representation.
		  
.. graphviz::

   digraph operations {
      "likelihood._log_likelihood()" -> "likelihood.__call__()";
      "likelihood._normalisation()" -> "likelihood._log_likelihood()" ;
      "likelihood._products()" -> "likelihood._log_likelihood()" ;

      "likelihood.psd" -> "likelihood._products()" ;
      "likelihood.data" -> "likelihood._products()";

      "waveform" -> "likelihood._products()";

      "model.frequency_domain()" -> "likelihood._call_model()" -> "waveform";

      "model._predict" -> "rfft(mean waveform)" -> "model.frequency_domain()";
      "model._predict" -> "rfft(covariance matrix)" -> "diag(freq covariance matrix)" -> "model.frequency_domain()";
      
   }

The time-domain waveform, and associated covariance matrix are 
