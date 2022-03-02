"""
Code for matched filtering using CUDA and pytorch.
"""
from copy import copy
import torch
import elk
import elk.waveform
from .utils import diag_cuda, Complex

from lal import antenna

import warnings

# TODO Change this so that disabling CUDA is handled more sensibly.
DISABLE_CUDA = False

if not DISABLE_CUDA and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class InnerProduct():
    """
    Perform the noise-weighted inner product of a signal and data.

    This implementation allows the signal to have a PSD in addition
    to the data.

    Parameters
    ----------
    psd : array-like
       The data PSD
    duration : float
       The duration of the signal
    signal_psd : array-like, optional
       The signal PSD. Defaults to None.
    f_min : float, optional
        The minimum frequency to be used.
    f_max : float, optional
        The maximium frequency to be used.

    Examples
    --------

    >>> asd = Complex(torch.ones((83, 2), device=device))
    >>> psd = asd * asd
    >>> ip = InnerProduct(psd)
    >>> a = np.random.rand(83)
    >>> b = np.random.rand(83)
    >>> ip(a, b)

    """
    def __init__(self, psd, duration, signal_psd=None, signal_cov=None, f_min=None, f_max=None):
        self.noise = psd
        self.noise2 = signal_psd
        self.signal_cov = signal_cov
        
        #if not isinstance(self.noise2, type(None)):
        #    reweight = torch.sum(self.noise.modulus / self.noise2.modulus)
        #    self.noise2 = self.noise2 / reweight
        self.duration = duration

        if not isinstance(self.signal_cov, type(None)):
            self.metric = 1./self.signal_cov
            self.metric += diag_cuda(1./(self.duration*self.noise))
        elif not isinstance(self.noise2, type(None)):
            self.metric = diag_cuda(1./(self.duration*self.noise + self.duration*self.noise2.abs()))
        else:
            self.metric = diag_cuda(1./(self.duration * self.noise))

        if f_min or f_max:
            warnings.warn("""f_min and f_max are not yet implemented. The full frequency series will be used.""",
                          RuntimeWarning)

        self.f_min = f_min
        self.f_max = f_max

    def __call__(self, a, b):
        return self.inner(a,b) #, self.duration)

    def inner(self, a, b):
        """
        Calculate the noise-weighted inner product of a and b.
        """
        c = a[1:-1].conj() @ self.metric[1:-1,1:-1] @ b[1:-1]
        return  4.0*c.real


class Likelihood():
    """
    A factory class for all heron likelihoods.
    """

    def _get_antenna_response(self, detector, ra, dec, psi, time):
        """
        Get the antenna responses for a given detector.

        Parameters
        ----------
        detectors : str
           The detector abbreviation, for example ``H1`` for the 4-km 
           detector at LIGO Hanford Observatory.
        ra : float
           The right-ascension of the source, in radians.
        dec : float
           The declination of the source, in radians.
        psi : float
           The polarisation angle, in radians.

        time : float, or array of floats
           The GPS time, or an array of GPS times, at which 
           the response should be evaluated.

        Returns
        -------
        plus : float, or array
           The 'plus' component of the response function.
        cross : float, or array
           The 'cross' component of the response function.
        """
        return antenna.AntennaResponse(detector, ra, dec, psi=psi, times=time)


class CUDALikelihood(Likelihood):
    """
    A general likelihood function for models with waveform uncertainty.

    Parameters
    ----------
    model : ``heron.model``
       The waveform model to use.
    data : ``elk.waveform.Timeseries``
       The timeseries data to analyse.
    window : ``torch.tensor``
       A tensor defining the windowing function to be applied to
       data and the model when an FFT is performed.
    asd : ``heron.utils.Complex``, optional
       The frequency series representing the amplitude spectral
       density of the noise in ``data``. Defaults to None.
    psd : ``heron.utils.Complex``, optional
       The frequency series representing the power spectral
       density of the noise in ``data``. Defaults to None.
    start : float, optional
       The start time for the signal.
       Defaults to 0.
    device : torch device
       The torch device which should be used to caluclate the likelihood.
       If CUDA is available on the system this will default to use CUDA,
       otherwise the calculation will fall-back to the CPU.
    generator_args : dict, optional
       Additional arguments to be passed to the generating function.
    f_min : float, optional
       The minimum frequency to be used in evaluating the likelihood.
    f_max : float, optional
       The maximum frequency to be used in evaluating the likelihood.

    Examples
    --------
    >>> import torch
    >>> import elk
    >>> import elk.waveform
    >>> from .utils import Complex
    >>> generator = HeronCUDAIMR()
    >>> window = torch.blackman_window(164)
    >>> noise = torch.randn(164, device=device) * 1e-20
    >>> asd = Complex((window*noise).rfft(1))
    >>> signal = generator.time_domain_waveform({'mass ratio': 0.9},
                              times=np.linspace(-0.01, 0.005, 164))
    >>> detection = Timeseries(
           data=(torch.tensor(signal[0].data, device=device)+noise).cpu(),
           times=signal[0].times)
    >>> l = Likelihood(generator, detection, window, asd=asd.clone())
    >>> l({"mass ratio": 0.5})

    Methods
    -------
    snr(frequencies, signal)
       Calculate the SNR of a given signal injection in the ASD for
       this likelihood.
    """

    def __init__(self, model, data, window, asd=None, psd=None, start=None, device=device, generator_args={}, f_min=None, f_max=None):
        """Produces a likelihood object given a model and some data."""
        self._cache_location = None
        self._cache = None
        self._weights_cache = None
        self.model = model
        self.window = window

        self.device = device

        self.f_min = f_min
        self.f_max = f_max
        self.gen_args = generator_args
        if isinstance(data, elk.waveform.Timeseries):
            self.data = self.window*torch.view_as_complex(torch.tensor(data.data, device=device).rfft(1)[1:])
            self.times = data.times
        elif isinstance(data, elk.waveform.FrequencySeries):
            self.data = data.data.clone()
            #self.data.tensor = self.data.tensor.clone()

            freqs = data.frequencies
            nt = 2*(len(freqs)-1)
            self.times = torch.linspace(0, nt/freqs[-1], nt) + start
            self.frequencies = data.frequencies
            self.start = start

        self.data *= self.model.strain_input_factor
            
        self.duration = self.times[-1] - self.times[0]
        if not isinstance(psd, type(None)):
            self.psd = psd * self.model.strain_input_factor**2
        else:
            if not isinstance(asd, type(None)):
                self.asd = asd
                #self.asd.tensor = self.asd.tensor.clone() 
            else:
                self.asd = torch.ones(len(self.data), 2)
            self.psd = (self.asd * self.asd) * self.model.strain_input_factor**2

    def _call_model(self, p):
        args = copy(self.gen_args)
        args.update(p)
        p = args
        if self._cache_location == p:
            waveform = self._cache
        else:
            waveform = self.model.frequency_domain_waveform(p, window=self.window, times=self.times)

        for pol, wf in waveform.items():
            # I've just changed these to division; need to check.
            waveform[pol].data /= self.model.strain_input_factor
            waveform[pol].variance /= self.model.strain_input_factor**2

            waveform[pol].covariance /= self.model.strain_input_factor**2
            
        return waveform

    def snr(self, signal):
        pass

    def _products(self, p, model_var):
        """
        Calculate the sum of the inner products.

        Notes
        -----
        The way that the waveform variance is currently treated
        assumes that the two polarisations are statistically
        independent.
        """

        polarisations = self._call_model(p)
        if "ra" in p.keys():

            response = self._antenna_reponse(detector=p['detector'],
                                             ra=p['ra'],
                                             dec=p['dec'],
                                             psi=p['psi'],
                                             time=p['gpstime'])

            waveform_mean = polarisations['plus'].data * response['plus'] + polarisation['cross'].data * response['cross']
            waveform_variance = polarisations['plus'].variance * response['plus']**2 + polarisations['cross'].variance * response['cross']**2

        else:
            waveform_mean = polarisations['plus'].data
            waveform_variance = polarisations['plus'].covariance

        if model_var:
            inner_product = InnerProduct(self.psd.clone(),
                                         signal_psd=waveform_variance,
                                         duration=self.duration,
                                         f_min=self.f_min,
                                         f_max=self.f_max)
            factor = torch.logdet(inner_product.metric.abs()[1:-1, 1:-1])
        else:
            inner_product = InnerProduct(self.psd.clone(),
                                         duration=self.duration,
                                         f_min=self.f_min, f_max=self.f_max)
            factor = torch.sum(torch.log(1./(self.duration*self.psd.abs())[1:-1]))
            
        products = 0
        products = -0.5 * (inner_product(self.data, self.data))
        products += -0.5 * (inner_product(waveform_mean, waveform_mean))
        products += inner_product(self.data.clone(), waveform_mean)
        products *= factor

        return products

    def _normalisation(self, p, model_var):
        """
        Calculate the normalisation.
        """
        waveform = self._call_model(p) 
        psd = self.psd.real[1:-1] / self.model.strain_input_factor**2
        if "ra" not in p.keys():
            waveform = waveform['plus']
        if model_var:
            variance = waveform.variance.abs()[1:-1]
            normalisation = (torch.sum(torch.log(psd))
                            - torch.log(torch.prod(psd / psd.max()))
                            + torch.log(psd.max())*len(psd))
            normalisation -= torch.sum(torch.log(variance))
            #normalisation += torch.logdet(variance)
            
        else:
            normalisation = (torch.sum(torch.log(psd))
                            - torch.log(torch.prod(psd / psd.max()))
                            + torch.log(psd.max())*len(psd))

        return normalisation

        
    def _log_likelihood(self, p, model_var):
        """
        Calculate the overall log-likelihood.
        """
        return self._products(p, model_var) #- self._normalisation(p, model_var)
    
    def __call__(self, p, model_var=True):
        """Calculate the log likelihood for a given set of model parameters.

        Parameters
        ----------
        p : dict
           The dictionary of waveform parameters.
        model_var : bool, optional
           Flag to include the waveform uncertainty in the likelihood estimate.
           Defaults to True.

        Returns
        -------
        log_likelihood : float
           The log-likelihood of the data at point ``p`` in the waveform
           parameter space.
        """
        return self._log_likelihood(p, model_var)
