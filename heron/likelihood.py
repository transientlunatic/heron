"""
Code for matched filtering using CUDA and pytorch.
"""
from math import sqrt
from copy import copy
import torch
import elk
import elk.waveform

import scipy.linalg
import scipy.signal

from lal import cached_detector_by_prefix

import warnings
import logging

# TODO Change this so that disabling CUDA is handled more sensibly.
DISABLE_CUDA = False

if not DISABLE_CUDA and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

logger = logging.getLogger("heron.likelihood")


def determine_overlap(timeseries_a, timeseries_b):
    def is_in(time, timeseries):
        diff = torch.min(torch.abs(timeseries - time))
        if diff < (timeseries[1] - timeseries[0]):
            return True, diff
        else:
            return False, diff

    overlap = None
    if (
        is_in(timeseries_a.times[-1], timeseries_b.times)[0]
        and is_in(timeseries_b.times[0], timeseries_a.times)[0]
    ):
        overlap = timeseries_b.times[0], timeseries_a.times[-1]
    elif (
        is_in(timeseries_a.times[0], timeseries_b.times)[0]
        and is_in(timeseries_b.times[-1], timeseries_a.times)[0]
    ):
        overlap = timeseries_a.times[0], timeseries_b.times[-1]
    elif (
        is_in(timeseries_b.times[0], timeseries_a.times)[0]
        and is_in(timeseries_b.times[-1], timeseries_a.times)[0]
        and not is_in(timeseries_a.times[-1], timeseries_b.times)[0]
    ):
        overlap = timeseries_b.times[0], timeseries_b.times[-1]
    elif (
        is_in(timeseries_a.times[0], timeseries_b.times)[0]
        and is_in(timeseries_a.times[-1], timeseries_b.times)[0]
        and not is_in(timeseries_b.times[-1], timeseries_a.times)[0]
    ):
        overlap = timeseries_a.times[0], timeseries_a.times[-1]
    else:
        overlap = None
        return None

    start_a = torch.argmin(torch.abs(timeseries_a.times - overlap[0]))
    finish_a = torch.argmin(torch.abs(timeseries_a.times - overlap[-1]))

    start_b = torch.argmin(torch.abs(timeseries_b.times - overlap[0]))
    finish_b = torch.argmin(torch.abs(timeseries_b.times - overlap[-1]))
    return (start_a, finish_a), (start_b, finish_b)


class Overlap:
    def __init__(
        self,
        psd,
        duration,
        window,
        signal_psd=None,
        signal_cov=None,
        f_min=None,
        f_max=None,
    ):
        self.psd = psd

        self.window = window

        self.duration = duration
        self.signal_psd = signal_psd
        self.signal_cov = signal_cov

        self.f_min = f_min
        self.f_max = f_max

        self.inner_product = InnerProduct(
            self.psd, duration=self.duration, f_min=self.f_min, f_max=self.f_max
        )

    def __call__(self, a, b):
        if isinstance(a, elk.waveform.Timeseries):
            self.data_a = torch.fft.rfft(a.data)  # self.window*a.data)
            self.times_a = a.times

        if isinstance(b, elk.waveform.Timeseries):
            b.data = torch.tensor(b.data)
            self.data_b = torch.fft.rfft(b.data)  # self.window*b.data)
            self.times_b = b.times

        overlap = self.inner_product(self.data_a, self.data_b)
        normalisation = (
            1.0
            / torch.sqrt(self.inner_product(self.data_a, self.data_a))
            / torch.sqrt(self.inner_product(self.data_b, self.data_b))
        )
        return (
            4
            * 1.0
            / (self.times_a[-1] - self.times_a[0])
            / len(self.times_a)
            * torch.abs(overlap)
            * normalisation
        )


class Match:
    def __init__(
        self,
        psd,
        duration,
        window,
        signal_psd=None,
        signal_cov=None,
        f_min=None,
        f_max=None,
    ):
        self.psd = psd

        self.window = window

        self.duration = duration
        self.signal_psd = signal_psd
        self.signal_cov = signal_cov

        self.f_min = f_min
        self.f_max = f_max

        self.inner_product = InnerProduct(
            self.psd, duration=self.duration, f_min=self.f_min, f_max=self.f_max
        )

    def __call__(self, a, b):
        if isinstance(a, elk.waveform.Timeseries):
            self.data_a = torch.fft.rfft(
                self.window(len(a.data), device=a.data.device) * a.data
            )
            self.times_a = a.times

        if isinstance(b, elk.waveform.Timeseries):
            b.data = torch.tensor(b.data, device=a.data.device)
            self.data_b = torch.fft.rfft(
                self.window(len(b.data), device=b.data.device) * b.data
            )
            self.times_b = b.times

        correlation = self.data_a.conj() * self.data_b
        if self.inner_product.noise:
            correlation /= self.inner_product.noise

        fs = (len(self.times_a) - 1) / (self.times_a[-1] - self.times_a[0])

        normalisation = (
            2
            * fs
            / torch.sqrt(self.inner_product(self.data_a, self.data_a))
            / torch.sqrt(self.inner_product(self.data_b, self.data_b))
        )
        return torch.fft.ifft(correlation) * normalisation


class InnerProduct:
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

    >>> asd = torch.ones(size=(83), device=device, dtype=torch.cfloat)
    >>> psd = asd * asd
    >>> ip = InnerProduct(psd=psd)
    >>> a = np.random.rand(83)
    >>> b = np.random.rand(83)
    >>> ip(a, b)

    """

    def __init__(
            self,
            duration,
            metric=None,
            f_min=None,
            f_max=None,
    ):
        self.duration = duration

        if metric is None:
            self.metric = None
        elif metric.ndim == 1:
            self.metric = torch.tensor(scipy.linalg.toeplitz(metric.cpu())).to(
                device=device, dtype=torch.complex128
            )

        elif metric.ndim == 2:
            self.metric = metric.to(
                device=device, dtype=torch.complex128
            )        

        self.f_min = f_min if f_min else None
        self.f_max = f_max if f_max else None

    def __call__(self, a, b):
        return self.inner(a, b)

    def inner(self, a, b):
        """
        Calculate the noise-weighted inner product of a and b.
        """
        # NB linalg.solve(A, B) == linalg.inv(A) @ B
        a = a.to(dtype=torch.complex128)
        b = b.to(dtype=torch.complex128)
        if self.metric is None:
            c = a[self.f_min: self.f_max].conj() * b[self.f_min: self.f_max]
        elif isinstance(self.metric, float):
            c = a[self.f_min: self.f_max].conj() * torch.linalg.solve(self.metric, b[self.f_min:self.f_max])
        else:
            c = (
                a[self.f_min: self.f_max].conj()
                @ torch.linalg.solve(self.metric[self.f_min: self.f_max, self.f_min: self.f_max].real.to(
                    dtype=torch.complex128
                ), b[self.f_min: self.f_max])
            )
        return torch.sum(4.0 * c.real * (1./self.duration))


class Likelihood:
    """
    A factory class for all heron likelihoods.
    """

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
        logger.debug(f"Likelihood: {self._log_likelihood(p, model_var)}")
        return self._log_likelihood(p, model_var)


class CUDATimedomainLikelihood(Likelihood):
    """
    A general likelihood function for models with waveform uncertainty in the time domain.

    Parameters
    ----------
    model : ``heron.model``
       The waveform model to use.
    data : ``elk.waveform.Timeseries``
       The timeseries data to analyse.
    asd : ``heron.utils.Complex``, optional
       The frequency series representing the amplitude spectral
       density of the noise in ``data``. Defaults to None.
    psd : ``heron.utils.Complex``, optional
       The frequency series representing the power spectral
       density of the noise in ``data``. Defaults to None.
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



    Methods
    -------

    """

    def __init__(
        self,
        model,
        data,
        detector_prefix=None,
        asd=None,
        psd=None,
        device=device,
        generator_args={"duration": 0.51, "sample rate": 2048},
        f_min=None,
        f_max=None,
    ):
        """Produces a likelihood object given a model and some data."""
        self._detector_prefix = detector_prefix
        self.detector = cached_detector_by_prefix[detector_prefix]
        self._cache_location = None
        self._cache = None
        self._weights_cache = None
        self.model = model
        self.device = device

        self.lower_frequency = 20

        self.psd = psd

        self.gen_args = generator_args

        if isinstance(data, elk.waveform.Timeseries):
            self.timeseries = data
            self.data = data.data.clone()
            self.times = data.times.clone()
            self.duration = self.times[-1] - self.times[0]

        elif isinstance(data, elk.waveform.FrequencySeries):
            # TODO: Correctly handle frequency domain input data
            pass

        # Convert PSD into the noise matrix

        self.C = (
            torch.fft.irfft(
                torch.tensor(psd.data, device=self.device, dtype=torch.double),
                norm="forward",
                n=len(self.times),
            )
            * psd.df
        )
        self.C = torch.tensor(scipy.linalg.toeplitz(self.C.cpu()), device=self.device)

        self.inverse_C = torch.inverse(self.C)

        logger.info(f"Heron likelihood initialised for {self._detector_prefix}")

    def _call_model(self, p, times):
        args = copy(self.gen_args)
        args.update(p)
        p = args
        if self._cache_location == p:
            logger.debug(f"Evaluating [cached] at {p}")
            # print(f"Evaluating [cached] at {p}")
            waveform = self._cache   
        else:
            logger.debug(f"Evaluating at {p}")
            waveform = self.model.time_domain_waveform(p=p, times=times)
            sos = scipy.signal.butter(
                10,
                self.lower_frequency,
                "hp",
                fs=float(1 / (waveform.times[1] - waveform.times[0])),
                output="sos",
            )
            waveform.data = torch.tensor(
                scipy.signal.sosfilt(sos, waveform.data.cpu()),
                device=waveform.data.device,
            )
            self._cache = waveform
            self._cache_location = p
        return waveform

    def _align_time_axis(self, times, data, draw):
        """
        Align the time axis of the drawn waveform to the time axis of the data.
        """
        epoch = torch.min(draw.times)
        times -= epoch
        draw.times -= epoch  # This probably isn't quite right
        # Find the closest bin to the starts
        zero_bin = int(torch.argmin(torch.abs(times - draw.times)))
        # First roll the data so it aligns with the waveform
        # data = torch.roll(data, -zero_bin)#, axis=0)

        return data[zero_bin : zero_bin + len(draw.data)]

    def determine_overlap(self, A, B):
        return determine_overlap(A, B)

    def _residual(self, draw):
        indices = self.determine_overlap(self.timeseries, draw)
        residual = (
            self.data  # [indices[0][0] : indices[0][1]]
            - draw.data  # [indices[1][0] : indices[1][1]]
        ).to(dtype=torch.double)
        return residual

    def snr(self, p, model_var=True):
        """
        Calculate the SNR.
        """
        draw = self._call_model(p)
        residual = self._residual(draw)

        if model_var:
            aligned_C = self._align_time_axis(self.times, self.C, draw)
            aligned_covariance = self._align_time_axis(
                self.times, draw.covariance, draw
            )
            snr = self._weighted_residual_power(
                residual, aligned_C + aligned_covariance
            )
            # snr = residual @ torch.inverse(self.C+draw.covariance) @ residual
        else:
            noise = torch.ones(self.C.shape[0]) * 1e-40
            noise = scipy.linalg.toeplitz(noise)
            noise = torch.tensor(noise, device=self.device)
            aligned_C = self._align_time_axis(self.times, self.C, noise)
            snr = residual @ torch.inverse(aligned_C + noise) @ residual
        return torch.sqrt(snr)
    
    def _normalisation(self, K, S):
        norm = - 1.5 * K.shape[0] * torch.log(torch.tensor(2)*torch.pi) \
            - 0.5 * self._safe_logdet(K) \
            + 0.5 * self._safe_logdet(self.C) \
            - 0.5 * self._safe_logdet(torch.linalg.solve(K, self.C)+torch.eye(K.shape[0], device=device))
        return norm

    def _safe_logdet(self, A):
        """
        Caclulate the log determinant of an awkward matrix.
        """
        X = torch.logdet(A*1e48) - A.shape[0] * 110.524084464
        if torch.isnan(X) or torch.isinf(X):
            X = torch.tensor(- A.shape[0] * 110.524084464)
        return X

    def _weighted_data(self):
        """Return the weighted data component"""
        # TODO This can all be pre-computed
        if not hasattr(self, "weighted_data_CACHE"):
            dw = self.weighted_data_CACHE = -0.5 * self.data.T @ torch.linalg.solve(self.C, self.data)
        else:
            dw = self.weighted_data_CACHE
        return dw

    def _weighted_model(self, mu, K):
        """Return the inner product of the GPR mean"""
        return -0.5 * mu.T @ torch.linalg.solve(K, mu)

    def _weighted_cross(self, mu, K):
        a = (torch.linalg.solve(self.C, self.data) + torch.linalg.solve(K, mu))
        b = (self.inverse_C + torch.inverse(K))
        return 0.5 * a.T @ torch.linalg.solve(b, a)

    def _log_likelihood(self, p, model_var=True, noise=None):
        """
        Calculate the overall log-likelihood.
        """
        p["detector"] = self._detector_prefix
        times = self.times.clone()
        waveform = self._call_model(p, times)

        if model_var:
            # Calculate the form of the likelihood which includes contributions
            # from all waveform draws.
            like = self._weighted_cross(waveform.data, waveform.covariance)
            #print("cross term", like)
            A = self._weighted_data()
            #print("data term", A)
            B = self._weighted_model(waveform.data, waveform.covariance)
            #print("model term", B)
            like = like + A + B
            norm = self._normalisation(waveform.covariance, self.C)
            #print("normalisation", norm)
            like += norm
        else:
            residual = self.data - waveform.data
            like = -0.5*residual.T@torch.linalg.solve(self.C, residual)
            like -= (self.C.shape[0]/2) * torch.log(torch.tensor(2 * torch.pi)) + 0.5 * self._safe_logdet(self.C)

        return like


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
    >>> noise = torch.rand(164, device=device) * 1e-20
    >>> asd = Complex((window*noise).rfft(1))
    >>> signal = generator.time_domain_waveform({'mass ratio': 0.9},
                              times=np.linspace(-0.01, 0.005, 164))
    >>> detection = Timeseries(
           data=(torch.tensor(signal[0].data, device=device)+noise).cpu(),
           times=signal[0].times)
    >>> l = Likelihood(generator, detection, window, 'H1', asd=asd.clone())
    >>> l({"mass ratio": 0.5})

    Methods
    -------
    snr(frequencies, signal)
       Calculate the SNR of a given signal injection in the ASD for
       this likelihood.
    """

    def __init__(
        self,
        model,
        data,
        window,
        detector_prefix,
        asd=None,
        psd=None,
        start=0,
        device=device,
        generator_args={},
        f_min=None,
        f_max=None,
    ):
        """Produces a likelihood object given a model and some data."""
        self._detector_prefix = detector_prefix
        self.detector = cached_detector_by_prefix[detector_prefix]
        self._cache_location = None
        self._cache = None
        self._weights_cache = None
        self.model = model
        self.window = window

        self.device = device

        self.f_min = f_min
        self.f_max = f_max
        self.gen_args = generator_args
        self.gen_args["detector"] = self._detector_prefix
        if isinstance(data, elk.waveform.Timeseries):
            self.data = data.to_frequencyseries(window=self.window).data
            self.times = data.times.clone()
            self.duration = self.times[-1] - self.times[0]

        elif isinstance(data, elk.waveform.FrequencySeries):
            # Work with a frequency-domain input.
            self.data = data.data.clone()
            freqs = data.frequencies
            nt = (len(freqs) - 1) * 2
            self.duration = nt / freqs[-1]
            self.times = torch.linspace(0, self.duration, nt) + start
            self.frequencies = torch.tensor(data.frequencies, device=data.data.device)

        self.start = start

        if not isinstance(psd, type(None)):
            self.psd = psd
        else:
            if not isinstance(asd, type(None)):
                self.asd = asd
            else:
                self.asd = torch.ones(len(self.data), 2)
            self.psd = self.asd * self.asd
        self.psd = self.psd.to(torch.complex128)
        self.data = self.data.to(torch.complex128)

        self.inner_product_data = InnerProduct(
            metric=self.psd.clone(),
            duration=self.duration,
            f_min=self.f_min,
            f_max=self.f_max,
        )
        self.data_inner_product = self.inner_product_data(self.data, self.data)
        
    def _call_model(self, p):
        args = copy(self.gen_args)
        args.update(p)
        p = args
        if self._cache_location == p:
            waveform = self._cache
        else:
            waveform = self.model.frequency_domain_waveform(
                p=p, window=self.window, times=self.times.clone()
            )
            self._cache = waveform
            self._cache_location = p
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
        waveform_variance = None
        polarisations = self._call_model(p)
        if "ra" in p.keys():
            waveform_mean = polarisations.data
            waveform_variance = polarisations.variance.abs()
            waveform_covariance = polarisations.covariance.abs()
        else:
            waveform_mean = polarisations["plus"].data
            waveform_variance = polarisations["plus"].variance.abs()

        if model_var:
            inner_product_main = InnerProduct(
                metric=torch.inverse(self.psd.clone()) + torch.inverse(waveform_variance),
                duration=self.duration,
                f_min=self.f_min,
                f_max=self.f_max,
            )
            inner_product_gpr = InnerProduct(
                metric=waveform_variance,
                duration=self.duration,
                f_min=self.f_min,
                f_max=self.f_max,
            )
        else:
            inner_product_main = InnerProduct(
                metric=self.psd,
                duration=self.duration,
                f_min=self.f_min,
                f_max=self.f_max
            )

        A = torch.transpose(self.data) * torch.inverse(
            scipy.linalg.toeplitz(self.psd.clone())
        ) - waveform_mean * torch.inverse(waveform_covariance)

        products = 0
        products += +0.5 * inner_product_main(A, A)
        products += -0.5 * self.data_inner_product
        products += -0.5 * inner_product_gpr(waveform_mean, waveform_mean)

        return products

    def _normalisation(self, p, model_var):
        """
        Calculate the normalisation.
        """
        waveform = self._call_model(p)
        psd = self.psd.abs()  # / self.model.strain_input_factor**2
        if "ra" not in p.keys():
            waveform = waveform["plus"]
        if model_var:
            variance = waveform.variance
            normalisation = torch.diag(psd) + torch.diag(variance)
        else:
            normalisation = torch.diag(psd)
        normalisation = torch.logdet(torch.sqrt(normalisation.abs())) * sqrt(
            2 * torch.pi
        )
        return normalisation

    def _log_likelihood(self, p, model_var):
        """
        Calculate the overall log-likelihood.
        """
        return self._products(p, model_var)  # - self._normalisation(p, model_var)
