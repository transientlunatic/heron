"""
This module contains likelihood functions for heron to allow it to perform parameter estimation
using the waveform models it supports.
"""

from gwpy.timeseries import TimeSeries

import numpy as np
import torch

import logging

logger = logging.getLogger("heron.likelihood")


disable_cuda = False
if not disable_cuda and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class LikelihoodBase:
    pass


class Likelihood(LikelihoodBase):

    def abs(self, A):
        return np.abs(A)

    def einsum(self, *args, **kwargs):
        return np.einsum(*args, **kwargs)
    
    def logdet(self, K):
        return np.linalg.slogdet(K).logabsdet

    def inverse(self, A):
        return np.linalg.inv(A)

    def solve(self, A, B):
        return np.linalg.solve(A, B)

    def eye(self, N, *args, **kwargs):
        return np.eye(N)

    def log(self, A):
        return np.log(A)

    @property
    def pi(self):
        return np.pi


class TimeDomainLikelihood(Likelihood):

    def __init__(self, data, psd, waveform=None, detector=None, fixed_parameters={}, timing_basis=None):
        self.psd = psd

        self.data = np.array(data.data)
        self.data_ts = data
        self.times = data.times
        
        self.logger = logger = logging.getLogger(
            "heron.likelihood.TimeDomainLikelihood"
        )
        self.N = len(self.times)
        self.C = self.psd.covariance_matrix(times=self.times)
        factor = 1e30
        self.inverse_C = self.inverse(self.C*factor**2)
        self.dt = self.abs((self.times[1] - self.times[0]).value)
        
        self.normalisation = - (self.N/2) * self.log(2*self.pi) + (self.logdet(self.C*1e30) - self.log(1e30)) *self.dt
        #* (self.dt * self.dt / 4) / 4
        self.logger.info(f"Normalisation: {self.normalisation}")

        self.N = len(self.times)

        if waveform is not None:
            self.waveform = waveform

        if detector is not None:
            self.detector = detector

        self.fixed_parameters = fixed_parameters
        if timing_basis is not None:
            self.fixed_parameters['reference_frame'] = timing_basis


    def snr(self, waveform):
        """
        Calculate the signal to noise ratio for a given waveform.
        """
        factor = 1e30
        N = len(self.times)
        h_h = (
            (np.array(waveform.data).T*factor @ self.solve(self.C*factor**2, np.array(waveform.data)*factor)) * self.dt
        )
        return np.sqrt(np.abs(h_h))

    def snr_f(self, waveform):
        dt = (self.times[1] - self.times[0]).value
        T = (self.times[-1] - self.times[0]).value
        wf = np.fft.rfft(waveform.data * dt)
        S = self.psd.frequency_domain(frequencies=np.arange(0, 0.5/dt, 1/T))
        A = (4 *  (wf.conj()*wf)[:-1] / S.value[:-1] / T)
        return np.sqrt(np.real(np.sum(A)))
    
    def log_likelihood(self, waveform, norm=True):
        """
        Calculate the log likelihood of a given waveform and the data.

        Parameters
        ----------
        waveform : `heron.types.Waveform`
           The waveform to compare to the data.

        Returns
        -------
        float
           The log-likelihood for the waveform.
        """
        factor = 1e30
        try:
            assert(np.all(self.times == waveform.times))
        except AssertionError:
            print(self.times, waveform.times)
            raise Exception
        residual = (self.data * factor) - (np.array(waveform.data) * factor)
        weighted_residual = (residual.T @ self.inverse_C @ residual) * (self.dt)
        # why is this negative using Toeplitz?
        self.logger.info(f"residual: {residual}; chisq: {weighted_residual}")
        out = - 0.5 * weighted_residual
        if norm:
            out += self.normalisation
        return out

    def __call__(self, parameters):
        self.logger.info(parameters)

        keys = set(parameters.keys())
        extrinsic = {"phase", "psi", "ra", "dec", "theta_jn", "zenith", "azimuth", "gpstime"}
        conversions = {"geocent_time", "mass_ratio", "total_mass", "luminosity_distance", "chirp_mass"}
        bad_keys = keys - set(self.waveform.allowed_parameters) - extrinsic - conversions
        if len(bad_keys) > 0:
            print("The following keys were not recognised", bad_keys)
        if self.fixed_parameters:
            parameters.update(self.fixed_parameters)
        test_waveform = self.waveform.time_domain(
            parameters=parameters, times=self.times
        )
        projected_waveform = test_waveform.project(self.detector)
        llike = self.log_likelihood(projected_waveform)
        self.logger.info(f"log likelihood: {llike}")
        return llike


class TimeDomainLikelihoodModelUncertainty(TimeDomainLikelihood):

    def __init__(self, data, psd, waveform=None, detector=None, fixed_parameters=None, timing_basis=None):
        super().__init__(data, psd, waveform, detector, fixed_parameters, timing_basis)

    def _weighted_data(self):
        """Return the weighted data component"""
        # TODO This can all be pre-computed
        factor = 1e23
        factor_sq = factor**2

        if not hasattr(self, "weighted_data_CACHE"):
            dw = self.weighted_data_CACHE = (
                -0.5 * (self.data*factor) @ self.solve(self.C*factor_sq, self.data*factor)
            )
        else:
            dw = self.weighted_data_CACHE
        return dw

    def log_likelihood(self, waveform, norm=True):

        waveform_d = np.array(waveform.data)
        waveform_c = np.array(waveform.covariance)
        K = waveform_c
        mu = waveform_d
        factor = 1/np.max(K)
        factor_sq = factor**2
        factor_sqi = factor**-2

        K = K*factor_sq
        C = self.C * factor_sq
        Ci = self.inverse(self.C * factor_sq)
        Ki = self.inverse(K)
        A = self.inverse(C + K)
        mu = mu * factor
        data = self.data*factor

        sigma = self.inverse(Ki+Ci)*factor_sqi
        B = (self.einsum("ij,i", Ki, mu) + (self.einsum("ij,i", Ci, data)))*factor

        N = - (self.N / 2) * self.log(2*self.pi) +  0.5 * (self.logdet(sigma) - self.logdet(self.C) - self.logdet(K) - 3 * self.log(factor_sq)) * self.dt

        data_like = (self._weighted_data())
        model_like = -0.5 * self.einsum("i,ij,j", mu, Ki, mu)
        shift = + 0.5 * self.einsum('i,ij,j', B, sigma, B) #* ((self.dt) / 4)
        like = data_like + model_like + shift

        if norm:
            like += N
        return like


class MultiDetector:
    """
    This class provides a means of calculating the log likelihood for multiple detectors.
    """

    def __init__(self, *args):
        self._likelihoods = []
        for detector in args:
            if isinstance(detector, LikelihoodBase):
                self._likelihoods.append(detector)
        self.logger = logger = logging.getLogger(
            "heron.likelihood.MultiDetector"
        )

    def __call__(self, parameters):
        out = 0
        self.logger.info(f"Calling likelihood at {parameters}")
        for detector in self._likelihoods:
            out += detector(parameters)

        return out

