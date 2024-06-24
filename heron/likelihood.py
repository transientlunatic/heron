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
        self.normalisation = - (self.N/2) * self.log(2*self.pi) - 0.5 * self.logdet(self.C)
        self.logger.info(f"Normalisation: {self.normalisation}")
        self.inverse_C = np.linalg.inv(self.C)

        self.dt = self.abs((self.times[1] - self.times[0]).value)
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
        dt = (self.times[1] - self.times[0]).value
        factor = 1e30
        N = len(self.times)
        h_h = (
            (np.array(waveform.data).T*factor @ self.solve(self.C*factor**2, np.array(waveform.data)*factor))
            * (dt * dt / 4)
            / 4
        )
        return np.sqrt(np.abs(h_h))

    def snr_f(self, waveform):
        dt = (self.data.times[1] - self.data.times[0]).value
        T = (self.data.times[-1] - self.data.times[0]).value
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
        assert(np.all(self.times == waveform.times))
        residual = (self.data * factor) - (np.array(waveform.data) * factor)
        #weighted_residual = (residual.T @ self.inverse(self.C) @ residual) * (self.dt * self.dt / 4) / 4
        weighted_residual = self.einsum('i,ij,j', residual, self.inverse(self.C*factor**2), residual) * (self.dt * self.dt / 4) / 4
        # why is this negative using Toeplitz?
        self.logger.info(f"residual: {residual}; chisq: {weighted_residual}")
        out = - 0.5 * weighted_residual
        if norm:
            out += 0.5 * self.normalisation
        return out

    def __call__(self, parameters):
        self.logger.info(parameters)

        keys = set(parameters.keys())
        extrinsic = {"phase", "psi", "ra", "dec", "theta_jn", "zenith", "azimuth", "gpstime"}
        conversions = {"mass_ratio", "total_mass", "luminosity_distance"}
        bad_keys = keys - set(self.waveform._args.keys()) - extrinsic - conversions
        if len(bad_keys) > 0:
            print("The following keys were not recognised", bad_keys)
        parameters.update(self.fixed_parameters)
        test_waveform = self.waveform.time_domain(
            parameters=parameters, times=self.times
        )
        projected_waveform = test_waveform.project(self.detector)
        llike = self.log_likelihood(projected_waveform)
        self.logger.info(f"log likelihood: {llike}")
        return llike


class TimeDomainLikelihoodModelUncertainty(TimeDomainLikelihood):

    def __init__(self, data, psd, waveform=None, detector=None):
        super().__init__(data, psd, waveform, detector)

    def _normalisation(self, K, S):
        norm = (
            -1.5 * K.shape[0] * self.log(2 * self.pi)
            - 0.5 * self.logdet(2*self.pi*self.inverse(K))
            - 0.5 * self.logdet(2*self.pi*self.inverse_C)
            - 0.5 * self.logdet(self.inverse_C)
            - 0.5 * self.logdet(self.solve(self.C, K) + self.eye(K.shape[0]))
        )
        return norm

    def _weighted_data(self):
        """Return the weighted data component"""
        # TODO This can all be pre-computed
        if not hasattr(self, "weighted_data_CACHE"):
            self.logger.info(f"Data max/min: {np.min(self.data)}/{np.max(self.data)}")
            self.logger.info(f"C max/min: {np.min(self.C)}/{np.max(self.C)}")
            dw = self.weighted_data_CACHE = (
                -0.5 * self.data.T @ self.solve(self.C, self.data)
            )
        else:
            dw = self.weighted_data_CACHE
        return dw

    def _weighted_model(self, mu, K):
        """Return the inner product of the GPR mean"""
        return -0.5 * np.array(mu).T @ self.solve(K, mu)

    def _weighted_cross(self, mu, K):
        a = self.solve(self.C, self.data) - self.solve(K, mu)
        b = self.inverse_C + self.inverse(K)
        return - 0.5 * a.T @ self.solve(b, a)

    def log_likelihood(self, waveform, norm=True):
        W = self._weighted_cross(waveform.data, waveform.covariance)
        A = self._weighted_data()
        B = self._weighted_model(waveform.data, waveform.covariance)
        N = self._normalisation(waveform.covariance, self.C)
        like = W + A + B
        if norm:
            like += N
        self.logger.info(f"Likelihood components: {W}, {A}, {B}, {N}")
        return like#, W, A, B, N


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


class LikelihoodPyTorch(Likelihood):

    def logdet(self, K):
        A = torch.slogdet(K)
        return A.logabsdet #* A.sign

    def inverse(self, A):
        out, info = torch.linalg.inv_ex(A)
        if info == 0:
            return out
        else:
            raise ValueError(f"Matrix could not be inverted: {info}")

    def solve(self, A, B):
        return torch.linalg.solve(A, B)

    def abs(self, A):
        return torch.abs(A)
    
    def eye(self, N, *args, **kwargs):
        return torch.eye(N, device=self.device, dtype=torch.double)

    def log(self, A):
        return torch.log(A)

    @property
    def pi(self):
        return torch.tensor(torch.pi, device=self.device)

    def einsum(self, *args, **kwargs):
        return torch.einsum(*args, **kwargs)


class TimeDomainLikelihoodPyTorch(LikelihoodPyTorch):

    def __init__(self, data, psd, waveform=None, detector=None, fixed_parameters={}, timing_basis=None):
        self.logger = logger = logging.getLogger(
            "heron.likelihood.TimeDomainLikelihoodPyTorch"
        )
        self.device = device
        self.logger.info(f"Using device {device}")
        self.psd = psd

        self.data = torch.tensor(data.data, device=self.device, dtype=torch.double)
        self.times = data.times

        self.C = self.psd.covariance_matrix(times=self.times)
        self.C = torch.tensor(self.C, device=self.device)
        self.inverse_C = torch.linalg.inv(self.C)

        self.dt = (self.times[1] - self.times[0]).value
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
        dt = (self.times[1] - self.times[0]).value
        self.logger.info(f"Contains {self.N} points with dt {dt}.")
        waveform_d = torch.tensor(waveform.data, device=self.device, dtype=torch.double)
        h_h = (waveform_d @ self.solve(self.C, waveform_d)) * (dt / 4)
        return 2*torch.sqrt(torch.abs(h_h))

    def snr_f(self, waveform):
        dt = (self.times[1] - self.times[0]).value
        T = (self.times[-1] - self.times[0]).value
        wf = np.fft.rfft(waveform.data * dt)
        # Single-sided PSD
        S = self.psd.frequency_domain(frequencies=np.arange(0, 0.5/dt, 1/T))
        A = (4 *  (wf.conj()*wf)[:-1] / (S.value[:-1]) / T)
        return np.sqrt(np.real(np.sum(A)))
    
    
    def log_likelihood(self, waveform, norm=True):
        waveform_d = torch.tensor(waveform.data, device=self.device, dtype=torch.double)
        residual = self.data - waveform_d
        weighted_residual = (
            (residual) @ self.solve(self.C, residual) * (self.dt * self.dt / 4) / 4
        )
        normalisation = self.logdet(2 * np.pi * self.C) * self.N
        #print("normalisation", normalisation)
        like = -0.5 * weighted_residual
        if norm:
            like -= 0.5 * normalisation
        return like

    def __call__(self, parameters):
        self.logger.info(parameters)

        keys = set(parameters.keys())
        extrinsic = {"phase", "psi", "ra", "dec", "theta_jn", "zenith", "azimuth"}
        conversions = {"mass_ratio", "total_mass", "luminosity_distance"}
        bad_keys = keys - set(self.waveform._args.keys()) - extrinsic - conversions
        if len(bad_keys) > 0:
            print("The following keys were not recognised", bad_keys)
        parameters.update(self.fixed_parameters)
        test_waveform = self.waveform.time_domain(
            parameters=parameters, times=self.times
        )
        projected_waveform = test_waveform.project(self.detector)
        return self.log_likelihood(projected_waveform)


class TimeDomainLikelihoodModelUncertaintyPyTorch(TimeDomainLikelihoodPyTorch):

    def __init__(self, data, psd, waveform=None, detector=None):
        super().__init__(data, psd, waveform, detector)

    def _weighted_data(self):
        """Return the weighted data component"""
        # TODO This can all be pre-computed
        factor = 1#1e23
        factor_sq = 1#factor**2

        if not hasattr(self, "weighted_data_CACHE"):
            dw = self.weighted_data_CACHE = (
                -0.5 * (self.data*factor) @ self.solve(self.C*factor_sq, self.data*factor) # * (self.dt * self.dt / 4) / 4
            )
        else:
            dw = self.weighted_data_CACHE
        return dw

    def log_likelihood(self, waveform, norm=True):
        waveform_d = torch.tensor(waveform.data, device=self.device, dtype=torch.double)
        waveform_c = torch.tensor(
            waveform.covariance, device=self.device, dtype=torch.double
        )
        K = waveform_c
        mu = waveform_d

        #print(torch.sum(mu.cpu()-self.data.cpu()))

        factor = 1/torch.max(K)
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

        N = - (self.N / 2) * self.log(2*self.pi) + 0.5 * self.logdet(sigma) - 0.5*self.logdet(self.C) - 0.5*self.logdet(K)

        data_like = (self._weighted_data())
        model_like = -0.5 * self.einsum("i,ij,j", mu, Ki, mu)
        shift = + 0.5 * self.einsum('i,ij,j', B, sigma, B) #* ((self.dt) / 4)
        like = data_like + model_like + shift

        if norm:
            like += N

        return like
