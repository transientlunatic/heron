"""
This module contains likelihood functions for heron to allow it to perform parameter estimation
using the waveform models it supports.
"""

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

    array = np.array
    device = "cpu"
    
    def logdet(self, K):
        (sign, logabsdet) = np.linalg.slogdet(K)
        return logabsdet

    def det(self, A):
        return np.linalg.det(A)
    
    def inverse(self, A):
        return np.linalg.inv(A)

    def solve(self, A, B):
        return np.linalg.solve(A, B)

    def eye(self, N, *args, **kwargs):
        return np.eye(N)

    def log(self, A):
        return np.log(A)

    def to_device(self, A, device):
        return A

    @property
    def pi(self):
        return np.pi


class TimeDomainLikelihood(Likelihood):

    def __init__(
        self,
        data,
        psd,
        waveform=None,
        detector=None,
        fixed_parameters={},
        timing_basis=None,
    ):
        self.psd = psd
        self.timeseries = data
        self.data = self.array(data.data)
        self.times = data.times
        self.C = self.psd.covariance_matrix(times=self.times)
        self.inverse_C = self.inverse(self.C)

        self.dt = (self.times[1] - self.times[0]).value
        self.N = len(self.times)

        if waveform is not None:
            self.waveform = waveform

        if detector is not None:
            self.detector = detector

        self.fixed_parameters = fixed_parameters
        if timing_basis is not None:
            self.fixed_parameters["reference_frame"] = timing_basis

        self.logger = logger = logging.getLogger(
            "heron.likelihood.TimeDomainLikelihood"
        )

    def snr(self, waveform):
        """
        Calculate the optimal signal to noise ratio for a given waveform.
        """
        dt = (self.times[1] - self.times[0]).value
        N = len(self.times)
        w = self.array(waveform.data, device=self.device)
        h_h = (
            (w.T @ self.solve(self.C, w)) 
        ) / len(w)**2
        
        return np.sqrt(2*np.abs(h_h))

    def log_likelihood(self, waveform, norm=True):
        w = self.timeseries.determine_overlap(self, waveform)
        if w is not None:
            (a,b) = w
        else:
            return -np.inf
        residual = self.array(self.data.data[a[0]:a[1]]) - self.array(waveform.data[b[0]:b[1]])
        residual = self.to_device(residual, self.device)
        weighted_residual = (
            (residual) @ self.solve(self.C[a[0]:a[1],b[0]:b[1]], residual) / len(residual)**2
        )
        N = len(residual)
        normalisation = N * self.log(2*np.pi) + self.logdet(self.C[a[0]:a[1],b[0]:b[1]]) if norm else 0
        return - 0.5 * weighted_residual - 0.5 * normalisation

    def __call__(self, parameters):
        self.logger.info(parameters)

        keys = set(parameters.keys())
        extrinsic = {"phase", "psi", "ra", "dec", "theta_jn", "gpstime", "geocent_time"}
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


class TimeDomainLikelihoodModelUncertainty(TimeDomainLikelihood):

    def __init__(self, data, psd, waveform=None, detector=None):
        super().__init__(data, psd, waveform, detector)

        self.norm_factor_2 = np.max(self.C)
        self.norm_factor = np.sqrt(self.norm_factor_2)

    def log_likelihood(self, waveform, norm=True):
        a, b = self.timeseries.determine_overlap(self, waveform)
        
        wf = self.to_device(self.array(waveform.data), self.device)[b[0]:b[1]]
        data = self.data[a[0]:a[1]]

        C = self.C[a[0]:a[1], a[0]:a[1]]
        K = self.to_device(self.array(waveform.covariance[b[0]:b[1],b[0]:b[1]]), self.device)
        W_0 = self.to_device(self.array(data - wf), device=self.device)
        N = len(W_0)
        W = - 0.5 * self.solve((K+C), W_0) @ W_0
        N = - 0.5 * N*self.log((2*self.pi)) - 0.5 * self.logdet((C+K)) if norm else 0
        return W + N


class MultiDetector:
    """
    This class provides a means of calculating the log likelihood for multiple detectors.
    """

    def __init__(self, *args):
        self._likelihoods = []
        for detector in args:
            if isinstance(detector, LikelihoodBase):
                self._likelihoods.append(detector)

    def __call__(self, parameters):
        out = 0
        for detector in self._likelihoods:
            out += detector(parameters)

        return out

