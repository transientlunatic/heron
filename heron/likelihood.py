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
        self.times = data.times

        self.C = self.psd.covariance_matrix(times=self.times)
        self.inverse_C = np.linalg.inv(self.C)

        self.dt = (self.times[1] - self.times[0]).value
        self.N = len(self.times)

        if waveform is not None:
            self.waveform = waveform

        if detector is not None:
            self.detector = detector

        self.fixed_parameters = fixed_parameters
        if timing_basis is not None:
            self.fixed_parameters['reference_frame'] = timing_basis

        self.logger = logger = logging.getLogger(
            "heron.likelihood.TimeDomainLikelihood"
        )

    def snr(self, waveform):
        """
        Calculate the signal to noise ratio for a given waveform.
        """
        dt = (self.times[1] - self.times[0]).value
        N = len(self.times)
        h_h = (
            (np.array(waveform.data).T @ self.solve(self.C, np.array(waveform.data)))
            * (dt * dt / N / 4)
            / 4
        )
        return np.sqrt(np.abs(h_h))

    def log_likelihood(self, waveform):
        residual = np.array(self.data.data) - np.array(waveform.data)
        weighted_residual = (
            (residual) @ self.solve(self.C, residual) * (self.dt * self.dt / 4) / 4
        )
        normalisation = self.logdet(2 * np.pi * self.C)
        return -0.5 * weighted_residual + 0.5 * normalisation

    def __call__(self, parameters):
        self.logger.info(parameters)

        keys = set(parameters.keys())
        extrinsic = {"phase", "psi", "ra", "dec", "theta_jn"}
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

    def _normalisation(self, K, S):
        norm = (
            -1.5 * K.shape[0] * self.log(2 * self.pi)
            - 0.5 * self.logdet(K)
            + 0.5 * self.logdet(self.C)
            - 0.5 * self.logdet(self.solve(K, self.C) + self.eye(K.shape[0]))
        )
        return norm

    def _weighted_data(self):
        """Return the weighted data component"""
        # TODO This can all be pre-computed
        if not hasattr(self, "weighted_data_CACHE"):
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
        a = self.solve(self.C, self.data) + self.solve(K, mu)
        b = self.inverse_C + self.inverse(K)
        return 0.5 * a.T @ self.solve(b, a)

    def log_likelihood(self, waveform):
        like = self._weighted_cross(waveform.data, waveform.covariance)
        # print("cross term", like)
        A = self._weighted_data()
        # print("data term", A)
        B = self._weighted_model(waveform.data, waveform.covariance)
        # print("model term", B)
        like = like + A + B
        norm = self._normalisation(waveform.covariance, self.C)
        # print("normalisation", norm)
        like += norm

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

    def __call__(self, parameters):
        out = 0
        for detector in self._likelihoods:
            out += detector(parameters)

        return out


class LikelihoodPyTorch(Likelihood):

    def logdet(self, K):
        return torch.slogdet(K).logabsdet

    def inverse(self, A):
        return torch.linalg.inv(A)

    def solve(self, A, B):
        return torch.linalg.solve(A, B)

    def eye(self, N, *args, **kwargs):
        return torch.eye(N, device=self.device, dtype=torch.double)

    def log(self, A):
        return torch.log(A)

    @property
    def pi(self):
        return torch.tensor(torch.pi, device=self.device)


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
        N = len(self.times)
        waveform_d = torch.tensor(waveform.data, device=self.device, dtype=torch.double)
        h_h = (waveform_d.T @ self.solve(self.C, waveform_d)) * (dt * dt / N / 4) / 4
        return torch.sqrt(torch.abs(h_h))

    def log_likelihood(self, waveform):
        waveform_d = torch.tensor(waveform.data, device=self.device, dtype=torch.double)
        residual = self.data - waveform_d
        weighted_residual = (
            (residual) @ self.solve(self.C, residual) * (self.dt * self.dt / 4) / 4
        )
        normalisation = self.logdet(2 * np.pi * self.C)
        return -0.5 * weighted_residual + 0.5 * normalisation

    def __call__(self, parameters):
        self.logger.info(parameters)

        keys = set(parameters.keys())
        extrinsic = {"phase", "psi", "ra", "dec", "theta_jn"}
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

    def _normalisation(self, K, S):
        norm = (
            -1.5 * K.shape[0] * self.log(2 * self.pi)
            - 0.5 * self.logdet(K)
            + 0.5 * self.logdet(self.C)
            - 0.5 * self.logdet(self.solve(K, self.C) + self.eye(K.shape[0]))
        )
        return norm

    def _weighted_data(self):
        """Return the weighted data component"""
        # TODO This can all be pre-computed
        if not hasattr(self, "weighted_data_CACHE"):
            dw = self.weighted_data_CACHE = (
                -0.5 * self.data.T @ self.solve(self.C, self.data)
            )
        else:
            dw = self.weighted_data_CACHE
        return dw

    def _weighted_model(self, mu, K):
        """Return the inner product of the GPR mean"""
        mu = torch.tensor(mu, device=self.device, dtype=torch.double)
        return -0.5 * mu.T @ self.solve(K, mu)

    def _weighted_cross(self, mu, K):
        a = self.solve(self.C, self.data) + self.solve(K, mu)
        b = self.inverse_C + self.inverse(K)
        return 0.5 * a.T @ self.solve(b, a)

    def log_likelihood(self, waveform):
        waveform_d = torch.tensor(waveform.data, device=self.device, dtype=torch.double)
        waveform_c = torch.tensor(
            waveform.covariance, device=self.device, dtype=torch.double
        )
        like = self._weighted_cross(waveform_d, waveform_c)
        # print("cross term", like)
        A = self._weighted_data()
        # print("data term", A)
        B = self._weighted_model(waveform_d, waveform_c)
        # print("model term", B)
        like = like + A + B
        norm = self._normalisation(waveform_c, self.C)
        # print("normalisation", norm)
        like += norm

        return like
