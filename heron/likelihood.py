"""
This module contains likelihood functions for heron to allow it to perform parameter estimation
using the waveform models it supports.
"""

import numpy as np
import torch
from scipy import linalg as scipy_linalg

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

class NumericallyScaled:
    """
    Represent a number which has a numerical scaling applied to it.
    """
    def __init__(self, 
                 value: np.ndarray | torch.Tensor, 
                 scale: float | None = None):
        
        self.value = value
        self.scale = scale if scale is not None else 1./np.min(np.diag(value))

    def __call__(self):
        return self.scaled
    
    def __array__(self):
        return self.scaled
    
    def __sub__(self, other):
        assert self.scale == other.scale, "Cannot subtract NumericallyScaled with different scales"
        return self.scaled - other.scaled

    def __add__(self, other):
        assert self.scale == other.scale, "Cannot add NumericallyScaled with different scales"
        return self.scaled + other.scaled

    def unscale(self, value):
        return value / self.scale
    
    @property
    def scaled(self):
        return self.value * self.scale


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
        """
        A time-domain matched filtering likelihood function.

        Parameters
        ----------
        data : heron.timeseries.TimeSeries
            The time series data to be used in the likelihood.
        psd : heron.psd.PSD
            The PSD object used to compute the noise covariance matrix.
        waveform : heron.waveform.Waveform, optional
            The waveform model to be used in the likelihood. If not provided,
            it must be provided when calling the likelihood.
        detector : heron.detector.Detector, optional
            The detector to be used in the likelihood. If not provided,
            it must be provided when calling the likelihood.
        fixed_parameters : dict, optional
            A dictionary of parameters to be held fixed in the likelihood.
        timing_basis : str, optional
            The timing basis to be used (e.g., 'geocentre_time').

        Examples
        --------
        >>> from heron.timeseries import TimeSeries
        >>> from heron.psd import FlatPSD
        >>> from heron.waveform.lalsimulation import IMRPhenomPv2
        >>> from heron.detector import LIGOHanford
        >>> data = TimeSeries(...) 
        >>> psd = FlatPSD()
        >>> waveform = IMRPhenomPv2()
        >>> detector = LIGOHanford()
        >>> likelihood = TimeDomainLikelihood(data, psd, waveform, detector)
        """
        # Initialize logger first
        self.logger = logger = logging.getLogger(
            "heron.likelihood.TimeDomainLikelihood"
        )

        self.psd = psd
        self.timeseries = data
        self.data = self.array(data.data)
        self.times = data.times
        self.C = NumericallyScaled(self.psd.covariance_matrix(times=self.times))
        self.C_scaled = self.C.scaled
        self.data = NumericallyScaled(self.data.data, scale=np.sqrt(self.C.scale))
        self.data_scaled = self.data.scaled

        # Pre-compute and cache Cholesky decomposition for performance
        # This is done once at initialization rather than on every likelihood call
        try:
            self.C_cholesky = np.linalg.cholesky(self.C_scaled)
            self._use_cholesky = True
            self.logger.info(f"Cholesky decomposition cached for covariance matrix (N={len(self.times)})")
        except np.linalg.LinAlgError as e:
            self.logger.warning(f"Cholesky decomposition failed: {e}. Falling back to direct solve.")
            self._use_cholesky = False
            self.C_cholesky = None

        self.dt = (self.times[1] - self.times[0]).value
        self.N = len(self.times)

        if waveform is not None:
            self.waveform = waveform

        if detector is not None:
            self.detector = detector

        self.fixed_parameters = fixed_parameters
        if timing_basis is not None:
            self.fixed_parameters["reference_frame"] = timing_basis

    def snr(self, waveform):
        """
        Calculate the optimal signal to noise ratio for a given waveform.
        """
        dt = (self.times[1] - self.times[0]).value
        N = len(self.times)
        w = self.array(waveform.data)
        w = self.to_device(w, self.device)

        if self._use_cholesky:
            # Use cached Cholesky decomposition: solve L @ (L.T @ x) = w
            # First solve L @ y = w
            y = scipy_linalg.solve_triangular(self.C_cholesky, w, lower=True)
            # Then solve L.T @ x = y
            x = scipy_linalg.solve_triangular(self.C_cholesky.T, y, lower=False)
            h_h = w.T @ x
        else:
            # Fallback to direct solve
            h_h = (w.T @ self.solve(self.C_scaled, w))

        return np.sqrt(h_h)

    def log_likelihood(self, waveform, norm=True):
        w = self.timeseries.determine_overlap(self, waveform)
        if w is not None:
            (a,b) = w
        else:
            return -np.inf

        wf = NumericallyScaled(self.array(waveform.data[b[0]:b[1]]), scale=np.sqrt(self.C.scale))
        data = self.array(self.data_scaled[a[0]:a[1]])

        residual = self.to_device(data-wf, self.device)
        N = len(residual)

        C_scaled = self.C_scaled[a[0]:a[1], a[0]:a[1]]

        # Use Cholesky decomposition if available and applicable
        if self._use_cholesky and a[0] == 0 and a[1] == len(self.times):
            # Full overlap - use cached Cholesky decomposition
            # Solve L @ (L.T @ x) = residual using cached L
            y = scipy_linalg.solve_triangular(self.C_cholesky, residual, lower=True)
            x = scipy_linalg.solve_triangular(self.C_cholesky.T, y, lower=False)
            weighted_residual = residual @ x

            # For normalization, use cached log determinant: 2 * sum(log(diag(L)))
            if norm:
                logdet_C = 2.0 * np.sum(np.log(np.diag(self.C_cholesky)))
                normalisation = N * self.log(2*np.pi) + logdet_C - 2 * N * self.log(wf.scale)
            else:
                normalisation = 0
        else:
            # Partial overlap or Cholesky not available - use direct solve on submatrix
            # For partial overlaps, we need to extract the submatrix
            if self._use_cholesky and (a[0] != 0 or a[1] != len(self.times)):
                # Extract Cholesky factor for the submatrix
                # Note: This is still more efficient than full solve for small overlaps
                try:
                    L_sub = np.linalg.cholesky(C_scaled)
                    y = scipy_linalg.solve_triangular(L_sub, residual, lower=True)
                    x = scipy_linalg.solve_triangular(L_sub.T, y, lower=False)
                    weighted_residual = residual @ x

                    if norm:
                        logdet_C = 2.0 * np.sum(np.log(np.diag(L_sub)))
                        normalisation = N * self.log(2*np.pi) + logdet_C - 2 * N * self.log(wf.scale)
                    else:
                        normalisation = 0
                except np.linalg.LinAlgError:
                    # Fall back to direct solve if Cholesky fails on submatrix
                    weighted_residual = (residual) @ self.solve(C_scaled, residual)
                    normalisation = N * self.log(2*np.pi) + self.logdet(C_scaled) - 2 * N * self.log(wf.scale) if norm else 0
            else:
                # Direct solve fallback
                weighted_residual = (residual) @ self.solve(C_scaled, residual)
                normalisation = N * self.log(2*np.pi) + self.logdet(C_scaled) - 2 * N * self.log(wf.scale) if norm else 0

        return (- 0.5 * weighted_residual - 0.5 * normalisation)

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

    def __init__(self,
    data,
    psd,
    fixed_parameters={},
    timing_basis=None,
    waveform=None,
    detector=None):
        super().__init__(data, psd, waveform, detector, fixed_parameters=fixed_parameters, timing_basis=timing_basis)

        #self.norm_factor_2 = np.max(self.C)
        #self.norm_factor = np.sqrt(self.norm_factor_2)

    def log_likelihood(self, waveform, norm=True):
        a, b = self.timeseries.determine_overlap(self, waveform)

        wf = NumericallyScaled(self.to_device(self.array(waveform.data), self.device)[b[0]:b[1]])
        data = NumericallyScaled(self.data[a[0]:a[1]], scale=wf.scale)

        C = NumericallyScaled(self.C[a[0]:a[1], a[0]:a[1]], scale=wf.scale**2)
        K = NumericallyScaled(
            self.to_device(self.array(waveform.covariance[b[0]:b[1],b[0]:b[1]]), self.device), 
            scale=wf.scale**2)
        total_cov = C.scaled + K.scaled
        residual = self.to_device(self.array(data.scaled - wf.scaled), device=self.device)
        N_samp = len(residual)

        self.logger.debug(f"Data scale: {np.mean(np.abs(data.scaled))}")
        self.logger.debug(f"Residual scale: {np.mean(np.abs(residual))}")
        self.logger.debug(f"Cov diagonal range: [{np.min(np.diag(total_cov))}, {np.max(np.diag(total_cov))}]")
        self.logger.debug(f"Condition number: {np.linalg.cond(total_cov)}")

        W = (- 0.5 * self.solve((total_cov), residual) @ residual)

        N = (- 0.5 * N_samp*self.log((2*self.pi)) - 0.5 * self.logdet((C+K)) + N_samp * self.log(wf.scale)) if norm else 0

        return (W + N)


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
