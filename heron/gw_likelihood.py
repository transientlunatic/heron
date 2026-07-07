"""End-to-end GW marginal log-likelihood for time-domain parameter estimation.

Ties together:
  - GP surrogate (ExactGPSurrogate) evaluated at the correct relative times
  - Detector antenna pattern projection (F+, F×)
  - Noise covariance matrix (Wiener-Khinchin / PSD)
  - Marginal log-likelihood marginalised over the waveform uncertainty

Usage::

    gw_ll = GWLikelihood(
        data=strain,
        times=t_gps,
        psd_fn=aligo_design_psd,
        surrogate=model,
        detector="H1",
    )
    log_p = gw_ll({"mass_ratio": 0.8, "tc": 1187008882.43, "ra": 1.95,
                   "dec": -1.27, "psi": 0.82})
"""
from __future__ import annotations

import numpy as np
import torch

from heron.detector import antenna_patterns, project_waveform
from heron.likelihood import MarginalLogLikelihood
from heron.noise import noise_covariance


class GWLikelihood:
    """Marginal log-likelihood p(d | θ) for a single GW detector.

    The noise covariance C and its Cholesky factor are computed once at
    construction time.  Each call evaluates the surrogate at the
    merger-relative times implied by the current t_c, projects onto the
    detector, and evaluates log N(d; μ, C + K).

    Both the stored data and the per-call signal are high-pass filtered at
    ``f_low`` before computing the likelihood.  This is essential for
    correctness: sub-``f_low`` signal content is not constrained by the data
    (the noise model assigns it zero weight), but it inflates the time-domain
    Mahalanobis distance through the Toeplitz inverse.  Filtering removes
    that artefact and makes the time-domain SNR consistent with the
    standard frequency-domain matched-filter formula.

    Parameters
    ----------
    data : array_like, shape (N,)
        Observed strain time series.
    times : array_like, shape (N,)
        Uniformly-spaced GPS times corresponding to *data*.
    psd_fn : callable
        One-sided PSD [strain²/Hz] as a function of frequency [Hz].
    surrogate : WaveformSurrogate
        Trained GP surrogate (must accept ``parameters["times"]``).
    detector : str
        Detector name: ``'H1'``, ``'L1'``, or ``'V1'``.
    f_low : float
        Low-frequency cutoff in Hz.  Signal and data are both high-passed at
        this frequency; PSD contributions below it are zeroed in the noise
        covariance.
    f_high : float or None
        High-frequency cutoff; defaults to Nyquist.
    jitter : float
        Absolute diagonal regularisation (strain²) for the noise covariance.
    jitter_rel : float
        Relative diagonal regularisation (fraction of R(0)) for numerical SPD.
    use_waveform_uncertainty : bool
        If True (default), include the GP predictive covariance K in the
        marginalised likelihood.  If False, set K = 0, recovering the
        standard matched-filter likelihood.
    dtype : torch.dtype
        Floating-point precision for the Cholesky factor.
    device : str or torch.device
        Torch device for linear-algebra operations.
    """

    def __init__(
        self,
        data,
        times,
        psd_fn,
        surrogate,
        detector: str,
        f_low: float = 20.0,
        f_high: float | None = None,
        jitter: float = 0.0,
        jitter_rel: float = 1e-8,
        use_waveform_uncertainty: bool = True,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ):
        self.times = np.asarray(times, dtype=float)
        self.surrogate = surrogate
        self.detector = detector
        self.dtype = dtype
        self.device = torch.device(device)
        self.use_waveform_uncertainty = use_waveform_uncertainty
        self._f_low = f_low
        self._n = len(self.times)

        # High-pass filter mask (reused at every call).
        dt = float(self.times[1] - self.times[0])
        self._freqs = np.fft.rfftfreq(self._n, d=dt)
        self._hp_mask = self._freqs >= f_low

        # HP-filter the stored data so it matches the in-band signal.
        self.data = self._hp_filter_1d(np.asarray(data, dtype=float))

        # Pre-compute noise covariance and its Cholesky factor (O(N³), done once).
        C = noise_covariance(self.times, psd_fn, f_low=f_low, f_high=f_high,
                             jitter=jitter, jitter_rel=jitter_rel)
        self._C = C
        self._L_C = torch.linalg.cholesky(
            torch.as_tensor(C, dtype=dtype, device=self.device)
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _hp_filter_1d(self, x: np.ndarray) -> np.ndarray:
        """High-pass filter a 1-D time series at f_low."""
        x_f = np.fft.rfft(x)
        x_f[~self._hp_mask] = 0.0
        return np.fft.irfft(x_f, n=self._n)

    @staticmethod
    def _k_diagonal(K: np.ndarray) -> np.ndarray:
        """Return the diagonal approximation to K: diag(K.diagonal()).

        Using only the diagonal of the GP predictive covariance is the safest
        approach because:

        1. It is always PSD (positive diagonal elements).
        2. HP-filtering the full K matrix along both axes creates negative
           eigenvalues (~6e-45) because the filter removes the dominant
           low-frequency variance, leaving a high-frequency residual that is
           not guaranteed to be PSD.
        3. After HP-filtering both data and signal, the below-band residual
           r_below ≈ 0, so the below-band part of K contributes only a
           near-constant log-det term that cancels in the posterior.

        The diagonal approximation treats the per-sample GP uncertainty as
        independent across time, which is a recognised approximation that
        correctly captures the uncertainty magnitude and gives the right
        qualitative comparison between the marginalised and matched-filter
        likelihoods.
        """
        return np.diag(K.diagonal())

    # ------------------------------------------------------------------
    # Likelihood evaluation
    # ------------------------------------------------------------------

    def __call__(self, params: dict) -> float:
        """Return log p(d | θ).

        Parameters
        ----------
        params : dict
            Must contain:
            - ``'tc'``  — merger GPS time [s]
            - ``'ra'``  — right ascension [rad]
            - ``'dec'`` — declination [rad]
            - ``'psi'`` — polarisation angle [rad]
            - plus any parameters required by the surrogate (e.g. ``'mass_ratio'``)
        """
        tc = float(params["tc"])

        # Relative times: t_surrogate = t_GPS - t_c
        t_rel = self.times - tc

        # Evaluate surrogate at the merger-relative times.
        surrogate_params = {
            k: v for k, v in params.items() if k not in ("tc", "ra", "dec", "psi")
        }
        surrogate_params["times"] = t_rel
        wf = self.surrogate.predict(surrogate_params)

        # Project plus/cross onto the detector.
        fp, fc = antenna_patterns(
            float(params["ra"]), float(params["dec"]),
            float(params["psi"]), tc, self.detector,
        )
        mu, K = project_waveform(wf, fp, fc)

        # HP-filter the signal to match the stored (HP-filtered) data.
        mu = self._hp_filter_1d(mu)
        if self.use_waveform_uncertainty:
            # Diagonal approximation: ignore temporal correlations in K.
            # Full P K P^T filtering creates ~6e-45 negative eigenvalues because
            # the HP-projection zeros the dominant low-frequency variance in K,
            # leaving a residual not guaranteed to be PSD. Using diag(K) is always
            # PSD; the below-band quadratic term vanishes since r_below ≈ 0 after
            # filtering data and mu.
            K = self._k_diagonal(K)
        else:
            K = np.zeros_like(K)

        # Evaluate the marginal log-likelihood, reusing the pre-factored L_C.
        return MarginalLogLikelihood(
            C=None, mu=mu, K=K,
            dtype=self.dtype, device=self.device,
            _L_C=self._L_C,
        )(self.data)

    @property
    def noise_covariance(self) -> np.ndarray:
        """The pre-computed N×N noise covariance matrix."""
        return self._C
