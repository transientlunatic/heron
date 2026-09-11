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
    k_smoothing_offsets : list of float or None
        Additional offsets (in the units of ``k_smoothing_param``, e.g. mass
        ratio) at which to re-evaluate the surrogate's variance and take the
        elementwise max with the value at the current point, before using it
        in the likelihood. Any GP's posterior variance is minimised at/near
        its own training inputs by construction, which imprints a spurious,
        sub-training-spacing oscillation onto K(theta) (period equal to the
        training grid spacing) that is an artefact of the discrete training
        grid, not a real feature of the model's uncertainty about theta. The
        marginalised likelihood's log-det term rewards this oscillation's
        dips regardless of whether the mean is actually more accurate there,
        biasing inference toward whichever training node is nearest. Probing
        a handful of nearby offsets and enveloping (max) removes the
        oscillation while preserving genuine, larger-scale variation of K
        with theta. Costs one extra `surrogate.predict()` call per offset,
        per likelihood evaluation. Default: disabled (matches prior
        behaviour exactly).
    k_smoothing_param : str
        Which surrogate parameter the offsets in ``k_smoothing_offsets`` are
        applied to. Defaults to ``'mass_ratio'`` — the only dimension with a
        discrete training grid in the checkpoints this has been tested on.
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
        k_smoothing_offsets: list[float] | None = None,
        k_smoothing_param: str = "mass_ratio",
    ):
        self.times = np.asarray(times, dtype=float)
        self.surrogate = surrogate
        self.detector = detector
        self.dtype = dtype
        self.device = torch.device(device)
        self.use_waveform_uncertainty = use_waveform_uncertainty
        self._f_low = f_low
        self._n = len(self.times)
        self._k_smoothing_offsets = list(k_smoothing_offsets) if k_smoothing_offsets else []
        self._k_smoothing_param = k_smoothing_param

        # High-pass filter mask (reused at every call).
        dt = float(self.times[1] - self.times[0])
        self._freqs = np.fft.rfftfreq(self._n, d=dt)
        self._hp_mask = self._freqs >= f_low
        self._P = self._build_projection_matrix()

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

    def _build_projection_matrix(self) -> np.ndarray:
        """The N×N linear operator implementing ``_hp_filter_1d``.

        ``P = irfft(mask * rfft(I))`` — applying ``_hp_filter_1d`` to every
        standard basis vector at once. Symmetric and idempotent (an
        orthogonal projector onto the passband subspace).
        """
        F = np.fft.rfft(np.eye(self._n), axis=0)
        F[~self._hp_mask, :] = 0.0
        return np.fft.irfft(F, n=self._n, axis=0)

    def _project_diag(self, var: np.ndarray) -> np.ndarray:
        """Band-limit a diagonal covariance ``diag(var)`` via ``P diag(var) P^T``.

        ``diag(var)`` is white (flat power at every frequency, including
        below ``f_low``), unlike the HP-filtered mean/data — even though
        ``var`` itself is a smooth, in-band function of time. Left unprojected,
        this white spectrum has full, generic overlap with the noise
        covariance's near-null sub-``f_low`` eigendirections (only kept
        positive-definite by ``jitter_rel``), inflating ``log|C+K|`` by an
        amount that tracks ``1/jitter_rel`` rather than anything physical —
        confirmed directly: the dominant eigenvalue of ``C^-1 K`` sits at
        0-28 Hz and scales exactly as ``1/jitter_rel``. Projecting first
        removes that sub-band content the same way the mean already is.

        Computed as ``B B^T`` with ``B = P @ diag(sqrt(var))`` (a Gram
        matrix), which is exactly PSD by construction — unlike projecting a
        *full*, non-diagonal K (the ``~6e-45`` negative eigenvalues noted
        below), a diagonal K's projection has no cross terms to go wrong;
        the two residual negative eigenvalues that do appear are pure
        float64 roundoff (~1e-16 relative to the real ones) and vanish
        against C's own regularisation once added into ``C + K``.
        """
        sqrt_var = np.sqrt(np.clip(var, 0.0, None))
        B = self._P * sqrt_var[None, :]
        return B @ B.T

    @staticmethod
    def _k_diagonal(K: np.ndarray) -> np.ndarray:
        """Return the raw per-sample variance diagonal of K (unprojected).

        Historically this was returned as ``diag(K.diagonal())`` directly.
        Using only the diagonal of the GP predictive covariance (rather than
        the full matrix) is the safest approach because:

        1. It is always PSD (positive diagonal elements).
        2. HP-filtering the full K matrix along both axes creates negative
           eigenvalues (~6e-45) because the filter removes the dominant
           low-frequency variance, leaving a high-frequency residual that is
           not guaranteed to be PSD.

        But the diagonal itself must still be passed through
        :meth:`_project_diag` before use — see its docstring for why leaving
        it unprojected silently reintroduces a large, spurious sub-``f_low``
        contribution to ``log|C+K|`` (harmless to the posterior *shape* in
        practice, since it is close to θ-independent, but corrupts
        ``log_evidence`` and is not a real feature of the model).
        """
        return K.diagonal()

    def _k_diagonal_envelope(
        self, surrogate_params: dict, fp: float, fc: float, center_K: np.ndarray,
    ) -> np.ndarray:
        """Elementwise-max the diagonal variance over nearby k_smoothing offsets.

        See ``k_smoothing_offsets`` in the class docstring for why this
        removes a spurious, grid-spacing-periodic oscillation in K(theta)
        rather than reflecting a real feature of the surrogate's uncertainty.

        When the surrogate exposes ``envelope_covariance_diagonal`` it computes
        the enveloped variance without re-evaluating the (potentially very
        expensive, e.g. LAL-backed) mean at each offset — the oscillation lives
        entirely in the GP variance, which is cheap kernel algebra. Otherwise
        we fall back to a full ``predict()`` per offset.

        Returns the raw (unprojected) variance diagonal — see
        :meth:`_project_diag`, which the caller applies afterwards.
        """
        surrogate = self.surrogate
        if hasattr(surrogate, "envelope_covariance_diagonal"):
            diags = surrogate.envelope_covariance_diagonal(
                surrogate_params, self._k_smoothing_offsets, self._k_smoothing_param,
            )
            return fp**2 * diags["plus"] + fc**2 * diags["cross"]

        # Fallback: works for any surrogate but pays the full predict() cost
        # (mean included) per offset.
        variances = [center_K.diagonal()]
        base = float(surrogate_params[self._k_smoothing_param])
        for offset in self._k_smoothing_offsets:
            p = dict(surrogate_params)
            p[self._k_smoothing_param] = base + offset
            wf = self.surrogate.predict(p)
            _, K_off = project_waveform(wf, fp, fc)
            variances.append(K_off.diagonal())
        return np.maximum.reduce(variances)

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
            # Diagonal approximation: ignore temporal correlations in K, then
            # band-limit that diagonal the same way mu/data already are (see
            # _project_diag) so it doesn't spuriously inflate log|C+K| via
            # the noise covariance's regularised sub-f_low null space.
            if self._k_smoothing_offsets:
                var = self._k_diagonal_envelope(surrogate_params, fp, fc, K)
            else:
                var = self._k_diagonal(K)
            K = self._project_diag(var)
        else:
            # Scalar 0.0 rather than an N×N zeros array: MarginalLogLikelihood
            # detects an all-zero K from a cheap count_nonzero and takes an
            # O(N²) fast path, so there's no reason to allocate a dense N×N
            # zero matrix just to represent "no K" — material at real-data N.
            K = 0.0

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
