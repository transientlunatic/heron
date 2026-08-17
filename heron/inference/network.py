"""Coherent multi-detector marginal log-likelihood.

:class:`NetworkLikelihood` generalises :class:`heron.gw_likelihood.GWLikelihood`
from one detector to a network.  The GP-marginalised log-likelihood factorises
across detectors (independent noise, coherent signal)::

    log p({d_k} | θ) = Σ_k  log N(d_k; μ_k(θ), C_k + K_k(θ))

Each detector contributes its own noise covariance ``C_k`` (from its PSD) and
sees the *same* intrinsic waveform, projected with its own antenna response and
delayed by its own geocentre time offset.  Distance, inclination and
coalescence phase are applied analytically at projection time
(:func:`heron.inference.projection.project_polarisations`), so the surrogate is
evaluated only at its reference distance/inclination — sampling those extrinsic
parameters costs no extra ``predict()`` call.

A single-detector ``NetworkLikelihood`` reproduces ``GWLikelihood`` numerically.

**Analytic phase marginalisation** (``marginalize_phase=True``): coalescence
phase enters the projected mean as a pure rotation of the plus/cross
quadratures (:func:`heron.inference.projection.project_polarisations`),
``mu(phi_c) = cos(2 phi_c) * A + sin(2 phi_c) * B`` for fixed A, B — the same
structure exploited by the standard GW phase-marginalised likelihood
(LALInference/bilby), exact for a dominant ``l=2`` (non-precessing,
single-harmonic) waveform, which is what every surrogate in this codebase is.
Phase couples the whole network coherently (it is one shared parameter, not
per-detector), so the marginalisation is done by summing a complex
"phase-domain SNR" ``P + iQ`` across detectors *before* the final
``log I0``, not by marginalising each detector separately — see
:meth:`NetworkLikelihood._log_likelihood_marginal_phase`.

This is **exact when** ``use_waveform_uncertainty=False``: ``Sigma = C`` alone
is stationary (built from a PSD), and ``A``, ``B`` are an exact quadrature pair
for the dominant harmonic, so the rotation preserves ``Sigma``-weighted norm
and orthogonality pointwise in frequency — no approximation beyond the
existing matched-filter treatment. It is an **approximation when**
``use_waveform_uncertainty=True``: the GP covariance ``K`` is diagonal in
*time*, not frequency (see ``heron/gw_likelihood.py``), so ``Sigma = C + K``
is no longer exactly stationary and the phase-independence of the quadratic
term is no longer exact. In that case ``K`` is evaluated once at a fixed
``coalescence_phase=0`` reference and held fixed across the (analytic) phase
integral — consistent with this codebase's existing K approximations
(diagonal K, k-smoothing envelopes), but unvalidated against nested-sampling
ground truth; treat as a working approximation, not a closed result.

Usage::

    from heron.inference.detectors import Detector
    from heron.inference.network import NetworkLikelihood

    like = NetworkLikelihood(
        data={"H1": d_h1, "L1": d_l1},
        times=times,
        detectors=[Detector.from_name("H1"), Detector.from_name("L1")],
        surrogate=model,
    )
    log_p = like({"mass_ratio": 0.8, "tc": 1187008882.43, "ra": 1.95,
                  "dec": -1.27, "psi": 0.82, "luminosity_distance": 400.0,
                  "inclination": 0.3, "coalescence_phase": 1.1})
"""
from __future__ import annotations

import inspect
import math

import numpy as np
import torch
from scipy.special import i0e

from heron.likelihood import MarginalLogLikelihood
from heron.noise import noise_covariance
from heron.inference.projection import project_polarisations, project_variances

_LOG_2PI = math.log(2.0 * math.pi)
_QUARTER_TURN = math.pi / 4.0


# Parameter names consumed by the likelihood/projection layer rather than passed
# to the surrogate.  Everything else in ``params`` (mass_ratio, total_mass, …)
# is intrinsic and forwarded to ``surrogate.predict``.
_EXTRINSIC = frozenset({
    "tc", "geocent_time",
    "ra", "dec", "psi",
    "inclination", "theta_jn",
    "coalescence_phase", "phase",
    "luminosity_distance", "distance",
})


def _waveform_variance(waveform) -> np.ndarray:
    """Diagonal variance of a Waveform, or zeros if it has no covariance."""
    var = waveform.variance
    return np.zeros_like(waveform.data) if var is None else np.asarray(var, dtype=float)


class _DetectorChannel:
    """Per-detector precomputed state: HP-filtered data + noise Cholesky."""

    __slots__ = ("detector", "data", "L_C", "C")

    def __init__(self, detector, data, L_C, C):
        self.detector = detector
        self.data = data
        self.L_C = L_C
        self.C = C


class NetworkLikelihood:
    """Marginal log-likelihood ``Σ_k log N(d_k; μ_k, C_k + K_k)`` over a network.

    Parameters
    ----------
    data : dict[str, ndarray] or ndarray
        Observed strain per detector, keyed by detector prefix.  A bare array is
        accepted when *detectors* is a single detector.
    times : array_like, shape (N,)
        Uniformly-spaced GPS times, shared by all detectors.
    detectors : list[Detector] or Detector
        The detector network.
    surrogate : WaveformSurrogate
        Trained GP surrogate.
    f_low, f_high : float
        Band-pass edges (Hz).  Signal and data are high-passed at ``f_low``;
        PSD below it is zeroed in each ``C_k``.
    jitter, jitter_rel : float
        Diagonal regularisation for the noise covariances.
    use_waveform_uncertainty : bool
        Include the GP predictive covariance K (default) or set K = 0 (matched
        filter).
    dtype, device : torch dtype / device
        Precision and device for the linear algebra.
    k_smoothing_offsets : list[float] or None
        Mass-ratio (or ``k_smoothing_param``) offsets over which to envelope the
        GP variance, removing the training-grid-periodic dip in K(θ).  See
        :class:`heron.gw_likelihood.GWLikelihood`.  Default: disabled.
    k_smoothing_param : str
        Parameter the offsets apply to (default ``'mass_ratio'``).
    marginalize_phase : bool
        Analytically marginalise coalescence phase instead of taking it as a
        sampled parameter (default ``False``, i.e. current behaviour). See the
        module docstring for the exact/approximate distinction. When enabled,
        ``params`` passed to ``__call__`` must NOT contain
        ``coalescence_phase``/``phase`` (raises ``ValueError`` if present).
    """

    def __init__(
        self,
        data,
        times,
        detectors,
        surrogate,
        f_low: float = 20.0,
        f_high: float | None = None,
        jitter: float = 0.0,
        jitter_rel: float = 1e-8,
        use_waveform_uncertainty: bool = True,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
        k_smoothing_offsets: list[float] | None = None,
        k_smoothing_param: str = "mass_ratio",
        marginalize_phase: bool = False,
    ):
        # Normalise detectors and data into aligned lists keyed by prefix.
        if not isinstance(detectors, (list, tuple)):
            detectors = [detectors]
        if not isinstance(data, dict):
            if len(detectors) != 1:
                raise ValueError(
                    "data must be a {prefix: array} dict for multi-detector networks"
                )
            data = {detectors[0].prefix: data}

        self.times = np.asarray(times, dtype=float)
        self.surrogate = surrogate
        self.dtype = dtype
        self.device = torch.device(device)
        self.use_waveform_uncertainty = use_waveform_uncertainty
        self._f_low = f_low
        self._n = len(self.times)
        self._k_smoothing_offsets = list(k_smoothing_offsets) if k_smoothing_offsets else []
        self._k_smoothing_param = k_smoothing_param
        self._distance_ref = getattr(surrogate, "distance_factor", None)
        self._marginalize_phase = marginalize_phase

        # Only the diagonal of K is ever used (project_variances). If the
        # surrogate's predict() accepts a `covariance` mode, request the cheap
        # one: 'diagonal' (per-sample variance, no N×N matrix) with K, or 'none'
        # (mean only) without K / when the variance comes from the k-smoothing
        # envelope. Surrogates without the kwarg fall back to the full path.
        try:
            self._predict_cov_kw = (
                "covariance" in inspect.signature(surrogate.predict).parameters
            )
        except (ValueError, TypeError):
            self._predict_cov_kw = False
        self._has_envelope = hasattr(surrogate, "envelope_covariance_diagonal")

        # Shared high-pass mask (all detectors share the time grid).
        dt = float(self.times[1] - self.times[0])
        self._freqs = np.fft.rfftfreq(self._n, d=dt)
        self._hp_mask = self._freqs >= f_low
        self._P = self._build_projection_matrix()

        self._channels: list[_DetectorChannel] = []
        for det in detectors:
            if det.prefix not in data:
                raise KeyError(f"no data supplied for detector {det.prefix}")
            C = noise_covariance(
                self.times, det.psd_fn, f_low=f_low, f_high=f_high,
                jitter=jitter, jitter_rel=jitter_rel,
            )
            L_C = torch.linalg.cholesky(
                torch.as_tensor(C, dtype=dtype, device=self.device)
            )
            d = self._hp_filter(np.asarray(data[det.prefix], dtype=float))
            self._channels.append(_DetectorChannel(det, d, L_C, C))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _hp_filter(self, x: np.ndarray) -> np.ndarray:
        """High-pass filter a 1-D time series at f_low."""
        x_f = np.fft.rfft(x)
        x_f[~self._hp_mask] = 0.0
        return np.fft.irfft(x_f, n=self._n)

    def _build_projection_matrix(self) -> np.ndarray:
        """The N×N linear operator implementing ``_hp_filter``.

        ``P = irfft(mask * rfft(I))`` — applying ``_hp_filter`` to every
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
        ``var`` itself is a smooth, in-band function of time. Left
        unprojected, this white spectrum has full, generic overlap with the
        noise covariance's near-null sub-``f_low`` eigendirections (only kept
        positive-definite by ``jitter_rel``), inflating ``log|C+K|`` by an
        amount that tracks ``1/jitter_rel`` rather than anything physical —
        confirmed directly on the demod checkpoint: the dominant eigenvalue
        of ``C^-1 K`` sat at 0-28 Hz and scaled exactly as ``1/jitter_rel``
        (up to ~4e7x C at SNR~250), inflating logZ by 100+ nats while barely
        touching the posterior (the residual, not K, carries the θ-shape,
        and it is already HP-filtered). Projecting first removes that
        sub-band content the same way the mean already is.

        Computed as ``B B^T`` with ``B = P @ diag(sqrt(var))`` (a Gram
        matrix), which is exactly PSD by construction — unlike projecting a
        full, non-diagonal K, a diagonal K's projection has no cross terms
        to go wrong; the residual negative eigenvalues that do appear are
        pure float64 roundoff (~1e-16 relative to the real ones) and vanish
        against C's own regularisation once added into ``C + K``.
        """
        sqrt_var = np.sqrt(np.clip(var, 0.0, None))
        B = self._P * sqrt_var[None, :]
        return B @ B.T

    def _predict(self, params: dict, mode: str):
        """Call the surrogate, requesting covariance ``mode`` when supported.

        ``mode`` is one of ``'full'`` / ``'diagonal'`` / ``'none'``. Surrogates
        whose ``predict`` lacks the ``covariance`` kwarg get the full path (the
        resulting Waveform still exposes ``.variance``, so consumers are
        unaffected — only the cost is higher).
        """
        if self._predict_cov_kw:
            return self.surrogate.predict(params, covariance=mode)
        return self.surrogate.predict(params)

    @staticmethod
    def _split_params(params: dict) -> tuple[dict, dict]:
        """Return (intrinsic surrogate params, resolved extrinsic params)."""
        intrinsic = {k: v for k, v in params.items() if k not in _EXTRINSIC}
        extr = {
            "tc": float(params.get("tc", params.get("geocent_time"))),
            "ra": float(params["ra"]),
            "dec": float(params["dec"]),
            "psi": float(params["psi"]),
            "inclination": float(params.get("inclination", params.get("theta_jn", 0.0))),
            "coalescence_phase": float(
                params.get("coalescence_phase", params.get("phase", 0.0))
            ),
            "distance": params.get("luminosity_distance", params.get("distance", None)),
        }
        if extr["distance"] is not None:
            extr["distance"] = float(extr["distance"])
        return intrinsic, extr

    def _enveloped_variances(
        self, wf, surrogate_params: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Elementwise-max plus/cross variance over the k_smoothing offsets.

        Reuses the surrogate's cheap ``envelope_covariance_diagonal`` (variance
        only, no re-run of an expensive mean) when available; otherwise falls
        back to a full ``predict()`` per offset.
        """
        surrogate = self.surrogate
        if hasattr(surrogate, "envelope_covariance_diagonal"):
            diags = surrogate.envelope_covariance_diagonal(
                surrogate_params, self._k_smoothing_offsets, self._k_smoothing_param,
            )
            return np.asarray(diags["plus"], float), np.asarray(diags["cross"], float)

        var_p = _waveform_variance(wf["plus"])
        var_c = _waveform_variance(wf["cross"])
        base = float(surrogate_params[self._k_smoothing_param])
        for offset in self._k_smoothing_offsets:
            p = dict(surrogate_params)
            p[self._k_smoothing_param] = base + offset
            wf_off = self._predict(p, "diagonal")
            var_p = np.maximum(var_p, _waveform_variance(wf_off["plus"]))
            var_c = np.maximum(var_c, _waveform_variance(wf_off["cross"]))
        return var_p, var_c

    # ------------------------------------------------------------------
    # Likelihood evaluation
    # ------------------------------------------------------------------

    def _predict_mode(self) -> str:
        """Cheapest ``predict()`` covariance mode for the current settings.

        K's diagonal is all the likelihood ever uses, so the full N×N is never
        requested: 'none' (mean only) without K or when the variance comes from
        the k-smoothing envelope instead, else 'diagonal'.
        """
        if not self.use_waveform_uncertainty:
            return "none"
        if self._k_smoothing_offsets and self._has_envelope:
            return "none"
        return "diagonal"

    def __call__(self, params: dict) -> float:
        """Return the network log-likelihood ``Σ_k log p(d_k | θ)``."""
        if self._marginalize_phase and (
            "coalescence_phase" in params or "phase" in params
        ):
            raise ValueError(
                "coalescence_phase is analytically marginalised "
                "(marginalize_phase=True); do not include it in params."
            )
        intrinsic, extr = self._split_params(params)
        if self._marginalize_phase:
            return self._log_likelihood_marginal_phase(intrinsic, extr)
        return self._log_likelihood_fixed_phase(intrinsic, extr)

    def _log_likelihood_fixed_phase(self, intrinsic: dict, extr: dict) -> float:
        tc = extr["tc"]
        total = 0.0
        predict_mode = self._predict_mode()

        for ch in self._channels:
            det = ch.detector
            dt_geo = det.time_delay_from_geocentre(extr["ra"], extr["dec"], tc)
            t_rel = self.times - (tc + dt_geo)

            surrogate_params = {**intrinsic, "times": t_rel}
            wf = self._predict(surrogate_params, predict_mode)

            fp, fc = det.antenna_patterns(extr["ra"], extr["dec"], extr["psi"], tc)
            mu, k_diag = project_polarisations(
                wf, f_plus=fp, f_cross=fc,
                distance=extr["distance"], distance_ref=self._distance_ref,
                inclination=extr["inclination"],
                coalescence_phase=extr["coalescence_phase"],
            )
            mu = self._hp_filter(mu)

            if self.use_waveform_uncertainty:
                if self._k_smoothing_offsets:
                    var_p, var_c = self._enveloped_variances(wf, surrogate_params)
                    k_diag = project_variances(
                        var_p, var_c, f_plus=fp, f_cross=fc,
                        distance=extr["distance"], distance_ref=self._distance_ref,
                        inclination=extr["inclination"],
                        coalescence_phase=extr["coalescence_phase"],
                    )
                K = self._project_diag(k_diag)
            else:
                K = np.zeros((self._n, self._n))

            total += MarginalLogLikelihood(
                C=None, mu=mu, K=K, dtype=self.dtype, device=self.device,
                _L_C=ch.L_C,
            )(ch.data)

        return float(total)

    def _log_likelihood_marginal_phase(self, intrinsic: dict, extr: dict) -> float:
        """Coherent network log-likelihood with coalescence phase marginalised.

        ``mu(phi_c) = cos(2 phi_c) * A + sin(2 phi_c) * B`` per detector, for
        fixed A (mean at ``phi_c=0``) and B (mean at ``phi_c=pi/4``, where
        ``2 phi_c = pi/2`` makes ``cos=0, sin=1`` so the projection returns B
        directly) — reusing :func:`~heron.inference.projection.project_polarisations`
        twice avoids re-deriving the inclination/antenna/distance chain
        algebraically. K is evaluated once at the ``phi_c=0`` reference and
        held fixed (see the module docstring for the exact/approximate split).

        Phase is one parameter shared by the whole network, so detectors are
        combined *before* the final nonlinearity: each contributes a complex
        "phase-domain SNR" ``P_k + i Q_k`` (``P_k = A_k^T Sigma_k^-1 d_k``,
        ``Q_k = B_k^T Sigma_k^-1 d_k``), these sum coherently across detectors,
        and the phase integral of the Gaussian likelihood becomes the standard
        von Mises normalisation ``log I0(|sum_k (P_k + i Q_k)|)`` (computed via
        ``scipy.special.i0e`` for numerical stability at large argument:
        ``log I0(x) = x + log(i0e(x))``).
        """
        tc = extr["tc"]
        predict_mode = self._predict_mode()

        const_total = 0.0
        p_total = 0.0
        q_total = 0.0

        for ch in self._channels:
            det = ch.detector
            dt_geo = det.time_delay_from_geocentre(extr["ra"], extr["dec"], tc)
            t_rel = self.times - (tc + dt_geo)

            surrogate_params = {**intrinsic, "times": t_rel}
            wf = self._predict(surrogate_params, predict_mode)

            fp, fc = det.antenna_patterns(extr["ra"], extr["dec"], extr["psi"], tc)
            mu_a, k_diag = project_polarisations(
                wf, f_plus=fp, f_cross=fc,
                distance=extr["distance"], distance_ref=self._distance_ref,
                inclination=extr["inclination"], coalescence_phase=0.0,
            )
            mu_q, _ = project_polarisations(
                wf, f_plus=fp, f_cross=fc,
                distance=extr["distance"], distance_ref=self._distance_ref,
                inclination=extr["inclination"], coalescence_phase=_QUARTER_TURN,
            )
            mu_a = self._hp_filter(mu_a)
            mu_b = self._hp_filter(mu_q)

            if self.use_waveform_uncertainty:
                if self._k_smoothing_offsets:
                    var_p, var_c = self._enveloped_variances(wf, surrogate_params)
                    k_diag = project_variances(
                        var_p, var_c, f_plus=fp, f_cross=fc,
                        distance=extr["distance"], distance_ref=self._distance_ref,
                        inclination=extr["inclination"], coalescence_phase=0.0,
                    )
                K = self._project_diag(k_diag)
            else:
                K = np.zeros((self._n, self._n))

            mll = MarginalLogLikelihood(
                C=None, mu=mu_a, K=K, dtype=self.dtype, device=self.device,
                _L_C=ch.L_C,
            )
            u_a = mll.whiten(mu_a)
            u_b = mll.whiten(mu_b)
            u_d = mll.whiten(ch.data)

            aa = float(torch.dot(u_a, u_a))
            dd = float(torch.dot(u_d, u_d))
            p_total += float(torch.dot(u_a, u_d))
            q_total += float(torch.dot(u_b, u_d))
            const_total += -0.5 * (dd + aa) - 0.5 * mll.log_det - 0.5 * mll.n * _LOG_2PI

        r = math.hypot(p_total, q_total)
        return float(const_total + r + math.log(i0e(r)))

    @property
    def noise_covariances(self) -> dict[str, np.ndarray]:
        """The pre-computed noise covariance per detector prefix."""
        return {ch.detector.prefix: ch.C for ch in self._channels}
