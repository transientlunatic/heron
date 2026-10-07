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
import warnings

import numpy as np
import torch
from scipy.special import i0e, logsumexp

from heron.likelihood import MarginalLogLikelihood
from heron.noise import noise_covariance
from heron.stationary import StationaryMarginalLikelihood, StationaryNoise, correlate
from heron.inference.projection import (
    project_polarisations, project_variances, variance_window,
)

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
    """Per-detector precomputed state: HP-filtered data + noise model.

    ``linalg="dense"`` fills ``L_C``/``C`` (dense Cholesky); ``"stationary"``
    fills ``noise`` (a :class:`~heron.stationary.StationaryNoise`) instead.
    """

    __slots__ = ("detector", "data", "L_C", "C", "noise")

    def __init__(self, detector, data, L_C=None, C=None, noise=None):
        self.detector = detector
        self.data = data
        self.L_C = L_C
        self.C = C
        self.noise = noise


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
    marginalize_time : bool
        Marginalise geocentre coalescence time over the uniform prior
        ``time_prior`` instead of sampling it (default ``False``). Needs
        ``linalg="stationary"``. Because ``C`` and ``P`` are circulant they
        commute with time shifts, and the template and its (tapered) variance
        move rigidly with ``tc``; so one surrogate evaluation at the centre of
        the prior gives the likelihood at every sample-grid shift in the prior
        exactly: a fixed log-determinant, the cross term by FFT
        cross-correlation, and the data term by one batched solve. The
        marginal is the average over that grid. Shifts are cyclic, so the
        template and its variance must be negligible within
        ``max|tc - centre|`` of the segment ends (the variance taper and the
        usual data taper see to this). Antenna patterns and detector delays
        are evaluated at the prior centre. ``params`` must not contain ``tc``
        / ``geocent_time``. Combines with ``marginalize_phase``. See
        :meth:`time_series` for reconstructing ``tc`` posterior samples.
    time_prior : (float, float) or None
        Geocentre ``tc`` prior bounds (GPS seconds) for ``marginalize_time``.
    time_resolution : float
        Spacing (seconds) of the ``tc`` grid used for time marginalisation,
        default 50 µs. The ``tc`` likelihood peak is ~1/(2π B ρ) wide (≈0.1 ms
        at SNR ~20), far narrower than a sample at typical rates, so the grid
        is refined below the sample spacing: the cross term (which carries the
        sharp ``tc`` dependence) by zero-padded FFT and the data-only term by
        phase-ramp-shifting the whitened data, i.e. both use the exact
        band-limited fractional shift. This requires the template to be
        band-limited at the sample rate: sampling a waveform with power above
        Nyquist aliases it, the fractional shift of the aliased samples is not
        the shifted waveform, and the result can be off by tens of nats (seen
        at 512 Hz for a 60 Msun merger, which has ~1% of its power above
        256 Hz). A warning is issued when the template carries significant
        power in the top of the band. Set to the sample spacing to disable
        refinement (whole-sample shifts are exact regardless).
    covariance_inflation : float
        Scalar multiplier applied to K's variance (not the mean) wherever it
        is used, after any k-smoothing envelope. Default 1.0 (no-op). Unlike
        :class:`~heron.models.gp.demod.DemodGPSurrogate`'s own attribute of
        the same name, this is a likelihood-level, experiment-only knob —
        not part of the surrogate's calibration and never serialized with a
        checkpoint — for synthetically studying "what would with-K vs no-K
        look like if K were N times larger," e.g. on representations/points
        where the surrogate's own (well-calibrated) K is too small relative
        to C for the effect to show up naturally at any realistic SNR.
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
        marginalize_time: bool = False,
        time_prior: tuple[float, float] | None = None,
        time_resolution: float = 5e-5,
        covariance_inflation: float = 1.0,
        data_taper: float | None = None,
        variance_taper: float | None = 0.02,
        linalg: str = "stationary",
        linalg_options: dict | None = None,
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
        self._covariance_inflation = float(covariance_inflation)
        if linalg not in ("stationary", "dense"):
            raise ValueError(f"linalg must be 'stationary' or 'dense'; got {linalg!r}")
        self._linalg = linalg
        self._variance_taper = None if variance_taper is None else float(variance_taper)
        self._linalg_options = dict(linalg_options or {})
        self._noise_args = dict(f_low=f_low, f_high=f_high,
                                jitter=jitter, jitter_rel=jitter_rel)
        self._data_window = None
        if data_taper is not None:
            from heron.inference.strain import tukey_window
            dt = float(self.times[1] - self.times[0])
            self._data_window = tukey_window(len(self.times), dt, float(data_taper))
        self._marginalize_time = marginalize_time
        self._taper_warned = False
        self._alias_warned = False
        if marginalize_time:
            if linalg != "stationary":
                raise ValueError("marginalize_time requires linalg='stationary'")
            if time_prior is None:
                raise ValueError("marginalize_time requires time_prior=(tc_min, tc_max)")
            lo, hi = map(float, time_prior)
            dt = float(self.times[1] - self.times[0])
            self._oversample = max(1, int(np.ceil(dt / float(time_resolution) - 1e-9)))
            step = dt / self._oversample
            half = int(np.floor(0.5 * (hi - lo) / step))
            self._tc_centre = 0.5 * (lo + hi)
            self._shifts = np.arange(-half, half + 1)            # in units of `step`
            self._tc_grid = self._tc_centre + self._shifts * step

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
        # The dense N x N projector is only needed by the dense path.
        self._P = self._build_projection_matrix() if linalg == "dense" else None

        self._channels: list[_DetectorChannel] = []
        for det in detectors:
            if det.prefix not in data:
                raise KeyError(f"no data supplied for detector {det.prefix}")
            d = self._hp_filter(np.asarray(data[det.prefix], dtype=float))
            if linalg == "stationary":
                noise = StationaryNoise(
                    self.times, det.psd_fn, dtype=dtype, device=self.device,
                    **self._noise_args,
                )
                self._channels.append(_DetectorChannel(det, d, noise=noise))
                continue
            C = noise_covariance(self.times, det.psd_fn, **self._noise_args)
            L_C = torch.linalg.cholesky(
                torch.as_tensor(C, dtype=dtype, device=self.device)
            )
            self._channels.append(_DetectorChannel(det, d, L_C=L_C, C=C))

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

    def _taper(self, intrinsic: dict, t_rel: np.ndarray, k_diag):
        """Apply the out-of-training-window variance taper (if enabled)."""
        bounds_fn = getattr(self.surrogate, "training_time_bounds", None)
        if k_diag is None or self._variance_taper is None or bounds_fn is None:
            return k_diag
        bounds = bounds_fn(intrinsic)
        if bounds is None:
            return k_diag
        return k_diag * variance_window(t_rel, bounds, self._variance_taper)

    def _marginal(self, ch: _DetectorChannel, mu: np.ndarray, k_diag):
        """Per-detector ``log N(., mu, C + P diag(k_diag) P)`` evaluator.

        ``k_diag`` is the raw (unprojected, uninflated) variance, or ``None``
        for K = 0.  Returns an object with ``__call__(d)`` and ``inner(x, y)``.
        """
        if k_diag is not None:
            k_diag = self._covariance_inflation * k_diag
        if self._linalg == "stationary":
            return StationaryMarginalLikelihood(
                ch.noise, mu, k_diag, **self._linalg_options
            )
        # Scalar 0.0 rather than an N×N zeros array: MarginalLogLikelihood
        # detects an all-zero K from a cheap count_nonzero and takes an O(N²)
        # fast path (a dense N×N zero matrix is multi-GB at real-data N).
        K = 0.0 if k_diag is None else self._project_diag(k_diag)
        return _DenseMarginal(MarginalLogLikelihood(
            C=None, mu=mu, K=K, dtype=self.dtype, device=self.device, _L_C=ch.L_C,
        ))

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
        if self._marginalize_time:
            series = self._time_series(params)
            return float(logsumexp(series) - math.log(len(series)))
        intrinsic, extr = self._split_params(params)
        if self._marginalize_phase:
            return self._log_likelihood_marginal_phase(intrinsic, extr)
        return self._log_likelihood_fixed_phase(intrinsic, extr)

    def time_series(self, params: dict) -> tuple[np.ndarray, np.ndarray]:
        """``(tc_grid, log L(tc))`` over the time prior grid (``marginalize_time``).

        ``log L`` is the (phase-marginalised, if enabled) log-likelihood at each
        geocentre ``tc`` on the sample grid. Draw ``tc`` posterior samples for a
        posterior point by sampling this grid with weights ``exp(log L)``.
        """
        if not self._marginalize_time:
            raise ValueError("time_series needs marginalize_time=True")
        return self._tc_grid.copy(), self._time_series(params)

    def _time_series(self, params: dict) -> np.ndarray:
        if "tc" in params or "geocent_time" in params:
            raise ValueError(
                "tc is analytically marginalised (marginalize_time=True); "
                "do not include it in params."
            )
        if self._marginalize_phase and ("coalescence_phase" in params or "phase" in params):
            raise ValueError(
                "coalescence_phase is analytically marginalised "
                "(marginalize_phase=True); do not include it in params."
            )
        intrinsic, extr = self._split_params({**params, "tc": self._tc_centre})
        shifts, u = self._shifts, self._oversample
        const = np.zeros(len(shifts))
        p_tot = np.zeros(len(shifts))
        q_tot = np.zeros(len(shifts))
        phases = [0.0, _QUARTER_TURN] if self._marginalize_phase else [extr["coalescence_phase"]]

        for ch in self._channels:
            mus, k_diag = self._detector_model(ch, intrinsic, extr, phases)
            if u > 1 and not self._alias_warned:
                self._check_band_limited(ch, mus[0])
            if self._data_window is not None and not self._taper_warned:
                self._check_taper_edges(ch, mus[0])
            mll = self._marginal(ch, mus[0], k_diag)
            d = ch.noise.tensor(ch.data)
            z_a = mll.solve(mus[0])
            # (S_k^T d - mu)^T Sigma^-1 (S_k^T d - mu) = D_k - 2 X_k + mu^T Sigma^-1 mu
            dd = mll.shifted_quadratic(d, shifts, oversample=u).cpu().numpy()
            xa = correlate(z_a, d, shifts, oversample=u).cpu().numpy()
            aa = float(torch.dot(ch.noise.tensor(mus[0]), z_a))
            base = -0.5 * mll.log_det - 0.5 * self._n * _LOG_2PI
            if self._marginalize_phase:
                z_b = mll.solve(mus[1])
                p_tot += xa
                q_tot += correlate(z_b, d, shifts, oversample=u).cpu().numpy()
                const += base - 0.5 * (dd + aa)
            else:
                const += base - 0.5 * (dd - 2.0 * xa + aa)

        if self._marginalize_phase:
            r = np.hypot(p_tot, q_tot)
            return const + r + np.log(i0e(r))
        return const

    def _check_taper_edges(self, ch, mu: np.ndarray, threshold: float = 1e-4) -> None:
        """Warn (once) if the windowed template has power where the window < 1.

        With ``marginalize_time`` the data window stays on the sample grid
        while the template shifts, so the shifted-template algebra is only
        exact where the window is flat.
        """
        edge = self._data_window < 1.0 - 1e-12
        shift = int(np.max(np.abs(self._shifts))) // self._oversample + 1
        if 2 * shift + 1 >= len(edge):
            edge = np.ones_like(edge, dtype=bool)
        else:
            edge = np.convolve(edge.astype(float), np.ones(2 * shift + 1), "same") > 0
        mu_edge = np.where(edge, mu, 0.0)
        g = ch.noise.g_spectrum()
        num = float((torch.abs(torch.fft.rfft(ch.noise.tensor(mu_edge))) ** 2 * g).sum())
        den = float((torch.abs(torch.fft.rfft(ch.noise.tensor(mu))) ** 2 * g).sum())
        if den > 0 and num / den > threshold:
            self._taper_warned = True
            warnings.warn(
                f"{num / den:.1e} of the whitened template power lies within the data "
                "taper's roll-off (widened by the time prior): time marginalisation "
                "holds the window fixed while the template shifts, so it is approximate "
                "here. Use a longer segment or a shorter roll-off.",
                stacklevel=3,
            )

    def _check_band_limited(self, ch, mu: np.ndarray, threshold: float = 1e-4) -> None:
        """Warn (once) if the whitened template has significant power near Nyquist.

        Sub-sample time marginalisation shifts the sampled template by a
        band-limited phase ramp, which is only the shifted waveform if the
        sampling did not alias it.  Power in the top 10% of the band is the
        tell-tale.
        """
        power = (torch.abs(torch.fft.rfft(ch.noise.tensor(mu))) ** 2
                 * ch.noise.g_spectrum()).cpu().numpy()
        top = self._freqs >= 0.9 * self._freqs[-1]
        frac = float(power[top].sum() / max(power.sum(), 1e-300))
        if frac > threshold:
            self._alias_warned = True
            warnings.warn(
                f"{frac:.1e} of the whitened template power is within 10% of Nyquist "
                f"({self._freqs[-1]:.0f} Hz): the template is probably aliased at this "
                "sample rate, and sub-sample time marginalisation can then be wrong "
                "by many nats. Raise the sample rate, or set time_resolution to the "
                "sample spacing.",
                stacklevel=3,
            )

    def _detector_model(self, ch, intrinsic: dict, extr: dict, phases: list[float]):
        """Projected, high-passed mean(s) and tapered variance for one detector.

        Returns ``([mu(phase) for phase in phases], k_diag)`` with ``k_diag``
        evaluated at ``phases[0]`` (``None`` without waveform uncertainty).
        """
        tc = extr["tc"]
        dt_geo = ch.detector.time_delay_from_geocentre(extr["ra"], extr["dec"], tc)
        t_rel = self.times - (tc + dt_geo)
        surrogate_params = {**intrinsic, "times": t_rel}
        wf = self._predict(surrogate_params, self._predict_mode())
        fp, fc = ch.detector.antenna_patterns(extr["ra"], extr["dec"], extr["psi"], tc)
        proj = dict(f_plus=fp, f_cross=fc, distance=extr["distance"],
                    distance_ref=self._distance_ref, inclination=extr["inclination"])
        win = self._data_window
        mus = []
        k_diag = None
        for i, phase in enumerate(phases):
            mu, kd = project_polarisations(wf, coalescence_phase=phase, **proj)
            if win is not None:
                mu = win * mu
            mus.append(self._hp_filter(mu))
            if i == 0:
                k_diag = kd
        if self.use_waveform_uncertainty:
            if self._k_smoothing_offsets:
                var_p, var_c = self._enveloped_variances(wf, surrogate_params)
                k_diag = project_variances(var_p, var_c, coalescence_phase=phases[0], **proj)
        else:
            k_diag = None
        k_diag = self._taper(intrinsic, t_rel, k_diag)
        if win is not None and k_diag is not None:
            k_diag = win**2 * k_diag
        return mus, k_diag

    def _log_likelihood_fixed_phase(self, intrinsic: dict, extr: dict) -> float:
        total = 0.0
        for ch in self._channels:
            mus, k_diag = self._detector_model(ch, intrinsic, extr, [extr["coalescence_phase"]])
            total += self._marginal(ch, mus[0], k_diag)(ch.data)
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
        const_total = 0.0
        p_total = 0.0
        q_total = 0.0

        for ch in self._channels:
            (mu_a, mu_b), k_diag = self._detector_model(
                ch, intrinsic, extr, [0.0, _QUARTER_TURN])
            mll = self._marginal(ch, mu_a, k_diag)
            aa = mll.inner(mu_a, mu_a)
            dd = mll.inner(ch.data, ch.data)
            p_total += mll.inner(mu_a, ch.data)
            q_total += mll.inner(mu_b, ch.data)
            const_total += -0.5 * (dd + aa) - 0.5 * mll.log_det - 0.5 * self._n * _LOG_2PI

        r = math.hypot(p_total, q_total)
        return float(const_total + r + math.log(i0e(r)))

    @property
    def noise_covariances(self) -> dict[str, np.ndarray]:
        """The noise covariance per detector prefix (dense; built on demand
        for the stationary path, so avoid at large N)."""
        return {
            ch.detector.prefix: ch.C if ch.C is not None else ch.noise.dense_covariance()
            for ch in self._channels
        }


class _DenseMarginal:
    """Adapts :class:`MarginalLogLikelihood` to the ``inner``/``log_det``
    interface shared with :class:`~heron.stationary.StationaryMarginalLikelihood`."""

    def __init__(self, mll: MarginalLogLikelihood):
        self._mll = mll
        self.log_det = mll.log_det

    def __call__(self, d) -> float:
        return self._mll(d)

    def inner(self, x, y) -> float:
        return float(torch.dot(self._mll.whiten(x), self._mll.whiten(y)))
