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

import numpy as np
import torch

from heron.likelihood import MarginalLogLikelihood
from heron.noise import noise_covariance
from heron.inference.projection import project_polarisations, project_variances


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

        # Shared high-pass mask (all detectors share the time grid).
        dt = float(self.times[1] - self.times[0])
        self._freqs = np.fft.rfftfreq(self._n, d=dt)
        self._hp_mask = self._freqs >= f_low

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
            wf_off = surrogate.predict(p)
            var_p = np.maximum(var_p, _waveform_variance(wf_off["plus"]))
            var_c = np.maximum(var_c, _waveform_variance(wf_off["cross"]))
        return var_p, var_c

    # ------------------------------------------------------------------
    # Likelihood evaluation
    # ------------------------------------------------------------------

    def __call__(self, params: dict) -> float:
        """Return the network log-likelihood ``Σ_k log p(d_k | θ)``."""
        intrinsic, extr = self._split_params(params)
        tc = extr["tc"]
        total = 0.0

        for ch in self._channels:
            det = ch.detector
            dt_geo = det.time_delay_from_geocentre(extr["ra"], extr["dec"], tc)
            t_rel = self.times - (tc + dt_geo)

            surrogate_params = {**intrinsic, "times": t_rel}
            wf = self.surrogate.predict(surrogate_params)

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
                K = np.diag(k_diag)
            else:
                K = np.zeros((self._n, self._n))

            total += MarginalLogLikelihood(
                C=None, mu=mu, K=K, dtype=self.dtype, device=self.device,
                _L_C=ch.L_C,
            )(ch.data)

        return float(total)

    @property
    def noise_covariances(self) -> dict[str, np.ndarray]:
        """The pre-computed noise covariance per detector prefix."""
        return {ch.detector.prefix: ch.C for ch in self._channels}
