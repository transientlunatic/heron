"""Coherent network injections for parameter estimation.

:class:`Injection` builds a simulated data set — signal plus coloured Gaussian
noise — across a detector network, from either a trained surrogate or a
reference LAL approximant.  It consolidates the injection logic that was
previously copy-pasted into ``scripts/injection_nested_sampling.py`` and
``scripts/pp_plot_demod.py``, and produces data compatible with
:class:`heron.inference.network.NetworkLikelihood`.

Two source types are supported and treated honestly:

- **A surrogate** (has ``.predict``) — evaluated at its reference distance /
  face-on inclination, then projected analytically via
  :func:`heron.inference.projection.project_polarisations` (the same path the
  likelihood uses, so surrogate self-injections are consistent by construction).
- **A reference approximant** (has ``.time_domain``) — LAL generates the
  waveform at the physical distance and inclination directly; the antenna
  response and coalescence-phase rotation are applied here.

Each detector sees the signal at its own geocentre-relative arrival time and
through its own antenna response, so the injection is a coherent network signal.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from heron.noise import noise_covariance
from heron.inference.projection import project_polarisations


@dataclass
class InjectionResult:
    """Output of :meth:`Injection.generate`."""

    data: dict[str, np.ndarray]        # prefix -> signal + noise
    signals: dict[str, np.ndarray]     # prefix -> noiseless projected signal
    snrs: dict[str, float]             # prefix -> optimal matched-filter SNR
    network_snr: float
    times: np.ndarray


@dataclass
class Injection:
    """A network injection specification.

    Parameters
    ----------
    times : ndarray
        Uniformly-spaced GPS times shared by all detectors.
    detectors : list[Detector]
        The detector network.
    parameters : dict
        Source parameters.  Required: ``mass_ratio``, ``tc`` (geocentre),
        ``ra``, ``dec``, ``psi``.  Optional: ``total_mass``,
        ``luminosity_distance``, ``inclination``, ``coalescence_phase``.
    f_low : float
        Low-frequency cutoff (Hz) for the noise PSD and SNR computation.
    generation_sample_rate : float
        Sample rate handed to LAL when the source is an approximant.
    """

    times: np.ndarray
    detectors: list
    parameters: dict
    f_low: float = 20.0
    generation_sample_rate: float = 4096.0
    _hp_mask: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=float)
        dt = float(self.times[1] - self.times[0])
        freqs = np.fft.rfftfreq(len(self.times), d=dt)
        self._hp_mask = freqs >= self.f_low

    # ------------------------------------------------------------------

    def _hp_filter(self, x: np.ndarray) -> np.ndarray:
        x_f = np.fft.rfft(x)
        x_f[~self._hp_mask] = 0.0
        return np.fft.irfft(x_f, n=len(self.times))

    def _extrinsic(self) -> dict:
        p = self.parameters
        return {
            "ra": float(p["ra"]),
            "dec": float(p["dec"]),
            "psi": float(p["psi"]),
            "inclination": float(p.get("inclination", p.get("theta_jn", 0.0))),
            "coalescence_phase": float(
                p.get("coalescence_phase", p.get("phase", 0.0))
            ),
            "distance": p.get("luminosity_distance", p.get("distance", None)),
        }

    def _surrogate_signal(self, source, t_rel, fp, fc, extr, distance_ref):
        intrinsic = {
            k: v for k, v in self.parameters.items()
            if k not in ("tc", "geocent_time", "ra", "dec", "psi", "inclination",
                         "theta_jn", "coalescence_phase", "phase",
                         "luminosity_distance", "distance")
        }
        wf = source.predict({**intrinsic, "times": t_rel})
        mu, _ = project_polarisations(
            wf, f_plus=fp, f_cross=fc,
            distance=extr["distance"], distance_ref=distance_ref,
            inclination=extr["inclination"],
            coalescence_phase=extr["coalescence_phase"],
        )
        return mu

    def _approximant_signal(self, source, t_rel, fp, fc, extr):
        import astropy.units as u

        p = {
            "mass_ratio": float(self.parameters["mass_ratio"]),
            "inclination": extr["inclination"],
            "f_min": self.f_low * u.Hertz,
            "delta_t": (1.0 / self.generation_sample_rate) * u.second,
        }
        if "total_mass" in self.parameters:
            p["total_mass"] = float(self.parameters["total_mass"]) * u.solMass
        if extr["distance"] is not None:
            p["luminosity_distance"] = float(extr["distance"]) * u.Mpc
        wf = source.time_domain(p, times=t_rel)
        hp = np.asarray(wf["plus"].data, dtype=float)
        hc = np.asarray(wf["cross"].data, dtype=float)
        phic = extr["coalescence_phase"]
        if phic != 0.0:
            c2, s2 = np.cos(2.0 * phic), np.sin(2.0 * phic)
            hp, hc = c2 * hp - s2 * hc, s2 * hp + c2 * hc
        return fp * hp + fc * hc

    # ------------------------------------------------------------------

    def generate(self, source, rng=None, distance_ref=None) -> InjectionResult:
        """Build injected data across the network.

        Parameters
        ----------
        source : WaveformSurrogate or approximant
            Signal model.  A surrogate is projected analytically; an approximant
            (``.time_domain``) is generated by LAL at the physical parameters.
        rng : numpy Generator or None
            Random source for the coloured noise.
        distance_ref : float or None
            Surrogate reference distance.  Defaults to
            ``getattr(source, 'distance_factor', None)``.

        Returns
        -------
        InjectionResult
        """
        rng = np.random.default_rng() if rng is None else rng
        if distance_ref is None:
            distance_ref = getattr(source, "distance_factor", None)
        is_surrogate = hasattr(source, "predict")

        extr = self._extrinsic()
        tc = float(self.parameters.get("tc", self.parameters.get("geocent_time")))

        data, signals, snrs = {}, {}, {}
        sq_network = 0.0
        for det in self.detectors:
            dt_geo = det.time_delay_from_geocentre(extr["ra"], extr["dec"], tc)
            t_rel = self.times - (tc + dt_geo)
            fp, fc = det.antenna_patterns(extr["ra"], extr["dec"], extr["psi"], tc)

            if is_surrogate:
                signal = self._surrogate_signal(source, t_rel, fp, fc, extr, distance_ref)
            else:
                signal = self._approximant_signal(source, t_rel, fp, fc, extr)

            C = noise_covariance(self.times, det.psd_fn, f_low=self.f_low, jitter_rel=1e-8)
            noise = np.linalg.cholesky(C) @ rng.standard_normal(len(self.times))

            sig_hp = self._hp_filter(signal)
            snr = float(np.sqrt(sig_hp @ np.linalg.solve(C, sig_hp)))

            signals[det.prefix] = signal
            data[det.prefix] = signal + noise
            snrs[det.prefix] = snr
            sq_network += snr**2

        return InjectionResult(
            data=data, signals=signals, snrs=snrs,
            network_snr=float(np.sqrt(sq_network)), times=self.times,
        )
