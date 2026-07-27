"""Detector abstraction for network parameter estimation.

A :class:`Detector` bundles the three things the network likelihood needs for
one interferometer: its noise PSD, its antenna response (via
``heron.detector``), and its position (for inter-detector arrival-time delays).

The heavy lifting lives in ``heron.detector`` — antenna patterns, ECEF
locations, and geocentre time delays are all implemented there and simply
delegated to here, so there is a single source of truth for detector geometry.

Usage::

    from heron.inference.detectors import Detector, KNOWN_DETECTORS

    h1 = Detector.from_name("H1")                 # default aLIGO design PSD
    l1 = Detector.from_name("L1", psd_fn=my_psd)  # custom / estimated PSD
    network = [h1, l1]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from heron.detector import antenna_patterns, time_delay_from_geocentre
from heron.evaluation.psd import aligo_design_psd


@dataclass
class Detector:
    """A single interferometer: geometry (by name) plus a noise PSD.

    Parameters
    ----------
    prefix : str
        Detector name understood by ``heron.detector``: ``'H1'``, ``'L1'`` or
        ``'V1'``.
    psd_fn : callable
        One-sided PSD [strain²/Hz] as a function of frequency [Hz].  Defaults
        to :func:`heron.evaluation.psd.aligo_design_psd`.  Pass an estimated
        PSD (e.g. from :func:`estimate_psd_welch`) for real data.
    """

    prefix: str
    psd_fn: Callable[[np.ndarray], np.ndarray] = aligo_design_psd

    @classmethod
    def from_name(
        cls,
        prefix: str,
        psd_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> "Detector":
        """Build a detector by name, using the default PSD unless overridden."""
        if psd_fn is None:
            return cls(prefix=prefix)
        return cls(prefix=prefix, psd_fn=psd_fn)

    def antenna_patterns(
        self, ra: float, dec: float, psi: float, gps_time: float
    ) -> tuple[float, float]:
        """Return ``(F+, F×)`` for this detector at the given sky position."""
        return antenna_patterns(ra, dec, psi, gps_time, self.prefix)

    def time_delay_from_geocentre(
        self, ra: float, dec: float, gps_time: float
    ) -> float:
        """Arrival-time delay ``t_detector − t_geocentre`` [s]."""
        return time_delay_from_geocentre(ra, dec, gps_time, self.prefix)


# Convenience registry of the three detectors this codebase supports.
KNOWN_DETECTORS: dict[str, Detector] = {
    prefix: Detector(prefix=prefix) for prefix in ("H1", "L1", "V1")
}


def estimate_psd_welch(
    strain: np.ndarray,
    dt: float,
    segment_duration: float = 4.0,
    f_low: float = 20.0,
) -> Callable[[np.ndarray], np.ndarray]:
    """Estimate a one-sided PSD from a strain time series (Welch's method).

    A basic, dependency-light PSD estimator for real-data PE: split the strain
    into 50%-overlapping Hann-windowed segments, average the periodograms, and
    return an interpolating callable with the same ``PSD(freqs) -> array``
    signature as :func:`heron.evaluation.psd.aligo_design_psd`.  Frequencies
    below ``f_low`` (and above the estimate's Nyquist) return ``np.inf`` so they
    carry zero weight, matching the analytic-PSD convention.

    This is a starting point — production PE typically uses a dedicated
    on/off-source PSD (e.g. BayesWave / median-Welch); swap in any callable
    with the same signature when that is available.

    Parameters
    ----------
    strain : ndarray
        Real strain time series (ideally off-source / signal-free).
    dt : float
        Sample spacing in seconds.
    segment_duration : float
        Length of each Welch segment in seconds.
    f_low : float
        Low-frequency cutoff; below this the returned PSD is ``inf``.

    Returns
    -------
    callable
        ``psd_fn(freqs)`` returning one-sided PSD values [strain²/Hz].
    """
    from scipy.signal import welch

    strain = np.asarray(strain, dtype=float)
    fs = 1.0 / dt
    nperseg = min(len(strain), int(round(segment_duration * fs)))
    freqs, pxx = welch(
        strain, fs=fs, window="hann", nperseg=nperseg, return_onesided=True,
    )

    f_max = float(freqs[-1])

    def psd_fn(query: np.ndarray) -> np.ndarray:
        query = np.asarray(query, dtype=float)
        out = np.interp(query, freqs, pxx, left=np.inf, right=np.inf)
        out[query < f_low] = np.inf
        out[query > f_max] = np.inf
        return out

    return psd_fn
