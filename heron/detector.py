"""Detector response functions for GW parameter estimation.

Antenna pattern functions (F+, F×) map sky position and polarisation angle to
the detector response.  The formula implemented here matches the convention used
by LAL's XLALComputeDetAMResponse:

    ha = GMST - RA
    p̂ = (sin ha, cos ha, 0)               [east / increasing-RA direction]
    q̂ = (sin δ cos ha, −sin δ sin ha, −cos δ)  [south / decreasing-dec direction]
    F+(ψ=0) = p̂^T D p̂ − q̂^T D q̂
    F×(ψ=0) = p̂^T (D + D^T) q̂              [factor of 2 from ê× symmetrisation]
    F+(ψ)   = F+(0) cos 2ψ + F×(0) sin 2ψ
    F×(ψ)   = −F+(0) sin 2ψ + F×(0) cos 2ψ

For environments where lal is available the implementation delegates to
lal.ComputeDetAMResponse; the pure-numpy fallback is used otherwise.
"""
from __future__ import annotations

import numpy as np

# ECEF arm unit vectors sourced from LAL DetectorSite.c.
_DETECTORS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "H1": (
        np.array([-0.22389266154, 0.79983062746, 0.55690487861]),
        np.array([-0.91397818574, 0.02609403989, -0.40492342125]),
    ),
    "L1": (
        np.array([-0.95457412153, -0.14158077340, -0.26218911324]),
        np.array([0.29774156894, -0.48791033647, -0.82054461286]),
    ),
    "V1": (
        np.array([-0.70045821479, 0.20848948619, 0.68256166277]),
        np.array([-0.05379255368, -0.96908180524, 0.24080451708]),
    ),
}

_J2000_GPS: float = 630763213.0
_GMST_J2000: float = 4.894961212823756
_OMEGA_EARTH: float = 7.2921150e-5  # rad/s

_C_SI: float = 299792458.0  # speed of light [m/s]

# Earth-fixed (ECEF) detector positions in metres, from LAL DetectorSite.c.
# Used for inter-detector / geocentre arrival-time delays.
_DETECTOR_LOCATIONS: dict[str, np.ndarray] = {
    "H1": np.array([-2.16141492636e06, -3.83469517889e06, 4.60035022664e06]),
    "L1": np.array([-7.42760447238e04, -5.49628371971e06, 3.22425701575e06]),
    "V1": np.array([4.54637409900e06, 8.42989697626e05, 4.37857696241e06]),
}


def gmst_at_gps(gps_time: float) -> float:
    """Greenwich Mean Sidereal Time in radians at the given GPS time."""
    return (_GMST_J2000 + _OMEGA_EARTH * (gps_time - _J2000_GPS)) % (2.0 * np.pi)


def detector_tensor(detector: str) -> np.ndarray:
    """Return the (3×3) symmetric detector tensor D = (x⊗x − y⊗y) / 2."""
    x, y = _DETECTORS[detector]
    return 0.5 * (np.outer(x, x) - np.outer(y, y))


def antenna_patterns(
    ra: float,
    dec: float,
    psi: float,
    gps_time: float,
    detector: str,
) -> tuple[float, float]:
    """Compute F+ and F× for *detector* at the given sky position and GPS time.

    Uses lal.ComputeDetAMResponse when LAL is available; falls back to the
    pure-numpy implementation otherwise.  Both produce identical results to
    within floating-point precision.

    Parameters
    ----------
    ra : float
        Right ascension in radians.
    dec : float
        Declination in radians.
    psi : float
        Polarisation angle in radians.
    gps_time : float
        GPS time of the event in seconds.
    detector : str
        Detector name: ``'H1'``, ``'L1'``, or ``'V1'``.

    Returns
    -------
    f_plus, f_cross : float
    """
    try:
        return _antenna_patterns_lal(ra, dec, psi, gps_time, detector)
    except ImportError:
        return _antenna_patterns_numpy(ra, dec, psi, gps_time, detector)


def _antenna_patterns_lal(
    ra: float,
    dec: float,
    psi: float,
    gps_time: float,
    detector: str,
) -> tuple[float, float]:
    import lal

    gmst = gmst_at_gps(gps_time)
    det = lal.cached_detector_by_prefix[detector]
    return lal.ComputeDetAMResponse(det.response, ra, dec, psi, gmst)


def _antenna_patterns_numpy(
    ra: float,
    dec: float,
    psi: float,
    gps_time: float,
    detector: str,
) -> tuple[float, float]:
    D = detector_tensor(detector)
    gmst = gmst_at_gps(gps_time)
    ha = gmst - ra

    p = np.array([np.sin(ha), np.cos(ha), 0.0])
    q = np.array([np.sin(dec) * np.cos(ha), -np.sin(dec) * np.sin(ha), -np.cos(dec)])

    fp0 = float(p @ D @ p - q @ D @ q)
    fc0 = float(p @ (D + D.T) @ q)  # 2 p^T D q for symmetric D

    cos2, sin2 = np.cos(2.0 * psi), np.sin(2.0 * psi)
    return fp0 * cos2 + fc0 * sin2, -fp0 * sin2 + fc0 * cos2


def project_waveform(
    waveform_dict,
    f_plus: float,
    f_cross: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project plus/cross polarisations onto a single detector.

    Returns the combined mean strain and covariance:
        μ(t)     = F+ h+(t) + F× h×(t)
        K(t, t') = F+² K+(t, t') + F×² K×(t, t')

    The plus/cross GP covariances are assumed uncorrelated (as produced by
    ExactGPSurrogate).

    Parameters
    ----------
    waveform_dict : WaveformDict
        Surrogate output containing ``'plus'`` and ``'cross'`` Waveform objects.
    f_plus, f_cross : float
        Antenna pattern values from :func:`antenna_patterns`.

    Returns
    -------
    mu : ndarray, shape (N,)
    K : ndarray, shape (N, N)
    """
    h_plus = waveform_dict["plus"]
    h_cross = waveform_dict["cross"]

    mu = f_plus * h_plus.data + f_cross * h_cross.data
    K = f_plus**2 * h_plus.covariance + f_cross**2 * h_cross.covariance

    return mu, K


def detector_location(detector: str) -> np.ndarray:
    """Return the Earth-fixed (ECEF) position of *detector* in metres.

    Uses ``lal.cached_detector_by_prefix`` when LAL is available (exact LAL
    values); falls back to the hardcoded positions in ``_DETECTOR_LOCATIONS``
    otherwise.  Both agree to sub-metre precision.
    """
    try:
        import lal

        return np.asarray(lal.cached_detector_by_prefix[detector].location, dtype=float)
    except ImportError:
        return _DETECTOR_LOCATIONS[detector]


def time_delay_from_geocentre(
    ra: float,
    dec: float,
    gps_time: float,
    detector: str,
) -> float:
    """Signal arrival-time delay at *detector* relative to the geocentre [s].

    Matches LAL's ``XLALTimeDelayFromEarthCenter`` convention: the returned
    value is ``t_detector − t_geocentre`` for a plane wave from sky position
    ``(ra, dec)``.  A source directly along the detector's position vector
    arrives *earlier* at the detector (negative delay).

    Parameters
    ----------
    ra : float
        Right ascension in radians.
    dec : float
        Declination in radians.
    gps_time : float
        GPS time of the event in seconds.
    detector : str
        Detector name: ``'H1'``, ``'L1'``, or ``'V1'``.

    Returns
    -------
    float
        Arrival-time delay in seconds; ``|delay| <= |r_det| / c ≈ 21 ms``.
    """
    gha = gmst_at_gps(gps_time) - ra  # Greenwich hour angle
    ehat_src = np.array([
        np.cos(dec) * np.cos(gha),
        -np.cos(dec) * np.sin(gha),
        np.sin(dec),
    ])
    r_det = detector_location(detector)
    return float(-np.dot(ehat_src, r_det) / _C_SI)
