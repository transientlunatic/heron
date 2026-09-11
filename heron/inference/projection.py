"""Analytic extrinsic projection of a surrogate waveform onto a detector.

The surrogates model ``h+``/``h×`` at a fixed *reference distance* and *face-on*
inclination (training data uses ``inclination: 0``,
``heron/models/lalsimulation.py``).  The remaining extrinsic parameters —
distance, inclination, and coalescence phase — enter analytically, with no
retraining and no extra ``predict()`` call, so they can be sampled cheaply.

:func:`project_polarisations` is the extrinsic generalisation of
:func:`heron.detector.project_waveform`: at ``distance == distance_ref``,
``inclination == reference_inclination`` and ``coalescence_phase == 0`` it
reproduces that function's mean exactly (returning the diagonal of ``K``, which
is all the marginal likelihood uses — see ``heron/gw_likelihood.py``).

Conventions (dominant ``l = 2`` mode; matches LAL / bilby):

- **Coalescence phase** ``φc`` rotates the plus/cross pair by ``2φc``::

      h+' = h+ cos2φc − h× sin2φc
      h×' = h+ sin2φc + h× cos2φc

- **Inclination** ``ι`` re-weights the (face-on) quadratures::

      h+ *= (1 + cos²ι) / 2
      h× *= cos ι

- **Distance** scales the amplitude as ``distance_ref / distance``.

Variances are propagated under the same plus/cross-independence approximation
already used throughout the pipeline (``K`` is treated as diagonal, with the
plus and cross GP covariances uncorrelated), so each linear step maps to a
quadratic step on the variance.
"""
from __future__ import annotations

import numpy as np


def _variance(waveform) -> np.ndarray:
    """Diagonal variance of a Waveform, or zeros if it carries no covariance."""
    var = waveform.variance
    if var is None:
        return np.zeros_like(waveform.data)
    return np.asarray(var, dtype=float)


def project_variances(
    var_plus: np.ndarray,
    var_cross: np.ndarray,
    *,
    f_plus: float,
    f_cross: float,
    distance: float | None = None,
    distance_ref: float | None = None,
    inclination: float = 0.0,
    coalescence_phase: float = 0.0,
) -> np.ndarray:
    """Project face-on plus/cross diagonal variances onto a detector.

    The variance analogue of :func:`project_polarisations`: applies the same
    coalescence-phase mixing, inclination re-weighting, antenna projection and
    distance scaling to the *variances* (each a quadratic step), under the
    plus/cross-independence approximation.  Factored out so the network
    likelihood's K-smoothing envelope can be projected without recomputing the
    mean.
    """
    var_p = np.asarray(var_plus, dtype=float)
    var_c = np.asarray(var_cross, dtype=float)

    if coalescence_phase != 0.0:
        c2, s2 = np.cos(2.0 * coalescence_phase), np.sin(2.0 * coalescence_phase)
        var_p, var_c = (
            c2**2 * var_p + s2**2 * var_c,
            s2**2 * var_p + c2**2 * var_c,
        )

    if inclination != 0.0:
        cos_i = np.cos(inclination)
        var_p = (0.5 * (1.0 + cos_i**2)) ** 2 * var_p
        var_c = cos_i**2 * var_c

    k_diag = f_plus**2 * var_p + f_cross**2 * var_c

    if distance is not None and distance_ref is not None:
        k_diag = k_diag * (distance_ref / distance) ** 2

    return k_diag


def project_polarisations(
    waveform_dict,
    *,
    f_plus: float,
    f_cross: float,
    distance: float | None = None,
    distance_ref: float | None = None,
    inclination: float = 0.0,
    coalescence_phase: float = 0.0,
    reference_inclination: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Project surrogate plus/cross onto a detector with extrinsic parameters.

    Parameters
    ----------
    waveform_dict : WaveformDict
        Surrogate output with ``'plus'`` and ``'cross'`` Waveform objects,
        evaluated at the reference distance and reference inclination.
    f_plus, f_cross : float
        Antenna pattern values (see :func:`heron.detector.antenna_patterns`).
    distance, distance_ref : float or None
        Luminosity distance to project to, and the surrogate's reference
        distance.  Both must be given to apply distance scaling; if either is
        ``None`` no scaling is applied (the surrogate is assumed already at the
        target distance).
    inclination : float
        Binary inclination ι in radians.
    coalescence_phase : float
        Coalescence phase φc in radians.
    reference_inclination : float
        Inclination the surrogate was trained at (0 = face-on).  Non-zero
        values are not yet supported and raise ``NotImplementedError``.

    Returns
    -------
    mu : ndarray, shape (N,)
        Projected detector strain mean.
    k_diag : ndarray, shape (N,)
        Projected diagonal predictive variance.
    """
    if reference_inclination != 0.0:
        raise NotImplementedError(
            "Only face-on (reference_inclination=0) surrogates are supported; "
            f"got {reference_inclination}."
        )

    hp = np.asarray(waveform_dict["plus"].data, dtype=float)
    hc = np.asarray(waveform_dict["cross"].data, dtype=float)

    # 1. Coalescence phase: rotate the plus/cross pair by 2 φc.
    if coalescence_phase != 0.0:
        c2, s2 = np.cos(2.0 * coalescence_phase), np.sin(2.0 * coalescence_phase)
        hp, hc = c2 * hp - s2 * hc, s2 * hp + c2 * hc

    # 2. Inclination: re-weight the face-on quadratures.
    if inclination != 0.0:
        cos_i = np.cos(inclination)
        hp = 0.5 * (1.0 + cos_i**2) * hp
        hc = cos_i * hc

    # 3. Antenna projection onto the detector.
    mu = f_plus * hp + f_cross * hc

    # 4. Distance scaling.
    if distance is not None and distance_ref is not None:
        mu = mu * (distance_ref / distance)

    # Variances follow the same transform (quadratic), via project_variances so
    # the K-smoothing envelope can reuse it without recomputing the mean.
    k_diag = project_variances(
        _variance(waveform_dict["plus"]),
        _variance(waveform_dict["cross"]),
        f_plus=f_plus,
        f_cross=f_cross,
        distance=distance,
        distance_ref=distance_ref,
        inclination=inclination,
        coalescence_phase=coalescence_phase,
    )

    return mu, k_diag
