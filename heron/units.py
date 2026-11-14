"""Geometric-unit conversions for compact-binary waveforms.

A non-spinning binary's waveform is *exactly* self-similar in total mass M
when written in geometric units (G = c = 1, lengths and times in units of M):

    tau      = t / (M * T_sun)                     (dimensionless time)
    h_geo    = h * D / (M * L_sun)                 (dimensionless strain, "r h / M")

so a surrogate trained in (q, tau) -> h_geo needs no total-mass dimension at
all, and total mass / luminosity distance become analytic rescalings applied
at predict time. That removes the ``output_scale`` hack the physical-unit
surrogates need to rescale ~1e-21 strain, and removes the per-mass retraining
/ regeneration that physical-unit models need.

Constants are the LAL values (``lal.MTSUN_SI``, ``lal.MRSUN_SI``,
``lal.PC_SI * 1e6``) inlined so this module has no lalsuite dependency.
"""
from __future__ import annotations

import numpy as np

#: G M_sun / c^3 in seconds (lal.MTSUN_SI).
MTSUN_SI = 4.925490947641267e-06
#: G M_sun / c^2 in metres (lal.MRSUN_SI).
MRSUN_SI = 1476.6250385063849
#: One megaparsec in metres (lal.PC_SI * 1e6).
MPC_SI = 3.085677581491367e22


def time_to_geometric(t, total_mass: float):
    """Physical time [s] -> dimensionless time tau = t / M."""
    return np.asarray(t, dtype=float) / (total_mass * MTSUN_SI)


def time_from_geometric(tau, total_mass: float):
    """Dimensionless time tau -> physical time [s]."""
    return np.asarray(tau, dtype=float) * (total_mass * MTSUN_SI)


def strain_scale(total_mass: float, distance_mpc: float) -> float:
    """Factor converting geometric strain to physical strain.

    ``h_physical = strain_scale(M, D) * h_geo``, i.e. ``M L_sun / D``.
    """
    return total_mass * MRSUN_SI / (distance_mpc * MPC_SI)


def strain_to_geometric(h, total_mass: float, distance_mpc: float):
    """Physical strain -> geometric strain ``h D / M``."""
    return np.asarray(h, dtype=float) / strain_scale(total_mass, distance_mpc)


def strain_from_geometric(h_geo, total_mass: float, distance_mpc: float):
    """Geometric strain -> physical strain."""
    return np.asarray(h_geo, dtype=float) * strain_scale(total_mass, distance_mpc)
