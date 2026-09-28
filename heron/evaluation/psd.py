"""
Noise PSD utilities for gravitational-wave mismatch computation.

Provides the aLIGO design-sensitivity PSD used to noise-weight inner products.
The mismatch thresholds (< 1e-3 detection-grade, < 1e-2 PE-grade) are defined
relative to this PSD, so flat-noise mismatches are physically meaningless.
"""

from __future__ import annotations

import numpy as np


def aligo_design_psd(freqs: np.ndarray) -> np.ndarray:
    """aLIGO Zero-Det High-Power one-sided PSD on a given frequency grid.

    Uses ``lalsimulation.SimNoisePSDaLIGOZeroDetHighPower`` when lalsuite is
    installed; otherwise falls back to the analytic fit from
    Ajith et al. (2011), Phys. Rev. D 84, 084037.

    Frequencies below 10 Hz are set to ``np.inf`` so they contribute zero
    weight to the noise-weighted inner product.

    Parameters
    ----------
    freqs : ndarray
        Positive frequencies in Hz (typically from ``scipy.fft.rfftfreq``).

    Returns
    -------
    ndarray
        PSD values in units of strain²/Hz, same shape as *freqs*.
    """
    freqs = np.asarray(freqs, dtype=float)

    try:
        psd = _lal_aligo_psd(freqs)
    except ImportError:
        psd = _analytic_aligo_psd(freqs)

    # Zero weight below low-frequency cutoff
    psd[freqs < 10.0] = np.inf
    # DC bin
    if len(psd) > 0 and freqs[0] == 0.0:
        psd[0] = np.inf

    return psd


def _lal_aligo_psd(freqs: np.ndarray) -> np.ndarray:
    """Evaluate aLIGO PSD via lalsimulation (raises ImportError if unavailable)."""
    import lalsimulation as ls

    psd = np.empty_like(freqs)
    for i, f in enumerate(freqs):
        if f <= 0.0:
            psd[i] = np.inf
        else:
            psd[i] = ls.SimNoisePSDaLIGOZeroDetHighPower(float(f))
    return psd


def _analytic_aligo_psd(freqs: np.ndarray) -> np.ndarray:
    """Analytic fit to aLIGO design PSD.

    From Ajith et al. (2011), valid from ~10 Hz to a few kHz.
    S0 and f0 are chosen to match the Zero-Det High-Power curve.
    """
    S0 = 1.0e-49
    f0 = 215.0

    f = np.where(freqs > 0.0, freqs, np.inf)
    x = f / f0

    psd = S0 * (
        x ** (-4.14)
        - 5.0 * x ** (-2.0)
        + 111.0 * (1.0 - x**2 + 0.5 * x**4) / (1.0 + 0.5 * x**2)
    )

    # The expression can go negative at very low frequencies — clip to inf there
    psd = np.where(psd > 0.0, psd, np.inf)
    psd[freqs <= 0.0] = np.inf
    return psd
