"""Overlap with *continuous* time maximisation.

``heron.evaluation.mismatch.compute_overlap`` only searches whole-sample lags,
so a sub-sample timing offset (e.g. LAL's sample-snapped epoch, task #399)
shows up as spurious mismatch.  Here the integer-lag peak is refined to a
continuous shift by evaluating the cross-correlation at fractional lags
(exact Fourier-domain phase ramp, no interpolation error).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def _setup(h1, h2, dt, psd):
    n = min(len(h1), len(h2))
    a = np.fft.rfft(np.asarray(h1, float)[:n])
    b = np.fft.rfft(np.asarray(h2, float)[:n])
    f = np.fft.rfftfreq(n, d=dt)
    w = np.ones_like(f) if psd is None else 1.0 / np.asarray(psd, float)[: len(f)]
    w = w.copy()
    w[0] *= 0.5            # trapezoid weights, as compute_overlap
    w[-1] *= 0.5
    return n, a, b, f, w


def subsample_overlap(h1, h2, dt, psd=None, maximize_phase=True,
                      max_time_shift=0.05, return_shift=False):
    """Fitting-factor overlap maximised over constant phase and a continuous
    time shift within ``+-max_time_shift`` seconds.

    ``psd`` is the one-sided PSD on ``rfftfreq(n, dt)`` (``None`` = flat).
    With ``return_shift=True`` also returns the best-fit shift in seconds:
    the delay to apply to ``h2`` to line it up with ``h1`` (negative if
    ``h2`` is later than ``h1``).
    """
    n, a, b, f, w = _setup(h1, h2, dt, psd)
    norm = np.sqrt(np.sum(w * np.abs(a) ** 2) * np.sum(w * np.abs(b) ** 2))
    if norm == 0:
        return (0.0, 0.0) if return_shift else 0.0
    c = w * a * np.conj(b)

    def z(tau):
        return np.sum(c * np.exp(2j * np.pi * f * tau))

    def val(tau):
        zz = z(tau)
        return abs(zz) if maximize_phase else zz.real

    # Coarse: all integer lags at once (complex ifft of the one-sided spectrum).
    arr = np.zeros(n, complex)
    arr[: len(c)] = c
    zint = np.fft.ifft(arr) * n
    lags = np.fft.fftfreq(n, d=1.0 / n)          # signed integer lags (samples)
    ok = np.abs(lags) * dt <= max_time_shift
    score = np.abs(zint) if maximize_phase else zint.real
    m = int(lags[ok][np.argmax(score[ok])])

    # Fine: bracket the integer peak by +-1 sample and refine continuously.
    lo = max(m - 1, -max_time_shift / dt)
    hi = min(m + 1, max_time_shift / dt)
    res = minimize_scalar(lambda s: -val(s * dt), bounds=(lo, hi),
                          method="bounded", options={"xatol": 1e-9})
    s = float(res.x)
    ov = float(min(val(s * dt) / norm, 1.0))
    return (ov, s * dt) if return_shift else ov


def subsample_mismatch(h1, h2, dt, psd=None, **kw):
    """``1 - subsample_overlap``."""
    return 1.0 - subsample_overlap(h1, h2, dt, psd, **kw)
