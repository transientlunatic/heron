"""Real strain data ingestion for time-domain network PE.

:func:`fetch_gwosc_strain` pulls open strain data around a trigger time and
conditions it for :class:`heron.inference.network.NetworkLikelihood`, which
is a *time-domain* likelihood: both its high-pass filter (``_hp_filter``) and
:func:`heron.noise.noise_covariance`'s Toeplitz construction operate on a
finite segment via FFT, implicitly treating it as periodic. A raw slice of a
continuous strain channel does not respect that assumption at its boundaries
— the discontinuity between the segment's last and first sample aliases
low-frequency power across the whole band when FFT'd. The standard LIGO/Virgo
fix (the same one bilby's frequency-domain likelihood uses, e.g.
``InterferometerStrainData.set_from_gwpy_timeseries``) is:

1. High-pass *before* windowing, not after — raw strain below ~10-15 Hz is
   orders of magnitude louder than the astrophysical band, and windowing that
   large a component first would leak its power across many nearby frequency
   bins (a window multiplication is a convolution in the frequency domain).
2. Taper the edges of the analysis segment itself with a Tukey window of a
   fixed ``roll_off`` duration, so the segment returns to (near) zero at both
   ends and the implicit periodic boundary is well-behaved.

This does cost a little SNR at the tapered edges, and windowed data no longer
exactly has covariance ``C`` as built from an unwindowed PSD estimate — both
standard, accepted approximations at this level (the same ones the rest of
the field makes), not unique to heron. Not yet addressed: matching this
windowing convention on the PSD-estimation side too (the BayesWave PSDs used
elsewhere in this codebase are external and pre-date this module).

Usage::

    from heron.inference.strain import fetch_gwosc_strain

    strain, times = fetch_gwosc_strain(
        "H1", trigger_time=1126259462.4, duration=4.0, sample_rate=4096.0,
    )
    # strain, times now drop straight into
    # NetworkLikelihood(data={"H1": strain, ...}, times=times, ...)
"""
from __future__ import annotations

import numpy as np


def taper_strain(strain: np.ndarray, dt: float, roll_off: float = 0.4) -> np.ndarray:
    """Apply a Tukey window to a strain segment's edges.

    Matches bilby's convention: a Tukey window whose flat top spans the
    segment except for ``roll_off`` seconds tapered to zero at each end
    (``alpha = 2 * roll_off / duration``, clipped to 1 for segments shorter
    than ``2 * roll_off``, which taper their full length).

    Parameters
    ----------
    strain : ndarray
        Strain time series.
    dt : float
        Sample spacing in seconds.
    roll_off : float
        Taper duration at each end, in seconds.

    Returns
    -------
    ndarray
        The windowed strain, same shape as the input.
    """
    from scipy.signal.windows import tukey

    strain = np.asarray(strain, dtype=float)
    n = len(strain)
    duration = n * dt
    alpha = min(1.0, 2.0 * roll_off / duration) if duration > 0 else 1.0
    return strain * tukey(n, alpha=alpha)


def fetch_gwosc_strain(
    detector: str,
    trigger_time: float,
    duration: float = 4.0,
    post_trigger_duration: float = 2.0,
    sample_rate: float = 4096.0,
    native_sample_rate: float = 4096.0,
    f_low: float = 20.0,
    roll_off: float = 0.4,
    cache: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Fetch and condition open GWOSC strain data around a trigger time.

    Selects ``[trigger_time + post_trigger_duration - duration,
    trigger_time + post_trigger_duration)`` (the usual convention: most of
    the segment precedes the trigger, with a short tail to capture
    ringdown), high-passes at ``f_low``, then tapers the segment edges (see
    the module docstring for why both steps, and in that order).

    GWOSC only serves strain at a couple of fixed product rates (4096 or
    16384 Hz) — asking gwpy for anything else is rejected server-side, not
    resampled. So this always fetches at ``native_sample_rate`` (must be one
    of those) and, only if ``sample_rate`` differs, downsamples afterwards
    via :meth:`gwpy.timeseries.TimeSeries.resample` (a proper anti-aliased
    resample, not decimation). Pick ``sample_rate`` to comfortably exceed
    twice the analysis band's highest frequency of interest — and be aware
    that :class:`heron.inference.network.NetworkLikelihood` does dense O(N^3)
    work per detector *per likelihood call* (see the module docstring's
    caution), so ``duration * sample_rate`` should stay in the same regime
    as whatever segment size that likelihood has actually been exercised at,
    not simply "as high as GWOSC allows."

    Parameters
    ----------
    detector : str
        Detector prefix understood by GWOSC (``'H1'``, ``'L1'``, ``'V1'``).
    trigger_time : float
        GPS trigger/merger time.
    duration : float
        Total analysis segment length, in seconds.
    post_trigger_duration : float
        How much of ``duration`` falls after ``trigger_time`` (ringdown +
        margin). Must be ``< duration``.
    sample_rate : float
        Sample rate of the *returned* data, in Hz.
    native_sample_rate : float
        Sample rate to request from GWOSC (4096.0 or 16384.0); downsampled
        to ``sample_rate`` afterwards if they differ. Must be >= sample_rate.
    f_low : float
        High-pass cutoff in Hz, matching the likelihood's own ``f_low``.
    roll_off : float
        Tukey taper duration at each end, in seconds (see :func:`taper_strain`).
    cache : bool
        Whether to let gwpy cache the downloaded file.

    Returns
    -------
    strain : ndarray
        Conditioned strain time series.
    times : ndarray
        GPS times, uniformly spaced, same length as ``strain`` — ready to
        pass to :class:`heron.inference.network.NetworkLikelihood`.
    """
    from gwpy.timeseries import TimeSeries

    if post_trigger_duration >= duration:
        raise ValueError("post_trigger_duration must be less than duration")
    if sample_rate > native_sample_rate:
        raise ValueError("sample_rate cannot exceed native_sample_rate")

    start = trigger_time + post_trigger_duration - duration
    end = start + duration

    ts = TimeSeries.fetch_open_data(
        detector, start, end, sample_rate=native_sample_rate, cache=cache,
    )
    if sample_rate != native_sample_rate:
        ts = ts.resample(sample_rate)
    ts = ts.highpass(f_low)

    strain = np.asarray(ts.value, dtype=float)
    times = np.asarray(ts.times.value, dtype=float)
    dt = 1.0 / sample_rate
    strain = taper_strain(strain, dt=dt, roll_off=roll_off)

    return strain, times
