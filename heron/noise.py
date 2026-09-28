"""
Time-domain noise covariance utilities.

For stationary Gaussian noise with one-sided PSD S(f), the time-domain
covariance is a symmetric Toeplitz matrix:

    C_ij = R(|t_i - t_j|),    R(τ) = ∫_0^∞ S(f) cos(2π f τ) df

This is the Wiener-Khinchin theorem in discrete form.  The integral is
evaluated via irfft, which applies a trapezoidal rule over the frequency
grid defined by the sample times.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import toeplitz


def noise_covariance(
    times: np.ndarray,
    psd_fn,
    f_low: float = 20.0,
    f_high: float | None = None,
    jitter: float = 0.0,
    jitter_rel: float = 1e-8,
) -> np.ndarray:
    """Noise covariance matrix for stationary Gaussian noise.

    Parameters
    ----------
    times : array_like, shape (N,)
        Uniformly-spaced sample times in seconds.
    psd_fn : callable
        One-sided PSD [strain²/Hz] as a function of frequency [Hz].
        May return ``np.inf`` at unsupported frequencies (e.g. below the
        detector's low-frequency wall); those bins are zeroed.
    f_low : float
        Low-frequency cutoff in Hz.  PSD contributions below this are zeroed.
    f_high : float or None
        High-frequency cutoff in Hz.  Defaults to the Nyquist frequency.
    jitter : float
        Absolute diagonal regularisation (strain²).  Defaults to 0.
    jitter_rel : float
        Relative diagonal regularisation added as ``jitter_rel * R(0)``,
        where R(0) is the zero-lag autocorrelation.  Keeps the matrix
        numerically SPD without depending on the physical scale of the PSD.
        Defaults to 1e-8.  Set to 0 to disable.

    Returns
    -------
    C : ndarray, shape (N, N)
        Symmetric positive-definite noise covariance matrix.
    """
    times = np.asarray(times, dtype=float)
    n = len(times)
    dt = float(times[1] - times[0])

    freqs = np.fft.rfftfreq(n, d=dt)  # shape (n//2 + 1,)

    psd = np.asarray(psd_fn(freqs), dtype=float).copy()

    # Zero non-finite values (inf below low-freq wall, NaN from bad fits).
    psd[~np.isfinite(psd)] = 0.0
    psd[freqs < f_low] = 0.0
    if f_high is not None:
        psd[freqs > f_high] = 0.0
    psd[0] = 0.0  # DC component always zero (mean-zero noise assumed)

    # Autocorrelation via irfft.
    #
    # np.fft.irfft(S, n)[l] = (1/n)[S[0] + 2 Σ_{k=1}^{n//2-1} S[k] cos(2πkl/n) + S[n//2] cos(πl)]
    #
    # Dividing by 2*dt gives the trapezoidal approximation to R(l*dt):
    #
    #   R(l·dt) ≈ df · [S[0]/2 + Σ_{k=1}^{n//2-1} S[k] cos(2πkl/n) + S[n//2]/2 · cos(πl)]
    #           = irfft(psd, n)[l] / (2·dt)
    autocorr = np.fft.irfft(psd, n=n).real / (2.0 * dt)

    C = toeplitz(autocorr)
    C = 0.5 * (C + C.T)  # enforce exact symmetry against irfft rounding

    diag_reg = jitter + jitter_rel * float(autocorr[0])
    if diag_reg > 0.0:
        C += diag_reg * np.eye(n)

    return C
