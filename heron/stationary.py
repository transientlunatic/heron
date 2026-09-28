"""
Fast marginal likelihood for stationary noise plus a band-limited diagonal K.

The production likelihood evaluates ``log N(d; mu, Sigma)`` with

    Sigma = C + P diag(v) P

where ``C`` is the noise covariance built by :func:`heron.noise.noise_covariance`,
``P`` is the high-pass projector applied to data and mean, and ``v`` is the
per-sample GP variance.  The dense route (:class:`heron.likelihood.MarginalLogLikelihood`)
costs O(N^3) time and O(N^2) memory per call, which is unusable at real
analysis-segment sizes (N ~ 1e4-1e5).

Two exact structural facts make this cheap:

1. ``C`` is *circulant*, not merely Toeplitz: ``noise_covariance`` takes its
   autocorrelation from ``irfft``, which is periodic (``c[k] == c[N-k]``), so
   ``toeplitz(c) == circulant(c)`` exactly.  ``P`` is circulant by
   construction.  Hence ``C = F^H diag(lam) F`` and ``P = F^H diag(m) F`` with
   ``F`` the DFT, and ``G = P C^{-1} P`` is applied with two FFTs.  No
   approximation is involved.

2. A constant part of ``v`` is absorbed exactly: ``P (c I) P = c P``, so
   ``C + c P`` is still circulant with eigenvalues ``lam + c m``.

Writing ``w = v - c >= 0``, ``W = diag(w)`` and ``A = W^{1/2} G W^{1/2}`` (G now
built from ``C + cP``), the matrix determinant lemma and Woodbury identity give

    log|Sigma|           = log|C + cP| + log|I + A|
    r^T Sigma^{-1} r     = r^T G r - b^T (I + A)^{-1} b,   b = W^{1/2} G r

for any ``r`` in the range of ``P`` (data and mean both are).  ``I + A`` is
symmetric positive definite with condition number at most ``1 + ||A||``, so
the solve is done by conjugate gradients with FFT matvecs, O(N log N) each.

``log|I + A|`` is evaluated in one of three ways, chosen per call:

``"series"``
    ``tr A - tr(A^2)/2``, both exact O(N log N) sums (``tr A = g[0] sum w``,
    ``tr A^2 = w^T circ(g^2) w``).  Used when the rigorous truncation bound
    ``||A|| tr(A^2) / (3 (1 - ||A||))`` is below ``logdet_tol``.
``"support"``
    Exact Cholesky of ``I + A`` restricted to the samples where ``w > 0``
    (the rows/columns elsewhere are identity).  ``G`` restricted to that
    support is read straight off the circulant kernel ``g``, so no
    triangular solves against ``C`` are needed.  O(M^3) for support size M.
``"dense"``
    The same Cholesky on the full grid.  O(N^3); a last resort that is still
    cheaper than the historical path (one Cholesky, no N x N triangular solves).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

_LOG_2PI = math.log(2.0 * math.pi)


class StationaryNoise:
    """Exact spectral representation of :func:`heron.noise.noise_covariance`.

    Holds the circulant eigenvalues of ``C`` and the high-pass mask of ``P``;
    never forms either N x N matrix.

    Parameters mirror :func:`heron.noise.noise_covariance`; ``f_low`` also sets
    the high-pass projector, matching :class:`heron.inference.network.NetworkLikelihood`.
    """

    def __init__(
        self,
        times,
        psd_fn,
        f_low: float = 20.0,
        f_high: float | None = None,
        jitter: float = 0.0,
        jitter_rel: float = 1e-8,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ):
        times = np.asarray(times, dtype=float)
        n = len(times)
        dt = float(times[1] - times[0])
        freqs = np.fft.rfftfreq(n, d=dt)

        # Same PSD treatment as noise_covariance, step for step.
        psd = np.asarray(psd_fn(freqs), dtype=float).copy()
        psd[~np.isfinite(psd)] = 0.0
        psd[freqs < f_low] = 0.0
        if f_high is not None:
            psd[freqs > f_high] = 0.0
        psd[0] = 0.0
        autocorr0 = float(np.fft.irfft(psd, n=n)[0] / (2.0 * dt))
        diag_reg = jitter + jitter_rel * autocorr0

        # Circulant eigenvalues: rfft of the first column psd-autocorrelation is
        # psd / (2 dt) (real, since irfft/rfft round-trip), plus the ridge.
        lam = psd / (2.0 * dt) + diag_reg
        if np.any(lam <= 0.0):
            raise ValueError(
                "noise covariance is singular; set jitter or jitter_rel > 0"
            )

        self.n = n
        self.dt = dt
        self.dtype = dtype
        self.device = torch.device(device)
        self.freqs = freqs
        self._lam = torch.as_tensor(lam, dtype=dtype, device=self.device)
        self._mask = torch.as_tensor(freqs >= f_low, dtype=dtype, device=self.device)
        # Multiplicity of each rfft bin in the full length-n spectrum.
        mult = np.full(len(freqs), 2.0)
        mult[0] = 1.0
        if n % 2 == 0:
            mult[-1] = 1.0
        self._mult = torch.as_tensor(mult, dtype=dtype, device=self.device)

    # -- construction helpers ----------------------------------------------

    def tensor(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(dtype=self.dtype, device=self.device)
        return torch.as_tensor(np.asarray(x, dtype=float), dtype=self.dtype, device=self.device)

    def eigenvalues(self, floor: float = 0.0) -> torch.Tensor:
        """Circulant eigenvalues of ``C + floor * P`` (rfft bins)."""
        return self._lam + floor * self._mask

    def log_det(self, floor: float = 0.0) -> float:
        """``log|C + floor * P|`` in closed form."""
        return float((self._mult * torch.log(self.eigenvalues(floor))).sum())

    def g_spectrum(self, floor: float = 0.0) -> torch.Tensor:
        """Spectrum of ``G = P (C + floor P)^{-1} P``."""
        return self._mask / self.eigenvalues(floor)

    def kernel(self, floor: float = 0.0) -> torch.Tensor:
        """First column ``g`` of the circulant ``G``."""
        return torch.fft.irfft(self.g_spectrum(floor), n=self.n)

    def apply(self, spectrum: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Multiply by the circulant with rfft-bin ``spectrum`` (last axis of ``x``)."""
        return torch.fft.irfft(torch.fft.rfft(x) * spectrum, n=self.n)

    def highpass(self, x) -> torch.Tensor:
        """Apply ``P``."""
        return self.apply(self._mask, self.tensor(x))

    def dense_covariance(self) -> np.ndarray:
        """``C`` as a dense matrix (validation only)."""
        from scipy.linalg import circulant

        c = torch.fft.irfft(self._lam, n=self.n).cpu().numpy()
        return circulant(c)


@dataclass
class _Factor:
    """Per-call factorisation of ``Sigma = C + P diag(v) P``."""

    floor: float
    g_spec: torch.Tensor           # spectrum of G (with floor absorbed)
    sqrt_w: torch.Tensor           # W^{1/2}, length n (zeros allowed)
    log_det: float                 # log|Sigma|
    method: str                    # "none" | "series" | "support" | "dense"
    logdet_error_bound: float      # rigorous bound on |log_det error| (0 if exact)
    chol: torch.Tensor | None = None       # Cholesky of I + A on `support`
    support: torch.Tensor | None = None    # indices of the Cholesky block


class StationaryMarginalLikelihood:
    """``log N(d; mu, C + P diag(v) P)`` using FFTs, CG and Woodbury.

    A drop-in for ``MarginalLogLikelihood(C, mu, P diag(v) P)`` when ``C``
    comes from a :class:`StationaryNoise` and ``d``, ``mu`` are high-passed.

    Parameters
    ----------
    noise : StationaryNoise
    mu : array_like, shape (N,)
        High-passed mean.
    v : array_like, shape (N,) or None
        Raw (unprojected) per-sample variance; ``None`` or all-zero means K = 0.
    floor : "min" or float
        Constant absorbed exactly into the circulant part. ``"min"`` uses
        ``min(v)``, which is optimal for making ``w`` small and sparse.
    logdet_tol : float
        Largest acceptable truncation bound (nats) for the series log-det.
    max_support : int
        Largest support for the exact "support" method before falling back
        to "dense" (or raising, if ``allow_dense`` is False).
    cg_tol : float
        Relative residual tolerance for the CG solves.
    """

    def __init__(
        self,
        noise: StationaryNoise,
        mu,
        v=None,
        floor="min",
        logdet_tol: float = 1e-3,
        max_support: int = 8192,
        allow_dense: bool = True,
        cg_tol: float = 1e-12,
        cg_maxiter: int = 500,
    ):
        self.noise = noise
        self.n = noise.n
        self._mu = noise.tensor(mu)
        self._cg_tol = cg_tol
        self._cg_maxiter = cg_maxiter
        self._f = self._factorise(v, floor, logdet_tol, max_support, allow_dense)

    # -- factorisation -------------------------------------------------------

    def _factorise(self, v, floor, logdet_tol, max_support, allow_dense) -> _Factor:
        noise = self.noise
        if v is None:
            v_t = None
        else:
            v_t = torch.clamp(noise.tensor(v), min=0.0)
            if not bool(torch.any(v_t > 0)):
                v_t = None

        if v_t is None:
            return _Factor(0.0, noise.g_spectrum(0.0), torch.zeros(self.n, dtype=noise.dtype,
                           device=noise.device), noise.log_det(0.0), "none", 0.0)

        c = float(v_t.min()) if floor == "min" else float(floor)
        w = torch.clamp(v_t - c, min=0.0)
        g_spec = noise.g_spectrum(c)
        log_det_c = noise.log_det(c)
        sqrt_w = torch.sqrt(w)

        nz = torch.nonzero(w > 0).squeeze(-1)
        if nz.numel() == 0:
            return _Factor(c, g_spec, sqrt_w, log_det_c, "none", 0.0)
        # Work on the bounding index range of the support: rows/columns with
        # w = 0 inside it are identity rows of I + A, so this is still exact.
        lo, hi = int(nz[0]), int(nz[-1]) + 1
        m = hi - lo

        # ||A|| <= max(w) * max(spec G): rigorous, O(N).
        a_norm = float(w.max() * g_spec.max())
        g = torch.fft.irfft(g_spec, n=self.n)
        if a_norm < 1.0:
            tr_a = float(g[0] * w.sum())
            # tr(A^2) = sum_ij w_i w_j g_{i-j}^2 = w^T circ(g^2) w
            tr_a2 = float(w @ noise.apply(torch.fft.rfft(g * g).real, w))
            bound = a_norm * tr_a2 / (3.0 * (1.0 - a_norm))
            if bound <= logdet_tol:
                return _Factor(c, g_spec, sqrt_w, log_det_c + tr_a - 0.5 * tr_a2,
                               "series", bound)

        if m > max_support:
            if not allow_dense:
                raise RuntimeError(
                    f"K support {m} exceeds max_support={max_support} and the "
                    f"series log-det bound is not met (||A|| <= {a_norm:.3g})"
                )
            method = "dense"
        else:
            method = "support"

        # I + W_S^{1/2} G_SS W_S^{1/2}.  For a contiguous index range G_SS is the
        # Toeplitz block G_SS[i, j] = g[(i - j) mod n] whatever the offset, so it
        # is read off the kernel through an unfold view (no index matrix).
        k = torch.arange(-(m - 1), m, device=noise.device) % self.n
        M = g[k].unfold(0, m, 1).flip(0)
        s = sqrt_w[lo:hi]
        M.mul_(s[:, None]).mul_(s[None, :])
        M.diagonal().add_(1.0)
        chol = torch.linalg.cholesky(M)
        del M
        log_det = log_det_c + 2.0 * float(chol.diagonal().log().sum())
        idx = torch.arange(lo, hi, device=noise.device)
        return _Factor(c, g_spec, sqrt_w, log_det, method, 0.0, chol, idx)

    # -- linear algebra --------------------------------------------------------

    def _apply_a(self, x: torch.Tensor) -> torch.Tensor:
        s = self._f.sqrt_w
        return s * self.noise.apply(self._f.g_spec, s * x)

    def _solve_i_plus_a(self, b: torch.Tensor) -> torch.Tensor:
        """Solve ``(I + A) y = b``."""
        f = self._f
        if f.chol is not None:
            y = b.clone()
            rhs = b[f.support]
            y[f.support] = torch.cholesky_solve(rhs.unsqueeze(-1), f.chol).squeeze(-1)
            return y
        return self._cg(b)

    def _cg(self, b: torch.Tensor) -> torch.Tensor:
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rs = torch.dot(r, r)
        stop = (self._cg_tol ** 2) * float(rs)
        for _ in range(self._cg_maxiter):
            if float(rs) <= stop:
                return x
            ap = p + self._apply_a(p)
            alpha = rs / torch.dot(p, ap)
            x = x + alpha * p
            r = r - alpha * ap
            rs_new = torch.dot(r, r)
            p = r + (rs_new / rs) * p
            rs = rs_new
        raise RuntimeError("CG did not converge; ||A|| may be unexpectedly large")

    def solve(self, y) -> torch.Tensor:
        """``Sigma^{-1} y`` for ``y`` in the range of ``P``."""
        y_ = self.noise.tensor(y)
        gy = self.noise.apply(self._f.g_spec, y_)
        if self._f.method == "none":
            return gy
        s = self._f.sqrt_w
        z = self._solve_i_plus_a(s * gy)
        return gy - self.noise.apply(self._f.g_spec, s * z)

    def inner(self, x, y) -> float:
        """``x^T Sigma^{-1} y`` for ``x``, ``y`` in the range of ``P``."""
        return float(torch.dot(self.noise.tensor(x), self.solve(y)))

    # -- public API mirroring MarginalLogLikelihood ---------------------------

    @property
    def log_det(self) -> float:
        return self._f.log_det

    @property
    def method(self) -> str:
        return self._f.method

    @property
    def logdet_error_bound(self) -> float:
        return self._f.logdet_error_bound

    def __call__(self, d) -> float:
        r = self.noise.tensor(d) - self._mu
        quad = self.inner(r, r)
        return -0.5 * (self.n * _LOG_2PI + self._f.log_det + quad)
