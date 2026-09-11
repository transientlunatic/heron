"""
GP-model marginalised likelihood for gravitational-wave parameter estimation.

The marginalised likelihood is derived by integrating over the waveform h:

    p(d | theta) = integral N(h; d, C) N(h; mu, K) dh = N(d; mu, C + K)

where C is the data noise covariance, mu the GP predictive mean, and K the
GP predictive covariance.  When a PN mean function is used (h = h_PN + delta_h,
delta_h ~ N(m, K_delta)), the form is unchanged with mu_eff = h_PN + m and
K = K_delta.
"""
from __future__ import annotations

import math

import numpy as np
import torch

_LOG_2PI = math.log(2.0 * math.pi)


class MarginalLogLikelihood:
    """Evaluates log N(d; mu, C + K) — the GP-marginalised log-likelihood.

    All O(N³) work happens at construction time via the whitening factorisation

        C + K  =  L_C (I + A) L_C^T,    A = L_C^{-1} K L_C^{-T}

    so each __call__(d) costs only two O(N²) triangular solves.

    Parameters
    ----------
    C : array-like, shape (N, N)
        Data noise covariance matrix (symmetric positive-definite).
    mu : array-like, shape (N,)
        GP predictive mean vector.
    K : array-like, shape (N, N)
        GP predictive covariance matrix (symmetric positive-semidefinite).
    dtype : torch.dtype
        Floating-point precision; defaults to float64.
    device : str or torch.device
        Torch compute device; defaults to CPU.
    """

    def __init__(
        self,
        C,
        mu,
        K,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
        _L_C: torch.Tensor | None = None,
    ):
        dev = torch.device(device)
        _t = lambda a: torch.as_tensor(np.asarray(a, dtype=float), dtype=dtype, device=dev)
        mu_ = _t(mu)
        K_ = _t(K)

        # Factor C once; reused across all calls.
        # If a pre-computed Cholesky factor is supplied (e.g. by GWLikelihood which
        # holds a fixed noise covariance), the O(N³) factorisation is skipped.
        if _L_C is not None:
            self._L_C = _L_C
        else:
            C_ = _t(C)
            self._L_C = torch.linalg.cholesky(C_)
        log_det_C = 2.0 * self._L_C.diagonal().log().sum()

        # K == 0 fast path: whitening K would give A = 0, L_A = I exactly, so
        # skip the three extra O(N^3) dense ops entirely (two triangular
        # solves with an N×N RHS plus a second Cholesky) — pure overhead when
        # there's no waveform-uncertainty term to whiten. This is the common
        # case for a matched-filter (use_waveform_uncertainty=False) evaluation,
        # where callers still pass an explicit all-zero K for interface
        # uniformity; at real-data N this turns an O(N³) no-op into the O(N²)
        # it should be.
        self._k_is_zero = bool(torch.count_nonzero(K_) == 0)
        if self._k_is_zero:
            self._L_A = None
            self._log_det = float(log_det_C)
        else:
            # Whiten K: A = L_C^{-1} K L_C^{-T}.
            B = torch.linalg.solve_triangular(self._L_C, K_, upper=False)      # L_C B = K
            A = torch.linalg.solve_triangular(self._L_C, B.T, upper=False).T   # A = B L_C^{-T}

            # Factor I + A (always well-conditioned: I + PSD matrix).
            self._L_A = torch.linalg.cholesky(
                torch.eye(A.shape[0], dtype=dtype, device=dev) + A
            )

            # log|C + K| = log|C| + log|I + A| = 2 Σ log L_C_ii + 2 Σ log L_A_ii.
            self._log_det = float(log_det_C + 2.0 * self._L_A.diagonal().log().sum())
        self._n = self._L_C.shape[0]

        # Whitened mean: u_mu = L_C^{-1} mu  (fixed for the given C, mu).
        self._u_mu = torch.linalg.solve_triangular(
            self._L_C, mu_.unsqueeze(-1), upper=False
        ).squeeze(-1)

        self.dtype = dtype
        self.device = dev

    @property
    def log_det(self) -> float:
        """``log|C + K|`` for the factorisation this instance was built with."""
        return self._log_det

    @property
    def n(self) -> int:
        """Dimensionality ``N`` of the covariance."""
        return self._n

    def whiten(self, v) -> torch.Tensor:
        """Return ``L_A^{-1} L_C^{-1} v``, the whitening transform for this ``C + K``.

        For any two vectors ``v``, ``w``: ``v^T (C+K)^{-1} w == whiten(v) . whiten(w)``.
        Exposed so callers that need bilinear forms other than the residual
        Mahalanobis distance computed by :meth:`__call__` (e.g. analytic
        marginalisation over a parameter that enters the mean linearly, as
        coalescence phase does — see ``heron.inference.network``) can reuse this
        instance's O(N^3) factorisation instead of repeating it.
        """
        v_ = torch.as_tensor(np.asarray(v, dtype=float), dtype=self.dtype, device=self.device)
        u = torch.linalg.solve_triangular(
            self._L_C, v_.unsqueeze(-1), upper=False
        ).squeeze(-1)
        if self._k_is_zero:
            return u
        return torch.linalg.solve_triangular(
            self._L_A, u.unsqueeze(-1), upper=False
        ).squeeze(-1)

    def __call__(self, d) -> float:
        """Return the full scalar log density log N(d; mu, C + K).

        Parameters
        ----------
        d : array-like, shape (N,)
            Observed data vector.

        Returns
        -------
        float
            -0.5 (n log 2π + log|C+K| + r^T (C+K)^{-1} r), r = d - mu.
        """
        d_ = torch.as_tensor(np.asarray(d, dtype=float), dtype=self.dtype, device=self.device)

        # Whitened residual: u = L_C^{-1}(d - mu).
        u_d = torch.linalg.solve_triangular(
            self._L_C, d_.unsqueeze(-1), upper=False
        ).squeeze(-1)
        u = u_d - self._u_mu

        # Mahalanobis term: u^T (I+A)^{-1} u = ||L_A^{-1} u||^2. When K == 0,
        # A == 0 and L_A == I exactly, so the whitened residual itself is v.
        if self._k_is_zero:
            v = u
        else:
            v = torch.linalg.solve_triangular(
                self._L_A, u.unsqueeze(-1), upper=False
            ).squeeze(-1)

        quad = float(torch.dot(v, v))
        return -0.5 * (self._n * _LOG_2PI + self._log_det + quad)
