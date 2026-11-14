"""Exact GP regression on a 2-D product grid via Kronecker algebra.

When the training inputs form a Cartesian product ``X1 x X2`` (``n1`` values
of the first coordinate, e.g. mass ratio, times ``n2`` values of the second,
e.g. warped time), a separable kernel

    k((a, s), (a', s')) = sigma_f^2 * k1(a, a') * k2(s, s')

gives a training covariance ``sigma_f^2 (K1 kron K2) + sigma_n^2 I``. With the
per-factor eigendecompositions ``K1 = Q1 L1 Q1^T`` and ``K2 = Q2 L2 Q2^T`` the
whole matrix is diagonalised by ``Q1 kron Q2`` with eigenvalues
``sigma_f^2 l1_i l2_j + sigma_n^2``. Every quantity exact GP regression needs
-- the solve ``(K + sigma_n^2 I)^{-1} y``, ``log|K + sigma_n^2 I|``, the
posterior mean and variance -- then costs ``O(n1^3 + n2^3 + n1 n2 (n1 + n2))``
instead of ``O((n1 n2)^3)``: a 30 x 200 grid is ~30^3 + 200^3 ~ 8e6 flops of
factorisation instead of 6000^3 ~ 2e11, with no Cholesky-size limit, no GPU,
and no CG.

This is the standard construction (Saatci 2012, "Scalable inference for
structured Gaussian process models"; Wilson et al. 2014). Two details are
specific to this implementation:

**Hyperparameter gradients without differentiating ``eigh``.** Autograd
through ``torch.linalg.eigh`` is unstable at (near-)degenerate eigenvalues,
which smooth kernels on dense grids have in abundance. Instead the loss
gradient is obtained from two *surrogate* scalars whose gradients equal the
true ones, using the identities

    d(y^T Kn^-1 y) = -alpha^T dK alpha              (alpha detached)
    d(log|Kn|)     = tr(Kn^-1 dK)                   (Kn^-1 detached)

and evaluating both in the Kronecker eigenbasis with the eigenvectors held
fixed. Autograd then only differentiates the small factor kernel matrices
``K1(theta)``, ``K2(theta)``. (For a degenerate eigenvalue group the weights
``1 / (sigma_f^2 l1_i l2_j + sigma_n^2)`` are equal across the group, so the
trace is basis-invariant within it and the gradient is well-defined.)

**A low-rank posterior covariance factor.** :meth:`KroneckerGP.posterior`
can return ``F`` with ``Cov ~= F F^T`` at the query points, built from a
Nystrom approximation on the training grid of the second coordinate (see
:meth:`KroneckerGP.posterior` for the derivation). Its rank is at most
``n2`` (and in practice a few tens of modes carry all the variance), which is
what lets the marginal likelihood use a Woodbury solve instead of a dense
``O(N^3)`` factorisation of ``C + K`` -- see
:class:`heron.likelihood.LowRankMarginalLogLikelihood`.

Hyperparameters (two lengthscales, signal variance, noise variance) are fit by
maximising the exact log marginal likelihood with L-BFGS-B, in log space, with
**lengthscale floors derived from the training grid itself** (by default: the
largest grid spacing in coordinate 1, twice the largest spacing in coordinate
2, the rules of thumb documented in CLAUDE.md) instead of hand-set per config.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import torch
from scipy.optimize import minimize

logger = logging.getLogger("heron.models.gp.kronecker")

_LOG_2PI = math.log(2.0 * math.pi)


def matern(x: torch.Tensor, y: torch.Tensor, lengthscale, nu: float) -> torch.Tensor:
    """Unit-variance 1-D Matern kernel matrix ``k(x_i, y_j)`` (torch, any dtype)."""
    r = torch.abs(x[:, None] - y[None, :]) / lengthscale
    if nu == 0.5:
        return torch.exp(-r)
    if nu == 1.5:
        a = math.sqrt(3.0) * r
        return (1.0 + a) * torch.exp(-a)
    if nu == 2.5:
        a = math.sqrt(5.0) * r
        return (1.0 + a + a * a / 3.0) * torch.exp(-a)
    raise ValueError(f"nu must be 0.5, 1.5 or 2.5; got {nu}")


def _max_spacing(x: np.ndarray) -> float:
    xs = np.sort(np.asarray(x, dtype=float))
    if len(xs) < 2:
        return 1.0
    return float(np.max(np.diff(xs)))


class KroneckerGP:
    """Exact zero-mean GP on a product grid ``X1 x X2`` with a separable Matern kernel.

    Parameters
    ----------
    x1 : array, shape (n1,)
        Grid of the first coordinate (kernel coordinates -- apply any input
        warping before passing it in).
    x2 : array, shape (n2,)
        Grid of the second coordinate (likewise already warped).
    y : array, shape (n1, n2)
        Targets, ``y[i, j] = f(x1[i], x2[j])``.
    nu : float
        Matern smoothness (0.5, 1.5 or 2.5) for both factors.
    ls_min : (float, float) or None
        Lengthscale floors. ``None`` entries are derived from the grid:
        ``ls_min_factor[k] * max spacing of x_k``.
    ls_min_factor : (float, float)
        Multipliers for grid-derived floors (default ``(1, 2)``).
    noise_floor_rel : float
        Floor on the noise variance relative to ``var(y)``.
    """

    def __init__(
        self,
        x1,
        x2,
        y,
        nu: float = 2.5,
        ls_min: tuple[float | None, float | None] = (None, None),
        ls_min_factor: tuple[float, float] = (1.0, 2.0),
        noise_floor_rel: float = 1e-6,
    ):
        self.x1 = np.asarray(x1, dtype=np.float64)
        self.x2 = np.asarray(x2, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.shape != (len(self.x1), len(self.x2)):
            raise ValueError(
                f"y must have shape (len(x1), len(x2)) = "
                f"{(len(self.x1), len(self.x2))}; got {y.shape}"
            )
        self.nu = float(nu)
        # Internal target scaling: makes sigma_f^2 ~ 1 whatever the units of
        # y, so no hand-set output_scale is needed.
        std = float(y.std())
        self.y_scale = std if std > 0 else 1.0
        self._y = y / self.y_scale

        floors = []
        for k, (x, given) in enumerate(((self.x1, ls_min[0]), (self.x2, ls_min[1]))):
            floors.append(float(given) if given is not None
                          else ls_min_factor[k] * _max_spacing(x))
        self.ls_min = tuple(floors)
        self.noise_floor_rel = float(noise_floor_rel)

        # Initial hyperparameters (set properly by fit() or set_hyperparameters()).
        self._set_theta(self._initial_theta())

    # -- hyperparameters ---------------------------------------------------

    def _ranges(self):
        r1 = float(np.ptp(self.x1)) or 1.0
        r2 = float(np.ptp(self.x2)) or 1.0
        return r1, r2

    def _bounds(self):
        r1, r2 = self._ranges()
        nf = max(self.noise_floor_rel, 1e-12)
        return [
            (math.log(self.ls_min[0]), math.log(max(20.0 * r1, 2 * self.ls_min[0]))),
            (math.log(self.ls_min[1]), math.log(max(20.0 * r2, 2 * self.ls_min[1]))),
            (math.log(1e-6), math.log(1e3)),
            (math.log(nf), math.log(1.0)),
        ]

    def _initial_theta(self) -> np.ndarray:
        r1, r2 = self._ranges()
        # Geometric mean of floor and range: away from both bounds, so the
        # optimiser does not start pinned on the floor.
        ls1 = math.sqrt(self.ls_min[0] * max(r1, self.ls_min[0]))
        ls2 = math.sqrt(self.ls_min[1] * max(r2, self.ls_min[1]))
        noise = max(self.noise_floor_rel, 1e-4)
        return np.log([ls1, ls2, 1.0, noise])

    def _set_theta(self, theta):
        theta = np.asarray(theta, dtype=np.float64)
        self._theta = theta.copy()
        ls1, ls2, sf2, sn2 = np.exp(theta)
        self.lengthscales = (float(ls1), float(ls2))
        self.signal_variance = float(sf2)
        self.noise_variance = float(sn2)
        self._factorise()

    def set_hyperparameters(self, lengthscales, signal_variance, noise_variance):
        """Set hyperparameters directly (signal/noise variance in *scaled* units)."""
        self._set_theta(np.log([lengthscales[0], lengthscales[1],
                                signal_variance, noise_variance]))

    @property
    def hyperparameters(self) -> dict:
        """Hyperparameters, with variances in the units of ``y``."""
        s2 = self.y_scale ** 2
        return {
            "lengthscale_1": self.lengthscales[0],
            "lengthscale_2": self.lengthscales[1],
            "signal_variance": self.signal_variance * s2,
            "noise_variance": self.noise_variance * s2,
            "ls_min_1": self.ls_min[0],
            "ls_min_2": self.ls_min[1],
        }

    # -- core algebra ------------------------------------------------------

    def _factor_kernels(self, ls1, ls2):
        x1 = torch.as_tensor(self.x1, dtype=torch.float64)
        x2 = torch.as_tensor(self.x2, dtype=torch.float64)
        return matern(x1, x1, ls1, self.nu), matern(x2, x2, ls2, self.nu)

    def _factorise(self):
        """Eigendecompose the factor kernels and form the solve at current theta."""
        ls1, ls2 = self.lengthscales
        K1, K2 = self._factor_kernels(ls1, ls2)
        l1, Q1 = np.linalg.eigh(K1.numpy())
        l2, Q2 = np.linalg.eigh(K2.numpy())
        self._l1 = np.clip(l1, 0.0, None)
        self._l2 = np.clip(l2, 0.0, None)
        self._Q1, self._Q2 = Q1, Q2
        self._D = self.signal_variance * np.outer(self._l1, self._l2) + self.noise_variance
        y_rot = Q1.T @ self._y @ Q2
        self._quad = float(np.sum(y_rot ** 2 / self._D))
        self._logdet = float(np.sum(np.log(self._D)))
        # alpha = Kn^{-1} y, reshaped to (n1, n2).
        self._A = Q1 @ (y_rot / self._D) @ Q2.T

    def log_marginal_likelihood(self) -> float:
        """Exact log p(y | theta) in the *scaled* target units."""
        n = self._y.size
        return -0.5 * (self._quad + self._logdet + n * _LOG_2PI)

    def _loss_and_grad(self, theta):
        self._set_theta(theta)
        loss = -self.log_marginal_likelihood()

        th = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
        ls1, ls2, sf2, sn2 = torch.exp(th)
        K1, K2 = self._factor_kernels(ls1, ls2)
        A = torch.as_tensor(self._A)
        Q1 = torch.as_tensor(self._Q1)
        Q2 = torch.as_tensor(self._Q2)
        Dinv = torch.as_tensor(1.0 / self._D)
        # 0.5 * d(quad) = -0.5 * alpha^T dK alpha ; K alpha = sf2 K1 A K2 + sn2 A.
        t_quad = -0.5 * (sf2 * torch.sum(A * (K1 @ A @ K2)) + sn2 * torch.sum(A * A))
        # 0.5 * d(logdet) = 0.5 * tr(Kn^-1 dK), in the (fixed) eigenbasis.
        a1 = torch.sum(Q1 * (K1 @ Q1), dim=0)  # diag(Q1^T K1 Q1)
        a2 = torch.sum(Q2 * (K2 @ Q2), dim=0)
        t_logdet = 0.5 * torch.sum((sf2 * a1[:, None] * a2[None, :] + sn2) * Dinv)
        (t_quad + t_logdet).backward()
        return loss, th.grad.numpy().copy()

    def fit(self, maxiter: int = 200, n_restarts: int = 0, seed: int = 0) -> "KroneckerGP":
        """Maximise the exact log marginal likelihood (L-BFGS-B, log space).

        ``n_restarts`` extra random starts inside the bounds guard against the
        multimodality GP marginal likelihoods often have in the lengthscales.
        """
        bounds = self._bounds()
        starts = [np.clip(self._initial_theta(), [b[0] for b in bounds],
                          [b[1] for b in bounds])]
        rng = np.random.default_rng(seed)
        for _ in range(n_restarts):
            starts.append(np.array([rng.uniform(lo, hi) for lo, hi in bounds]))

        best = None
        for x0 in starts:
            res = minimize(self._loss_and_grad, x0, jac=True, method="L-BFGS-B",
                           bounds=bounds, options={"maxiter": maxiter})
            if best is None or res.fun < best.fun:
                best = res
        self._set_theta(best.x)
        at_floor = [
            name for name, val, b in (
                ("lengthscale_1", best.x[0], bounds[0]),
                ("lengthscale_2", best.x[1], bounds[1]),
            ) if abs(val - b[0]) < 1e-3
        ]
        logger.info(
            "KroneckerGP fit: -logML %.4g, ls=(%.4g, %.4g), sf2=%.3g, sn2=%.3g%s",
            best.fun, *self.lengthscales, self.signal_variance, self.noise_variance,
            f" [at floor: {', '.join(at_floor)}]" if at_floor else "",
        )
        self.fit_result = best
        return self

    # -- prediction --------------------------------------------------------

    def posterior(self, x1_star: float, x2_star, mode: str = "diagonal",
                  rank: int | None = None, eig_rtol: float = 1e-12):
        """Posterior at one value of coordinate 1 and many of coordinate 2.

        Parameters
        ----------
        x1_star : float
            Query value of the first coordinate (kernel coordinates).
        x2_star : array, shape (m,)
            Query values of the second coordinate.
        mode : {'none', 'diagonal', 'factor', 'full'}
            What to return alongside the mean (see Returns).
        rank : int or None
            For ``mode='factor'``: keep at most this many modes (largest
            variance first). ``None`` keeps every numerically non-negligible
            mode (at most ``n2``).
        eig_rtol : float
            Factor modes whose ``K2`` eigenvalue is below ``eig_rtol * max``
            are dropped (they carry no variance and are numerically noisy).

        Returns
        -------
        mean : ndarray (m,)
        second : None | ndarray (m,) | ndarray (m, r) | ndarray (m, m)
            ``'diagonal'``: exact posterior variance. ``'factor'``: ``F`` with
            ``Cov ~= F F^T``. ``'full'``: exact dense posterior covariance.

        Notes
        -----
        With unit factor kernels, ``k1* = k1(x1*, X1)``, ``K2* = k2(x2*, X2)``,
        ``u1 = Q1^T k1*``, ``U2 = K2* Q2`` and ``M_j = sum_i u1_i^2 / D_ij``,
        the exact posterior covariance is

            Cov = sf2 * k2(x2*, x2*) - sf2^2 * U2 diag(M) U2^T.

        Replacing the prior term by its Nystrom approximation on the training
        grid ``X2`` (``k2(x2*, x2*) ~= U2 L2^{-1} U2^T``) makes the whole thing
        a congruence of one diagonal core,

            Cov ~= sum_j c_j b_j b_j^T,   b_j = U2[:, j] / sqrt(l2_j),
                                          c_j = sf2 - sf2^2 l2_j M_j  (>= 0),

        so ``F = [b_j sqrt(c_j)]``. The only approximation is the Nystrom
        residual ``sf2 * (1 - sum_j b_j^2)`` on the diagonal, which is ~0 for
        query points inside the training range of ``x2`` (lengthscale above
        the grid spacing) and grows to the full prior variance far outside it
        -- where the factor, unlike the exact diagonal, decays to zero. See
        :meth:`nystrom_residual`.
        """
        sf2 = self.signal_variance
        x2s = torch.as_tensor(np.atleast_1d(np.asarray(x2_star, dtype=np.float64)))
        k1s = matern(torch.as_tensor([float(x1_star)], dtype=torch.float64),
                     torch.as_tensor(self.x1), self.lengthscales[0], self.nu)[0].numpy()
        K2s = matern(x2s, torch.as_tensor(self.x2), self.lengthscales[1], self.nu).numpy()

        mean = sf2 * (K2s @ (self._A.T @ k1s)) * self.y_scale
        if mode == "none":
            return mean, None

        u1 = self._Q1.T @ k1s
        U2 = K2s @ self._Q2
        M = (u1 ** 2) @ (1.0 / self._D)  # (n2,)
        s2 = self.y_scale ** 2

        if mode == "diagonal":
            var = sf2 - sf2 ** 2 * ((U2 ** 2) @ M)
            return mean, np.clip(var, 0.0, None) * s2

        if mode == "full":
            K2ss = matern(x2s, x2s, self.lengthscales[1], self.nu).numpy()
            cov = sf2 * K2ss - sf2 ** 2 * (U2 * M[None, :]) @ U2.T
            return mean, 0.5 * (cov + cov.T) * s2

        if mode == "factor":
            keep = self._l2 > eig_rtol * self._l2.max()
            l2 = self._l2[keep]
            B = U2[:, keep] / np.sqrt(l2)[None, :]
            c = np.clip(sf2 - sf2 ** 2 * l2 * M[keep], 0.0, None)
            # b_j restricted to the training grid is Q2[:, j] sqrt(l2_j): the modes
            # are orthogonal there with squared norm l2_j, so c_j l2_j is the
            # exact eigen-weight and the top-r cut is the optimal rank-r
            # truncation (in the grid norm), independent of the query window.
            weight = c * l2
            order = np.argsort(weight)[::-1]
            if rank is not None:
                order = order[:rank]
            F = B[:, order] * np.sqrt(c[order])[None, :]
            return mean, F * self.y_scale

        raise ValueError(f"mode must be 'none', 'diagonal', 'factor' or 'full'; got {mode!r}")

    def nystrom_residual(self, x2_star) -> np.ndarray:
        """``1 - k2*^T K2^{-1} k2*``: fraction of prior variance the factor drops."""
        x2s = torch.as_tensor(np.atleast_1d(np.asarray(x2_star, dtype=np.float64)))
        K2s = matern(x2s, torch.as_tensor(self.x2), self.lengthscales[1], self.nu).numpy()
        keep = self._l2 > 1e-12 * self._l2.max()
        U2 = (K2s @ self._Q2)[:, keep]
        return 1.0 - np.sum(U2 ** 2 / self._l2[keep][None, :], axis=1)

    # -- persistence -------------------------------------------------------

    def state(self) -> dict:
        """Everything needed to rebuild this GP exactly (plain numpy/python)."""
        return {
            "x1": self.x1, "x2": self.x2, "y": self._y * self.y_scale,
            "nu": self.nu, "ls_min": list(self.ls_min),
            "noise_floor_rel": self.noise_floor_rel, "theta": self._theta,
        }

    @classmethod
    def from_state(cls, state: dict) -> "KroneckerGP":
        gp = cls(state["x1"], state["x2"], state["y"], nu=state["nu"],
                 ls_min=tuple(state["ls_min"]),
                 noise_floor_rel=state["noise_floor_rel"])
        gp._set_theta(state["theta"])
        return gp
