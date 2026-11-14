"""Tests for heron.models.gp.kronecker.KroneckerGP.

Every Kronecker shortcut is checked against the dense O(N^3) exact-GP algebra
it replaces, on a grid small enough to do densely.
"""
import numpy as np
import pytest
import torch

from heron.models.gp.kronecker import KroneckerGP, matern

LS = (0.3, 0.2)
SF2, SN2 = 1.3, 1e-3


@pytest.fixture(scope="module")
def grid():
    rng = np.random.default_rng(1)
    x1 = np.linspace(0.1, 0.9, 9)
    x2 = np.linspace(-1.0, 0.2, 40)
    y = np.sin(3 * x1)[:, None] * np.cos(5 * x2)[None, :] \
        + 0.01 * rng.standard_normal((9, 40))
    return x1, x2, y


def _k(a, b, ls):
    return matern(torch.as_tensor(np.atleast_1d(a), dtype=torch.float64),
                  torch.as_tensor(np.atleast_1d(b), dtype=torch.float64), ls, 2.5).numpy()


@pytest.fixture(scope="module")
def gp_and_dense(grid):
    x1, x2, y = grid
    gp = KroneckerGP(x1, x2, y)
    gp.set_hyperparameters(LS, SF2, SN2)
    Kn = SF2 * np.kron(_k(x1, x1, LS[0]), _k(x2, x2, LS[1])) + SN2 * np.eye(y.size)
    return gp, Kn


class TestDenseEquivalence:

    def test_log_marginal_likelihood(self, gp_and_dense):
        gp, Kn = gp_and_dense
        y = gp._y.ravel()
        L = np.linalg.cholesky(Kn)
        lml = -0.5 * (y @ np.linalg.solve(Kn, y) + 2 * np.log(np.diag(L)).sum()
                      + y.size * np.log(2 * np.pi))
        assert gp.log_marginal_likelihood() == pytest.approx(lml, rel=1e-10)

    def test_gradient_matches_finite_differences(self, gp_and_dense):
        gp, _ = gp_and_dense
        theta = gp._theta.copy()
        _, grad = gp._loss_and_grad(theta)
        eps = 1e-6
        fd = [(gp._loss_and_grad(theta + eps * e)[0] - gp._loss_and_grad(theta - eps * e)[0])
              / (2 * eps) for e in np.eye(4)]
        gp._set_theta(theta)
        np.testing.assert_allclose(grad, fd, rtol=1e-5)

    def test_posterior_mean_variance_and_covariance(self, gp_and_dense, grid):
        gp, Kn = gp_and_dense
        x1, x2, _ = grid
        xs, ts = 0.43, np.linspace(-0.9, 0.15, 25)
        ks = SF2 * np.kron(_k(xs, x1, LS[0]), _k(ts, x2, LS[1]))
        mean = ks @ np.linalg.solve(Kn, gp._y.ravel()) * gp.y_scale
        cov = (SF2 * _k(ts, ts, LS[1]) - ks @ np.linalg.solve(Kn, ks.T)) * gp.y_scale ** 2

        m, var = gp.posterior(xs, ts, mode="diagonal")
        _, full = gp.posterior(xs, ts, mode="full")
        np.testing.assert_allclose(m, mean, rtol=1e-7, atol=1e-9 * abs(mean).max())
        np.testing.assert_allclose(var, np.diag(cov), atol=1e-6 * np.diag(cov).max())
        np.testing.assert_allclose(full, cov, atol=1e-6 * abs(cov).max())


class TestLowRankFactor:

    def test_factor_approximates_full_covariance_inside_grid(self, gp_and_dense):
        gp, _ = gp_and_dense
        ts = np.linspace(-0.9, 0.15, 25)
        _, full = gp.posterior(0.43, ts, mode="full")
        _, F = gp.posterior(0.43, ts, mode="factor")
        # Error is the Nystrom residual of the *prior* (~1e-5 here), which is
        # a percent-level fraction of the much smaller *posterior* covariance.
        assert np.abs(F @ F.T - full).max() < 0.05 * np.abs(full).max()
        assert gp.nystrom_residual(ts).max() < 1e-4

    def test_factor_diagonal_never_exceeds_exact_variance(self, gp_and_dense):
        gp, _ = gp_and_dense
        ts = np.linspace(-1.5, 0.6, 60)  # partly outside the training range
        _, var = gp.posterior(0.43, ts, mode="diagonal")
        _, F = gp.posterior(0.43, ts, mode="factor")
        assert np.all(np.sum(F ** 2, axis=1) <= var * (1 + 1e-8) + 1e-14)

    def test_rank_truncation_is_near_optimal(self, gp_and_dense):
        """Top-r modes by grid eigen-weight capture nearly as much variance as
        the optimal (eigendecomposition) rank-r truncation of the exact
        posterior covariance at the query points."""
        gp, _ = gp_and_dense
        ts = np.linspace(-0.9, 0.15, 25)
        _, full = gp.posterior(0.43, ts, mode="full")
        w = np.sort(np.linalg.eigvalsh(full))[::-1]
        _, F_all = gp.posterior(0.43, ts, mode="factor")
        for r in (3, 5, 10):
            _, F = gp.posterior(0.43, ts, mode="factor", rank=r)
            assert F.shape[1] == r
            ours = np.sum(F ** 2) / np.sum(F_all ** 2)
            optimal = w[:r].sum() / w.sum()
            assert ours > 0.95 * optimal


class TestFitting:

    def test_grid_derived_floors(self, grid):
        x1, x2, y = grid
        gp = KroneckerGP(x1, x2, y)
        assert gp.ls_min[0] == pytest.approx(np.diff(x1).max())
        assert gp.ls_min[1] == pytest.approx(2 * np.diff(x2).max())

    def test_fit_improves_likelihood_and_respects_floors(self, grid):
        x1, x2, y = grid
        gp = KroneckerGP(x1, x2, y)
        before = gp.log_marginal_likelihood()
        gp.fit()
        assert gp.log_marginal_likelihood() > before
        assert gp.lengthscales[0] >= gp.ls_min[0] * (1 - 1e-9)
        assert gp.lengthscales[1] >= gp.ls_min[1] * (1 - 1e-9)

    def test_state_roundtrip(self, grid):
        x1, x2, y = grid
        gp = KroneckerGP(x1, x2, y).fit(maxiter=30)
        gp2 = KroneckerGP.from_state(gp.state())
        ts = np.linspace(-0.9, 0.15, 11)
        for mode in ("diagonal", "factor"):
            a = gp.posterior(0.5, ts, mode=mode)
            b = gp2.posterior(0.5, ts, mode=mode)
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(a[1], b[1])

    def test_shape_validation(self, grid):
        x1, x2, y = grid
        with pytest.raises(ValueError):
            KroneckerGP(x1, x2, y.T)
