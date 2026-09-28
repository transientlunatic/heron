"""Unit tests for MarginalLogLikelihood (whitening optimisation)."""
import numpy as np
import pytest
from numpy.linalg import slogdet, solve

from heron.likelihood import MarginalLogLikelihood


def _oracle(d, mu, C, K):
    S = C + K
    r = np.asarray(d) - np.asarray(mu)
    n = len(r)
    _, logdet = slogdet(S)
    return -0.5 * (n * np.log(2 * np.pi) + logdet + r @ solve(S, r))


def _random_pd(n, rng, scale=1.0):
    A = rng.standard_normal((n, n))
    return scale * (A @ A.T + n * np.eye(n))


# ---------------------------------------------------------------------------
# Pre-computed internals
# ---------------------------------------------------------------------------

class TestPrecomputedInternals:
    """The O(N³) work is done at construction; verify each quantity."""

    @pytest.fixture(autouse=True)
    def build(self):
        rng = np.random.default_rng(42)
        n = 5
        self.C = _random_pd(n, rng, scale=3.0)
        self.K = _random_pd(n, rng, scale=0.5)
        self.mu = rng.standard_normal(n)
        self.ll = MarginalLogLikelihood(self.C, self.mu, self.K)

    def test_logdet_matches_numpy(self):
        _, ref = slogdet(self.C + self.K)
        assert self.ll._log_det == pytest.approx(ref, rel=1e-10)

    def test_whitened_mu_matches_numpy(self):
        L_C = np.linalg.cholesky(self.C)
        u_mu_ref = solve(L_C, self.mu)
        np.testing.assert_allclose(self.ll._u_mu.numpy(), u_mu_ref, rtol=1e-10)

    def test_factorisation_round_trip(self):
        # C + K == L_C (I + A) L_C^T must hold to floating-point precision.
        L_C = self.ll._L_C.numpy()
        L_A = self.ll._L_A.numpy()
        reconstructed = L_C @ (L_A @ L_A.T) @ L_C.T
        np.testing.assert_allclose(reconstructed, self.C + self.K, rtol=1e-10)

    def test_l_a_is_lower_triangular(self):
        L_A = self.ll._L_A.numpy()
        np.testing.assert_allclose(np.triu(L_A, k=1), 0.0, atol=1e-15)
        assert np.all(np.diag(L_A) > 0)


# ---------------------------------------------------------------------------
# Correctness: matches oracle for varied inputs
# ---------------------------------------------------------------------------

class TestCorrectness:

    @pytest.mark.parametrize("n", [3, 8, 20])
    def test_matches_oracle(self, n):
        rng = np.random.default_rng(n * 7)
        C = _random_pd(n, rng, scale=5.0)
        K = _random_pd(n, rng, scale=1.0)
        mu = rng.standard_normal(n)
        d = rng.standard_normal(n)
        assert MarginalLogLikelihood(C, mu, K)(d) == pytest.approx(_oracle(d, mu, C, K), rel=1e-10)

    def test_k_near_zero_recovers_standard(self):
        rng = np.random.default_rng(7)
        n = 4
        C = _random_pd(n, rng)
        K_small = _random_pd(n, rng) * 1e-12
        mu = rng.standard_normal(n)
        d = rng.standard_normal(n)
        got = MarginalLogLikelihood(C, mu, K_small)(d)
        ref = _oracle(d, mu, C, K_small)
        assert got == pytest.approx(ref, rel=1e-8)

    def test_symmetry_d_mu(self):
        rng = np.random.default_rng(13)
        n = 4
        C = _random_pd(n, rng)
        K = _random_pd(n, rng)
        d = rng.standard_normal(n)
        mu = rng.standard_normal(n)
        a = MarginalLogLikelihood(C, mu, K)(d)
        b = MarginalLogLikelihood(C, d, K)(mu)
        assert a == pytest.approx(b, rel=1e-10)

    def test_d_equals_mu_is_peak(self):
        # At d == mu the quadratic term vanishes; loglike == -0.5(n log2π + log|C+K|).
        rng = np.random.default_rng(21)
        n = 4
        C = _random_pd(n, rng)
        K = _random_pd(n, rng)
        mu = rng.standard_normal(n)
        ll = MarginalLogLikelihood(C, mu, K)
        _, logdet = slogdet(C + K)
        expected = -0.5 * (n * np.log(2 * np.pi) + logdet)
        assert ll(mu) == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# Repeated calls don't corrupt pre-computed state
# ---------------------------------------------------------------------------

class TestRepeatedCalls:

    def test_multiple_d_vectors_match_oracle(self):
        rng = np.random.default_rng(99)
        n = 5
        C = _random_pd(n, rng, scale=3.0)
        K = _random_pd(n, rng, scale=0.5)
        mu = rng.standard_normal(n)
        ll = MarginalLogLikelihood(C, mu, K)
        for _ in range(20):
            d = rng.standard_normal(n)
            assert ll(d) == pytest.approx(_oracle(d, mu, C, K), rel=1e-10)

    def test_logdet_unchanged_after_calls(self):
        rng = np.random.default_rng(55)
        n = 4
        C = _random_pd(n, rng)
        K = _random_pd(n, rng)
        mu = rng.standard_normal(n)
        ll = MarginalLogLikelihood(C, mu, K)
        logdet_before = ll._log_det
        for _ in range(5):
            ll(rng.standard_normal(n))
        assert ll._log_det == logdet_before
