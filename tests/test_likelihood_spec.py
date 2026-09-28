"""
Tier-0 likelihood specification tests for heron's model-uncertainty likelihood.

WHAT THIS FILE IS
-----------------
A *specification*: it encodes the correct mathematical behaviour of the
marginalized likelihood

    p(d | theta) = integral N(h; d, C) N(h; mu, K) dh
                 = N(d; mu, C + K)                                     (Eq. 5)

as an INDEPENDENT pure-numpy oracle (`_ref_*` functions below) plus a set of
FROZEN reference values (computed once, offline, and checked by brute-force
Gauss-Hermite integration to rtol 1e-4). The reference numbers are baked in as
literals on purpose: a test that recomputes its expected value *from the code
under test* is circular and passes even when that code is wrong. These literals
are the external ground truth.

WHAT CLAUDE CODE NEEDS TO DO
----------------------------
Each test currently calls `heron_marginal_loglike(...)` / `heron_predict(...)`,
which are thin ADAPTERS defined at the top of this file and marked
`raise NotImplementedError`. Bind each adapter to the REAL heron API on the
`update-htcondor` branch (likely in `heron/models/gpytorch.py` and
`heron/likelihood.py` / wherever the likelihood lives). Do NOT touch the
`_ref_*` oracle or the frozen literals — those are the spec. When the adapters
are correctly bound, every test must pass with the tolerances as written.

See `CLAUDE_CODE_BRIEF.md` (same directory) for the binding checklist.
"""
import numpy as np
import pytest

# ============================================================================
# ADAPTERS — Claude Code binds these to the real heron API. Everything else
# in this file is the frozen specification and must not change.
# ============================================================================

def heron_marginal_loglike(d, mu, C, K):
    """Return the scalar log marginal likelihood log N(d; mu, C+K) as computed
    by heron's *actual* likelihood object.
    """
    from heron.likelihood import MarginalLogLikelihood
    return MarginalLogLikelihood(C=C, mu=mu, K=K)(d)


def heron_predict(model, x):
    """Return (mu, K): GP predictive mean vector and full predictive covariance
    matrix at input locations x, from a trained heron GP model.

    x is a parameters dict accepted by ExactGPSurrogate.predict().
    Returns the plus-polarisation mean and covariance.
    """
    wf = model.predict(x)
    return wf["plus"].data, wf["plus"].covariance


# ============================================================================
# ORACLE — independent pure-numpy reference. DO NOT bind to heron. DO NOT edit.
# ============================================================================

def _ref_marginal_loglike(d, mu, C, K):
    S = C + K
    r = np.asarray(d) - np.asarray(mu)
    n = len(r)
    _, logdet = np.linalg.slogdet(S)
    return -0.5 * (n * np.log(2 * np.pi) + logdet + r @ np.linalg.solve(S, r))


def _ref_posterior_centre(d, mu, C, K):
    """Correct completed-square centre: data weighted by K(C+K)^-1, model by
    C(C+K)^-1. (The draft Appendix A had these weights swapped.)"""
    CpKi = np.linalg.inv(C + K)
    return K @ CpKi @ d + C @ CpKi @ mu


# ============================================================================
# FROZEN FIXTURES — computed offline, verified by brute-force integration.
# ============================================================================

C_FIX = np.array([
        [6.12532121, 1.968879487, 1.197970813],
        [1.968879487, 12.25549321, -4.262668228],
        [1.197970813, -4.262668228, 8.85684504]
    ])
K_FIX = np.array([
        [2.053238667, -0.8946692018, 0.5317952993],
        [-0.8946692018, 2.244217529, -0.0438216947],
        [0.5317952993, -0.0438216947, 4.342066698]
    ])
D_FIX = np.array([0.5201341562, -1.002165794, 0.2683455404])
MU_FIX = np.array([0.7671747005, 1.191272027, -1.157410807])

LL_MARGINAL      = -6.5507353187
LL_STANDARD      = -6.0717737025      # K = 0 case
LL_KZERO_LIMIT   = -6.0717737025   # K -> 0 must recover standard
Z_CENTRE         = np.array([0.8615413883, 0.9250409763, -0.8637013297])
RESID_QUADFORM   = 0.3882218934
DET_CHAIN_LHS    = 1338.91858995
DET_CHAIN_RHS    = 1338.91858995

H_PHENOM = np.array([0.6962793953, 0.3513836858, -0.032415083])
M_DELTA  = np.array([0.000659079, -0.0339624985, -0.0310266014])
K_DELTA  = np.array([
        [0.1014191086, -0.0660119943, -0.0490032434],
        [-0.0660119943, 0.2046583806, 0.0789171772],
        [-0.0490032434, 0.0789171772, 0.1419691587]
    ])
LL_MEANFUNC = -5.9564455073

ATOL = 1e-9
RTOL = 1e-7


# ============================================================================
# TESTS
# ============================================================================

class TestOracleSelfConsistency:
    """Guard the spec itself: the frozen literals must match the oracle.
    These do NOT touch heron and should pass immediately on checkout."""

    def test_marginal_matches_literal(self):
        assert _ref_marginal_loglike(D_FIX, MU_FIX, C_FIX, K_FIX) == pytest.approx(LL_MARGINAL, abs=ATOL)

    def test_standard_matches_literal(self):
        Z = np.zeros_like(C_FIX)
        assert _ref_marginal_loglike(D_FIX, MU_FIX, C_FIX, Z) == pytest.approx(LL_STANDARD, abs=ATOL)

    def test_determinant_chain_identity(self):
        # |C||K||C^-1 + K^-1| == |C+K|
        assert DET_CHAIN_LHS == pytest.approx(DET_CHAIN_RHS, rel=1e-6)

    def test_posterior_centre_matches_literal(self):
        z = _ref_posterior_centre(D_FIX, MU_FIX, C_FIX, K_FIX)
        assert np.allclose(z, Z_CENTRE, atol=ATOL)


class TestHeronLikelihood:
    """The real assertions: heron's likelihood == the oracle.
    Fail with NotImplementedError until the adapters are bound."""

    def test_marginal_loglike_value(self):
        got = heron_marginal_loglike(D_FIX, MU_FIX, C_FIX, K_FIX)
        assert got == pytest.approx(LL_MARGINAL, rel=RTOL)

    def test_reduces_to_standard_when_K_zero(self):
        # K -> 0 must recover the standard GW likelihood exactly.
        got = heron_marginal_loglike(D_FIX, MU_FIX, C_FIX, K_FIX * 1e-12)
        assert got == pytest.approx(LL_KZERO_LIMIT, rel=1e-5)

    def test_broadening_lowers_peak_density(self):
        # (C+K) >= C  =>  |C+K| >= |C|  =>  marginal peak density <= standard.
        # At d = mu (residual zero) the marginal loglike must be <= standard loglike.
        marg = heron_marginal_loglike(MU_FIX, MU_FIX, C_FIX, K_FIX)
        std  = heron_marginal_loglike(MU_FIX, MU_FIX, C_FIX, K_FIX * 1e-12)
        assert marg <= std + 1e-9

    def test_symmetry_under_data_model_swap(self):
        # N(d; mu, C+K) is symmetric in (d <-> mu).
        a = heron_marginal_loglike(D_FIX, MU_FIX, C_FIX, K_FIX)
        b = heron_marginal_loglike(MU_FIX, D_FIX, C_FIX, K_FIX)
        assert a == pytest.approx(b, rel=RTOL)

    @pytest.mark.parametrize("eps", [1e-4, 1e-5, 1e-6])
    def test_gradient_matches_finite_difference(self, eps):
        # d(logL)/d(mu_i) analytic == central finite difference.
        # Oracle gradient: (C+K)^-1 (d - mu).
        analytic = np.linalg.solve(C_FIX + K_FIX, D_FIX - MU_FIX)
        fd = np.zeros_like(MU_FIX)
        for i in range(len(MU_FIX)):
            mp = MU_FIX.copy(); mp[i] += eps
            mm = MU_FIX.copy(); mm[i] -= eps
            fp = heron_marginal_loglike(D_FIX, mp, C_FIX, K_FIX)
            fm = heron_marginal_loglike(D_FIX, mm, C_FIX, K_FIX)
            fd[i] = (fp - fm) / (2 * eps)
        assert np.allclose(analytic, fd, atol=1e-3, rtol=1e-3)


class TestMeanFunctionDecomposition:
    """PR #60: with IMRPhenomD as the GP mean, h = h_PhenomD + delta_h,
    delta_h ~ N(m_delta, K_delta), so the likelihood must equal
    N(d; h_PhenomD + m_delta, C + K_delta)."""

    def test_meanfunc_effective_mu(self):
        mu_eff = H_PHENOM + M_DELTA
        got = heron_marginal_loglike(D_FIX, mu_eff, C_FIX, K_DELTA)
        assert got == pytest.approx(LL_MEANFUNC, rel=RTOL)

    def test_meanfunc_equals_full_mean_form(self):
        # Passing (h_PhenomD + m_delta, K_delta) must equal passing the combined
        # mean directly — i.e. the decomposition is exact, not approximate.
        mu_eff = H_PHENOM + M_DELTA
        via_decomp = heron_marginal_loglike(D_FIX, mu_eff, C_FIX, K_DELTA)
        assert via_decomp == pytest.approx(LL_MEANFUNC, rel=RTOL)

    def test_predictive_covariance_is_psd(self, tiny_gp):
        x = {"mass_ratio": 0.7, "time": {"lower": -0.3, "upper": 0.02, "number": 10}}
        mu, K = heron_predict(tiny_gp, x)
        assert np.allclose(K, K.T, atol=1e-6)
        assert np.linalg.eigvalsh(K).min() > -1e-8