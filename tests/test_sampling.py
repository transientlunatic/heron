"""Tests for heron.sampling — prior, result, and DynestySampler."""
import numpy as np
import pytest

from heron.sampling import (
    DynestySampler,
    Parameter,
    SamplerResult,
    UniformPrior,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prior():
    return UniformPrior([
        Parameter("mass_ratio", 0.5, 1.0),
        Parameter("tc",         1187008880.0, 1187008885.0),
        Parameter("ra",         0.0, 2 * np.pi),
    ])


def _gaussian_loglike(params: dict) -> float:
    """3-D Gaussian centred at (0.75, 1187008882.4, 1.95) in prior coords."""
    q   = params["mass_ratio"] - 0.75
    tc  = params["tc"]         - 1187008882.4
    ra  = params["ra"]         - 1.95
    return -0.5 * (q**2 / 0.01**2 + tc**2 / 0.01**2 + ra**2 / 0.1**2)


# ---------------------------------------------------------------------------
# UniformPrior
# ---------------------------------------------------------------------------

class TestUniformPrior:

    def test_ndim(self):
        assert _make_prior().ndim == 3

    def test_parameter_names(self):
        assert _make_prior().parameter_names == ["mass_ratio", "tc", "ra"]

    def test_transform_lower_corner(self):
        prior = _make_prior()
        theta = prior.transform(np.zeros(3))
        assert theta[0] == pytest.approx(0.5)
        assert theta[1] == pytest.approx(1187008880.0)
        assert theta[2] == pytest.approx(0.0)

    def test_transform_upper_corner(self):
        prior = _make_prior()
        theta = prior.transform(np.ones(3))
        assert theta[0] == pytest.approx(1.0)
        assert theta[1] == pytest.approx(1187008885.0)
        assert theta[2] == pytest.approx(2 * np.pi)

    def test_transform_midpoint(self):
        prior = _make_prior()
        theta = prior.transform(np.full(3, 0.5))
        assert theta[0] == pytest.approx(0.75)
        assert theta[1] == pytest.approx(1187008882.5)
        assert theta[2] == pytest.approx(np.pi)

    def test_to_dict_keys(self):
        prior = _make_prior()
        d = prior.to_dict(np.array([0.7, 1187008882.0, 1.5]))
        assert set(d.keys()) == {"mass_ratio", "tc", "ra"}

    def test_to_dict_values(self):
        prior = _make_prior()
        d = prior.to_dict(np.array([0.7, 1187008882.0, 1.5]))
        assert d["mass_ratio"] == pytest.approx(0.7)
        assert d["tc"]         == pytest.approx(1187008882.0)
        assert d["ra"]         == pytest.approx(1.5)

    def test_log_prior_inside(self):
        prior = _make_prior()
        theta = np.array([0.75, 1187008882.5, np.pi])
        assert prior.log_prior(theta) == 0.0

    def test_log_prior_outside(self):
        prior = _make_prior()
        theta = np.array([2.0, 1187008882.5, np.pi])  # mass_ratio out of range
        assert prior.log_prior(theta) == -np.inf

    @pytest.mark.parametrize("u", [
        np.array([0.3, 0.7, 0.1]),
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0]),
    ])
    def test_transform_roundtrip(self, u):
        prior = _make_prior()
        theta = prior.transform(u)
        # Each component must lie within [lower, upper].
        for i, p in enumerate(prior.parameters):
            assert p.lower <= theta[i] <= p.upper


# ---------------------------------------------------------------------------
# SamplerResult
# ---------------------------------------------------------------------------

class TestSamplerResult:

    @pytest.fixture
    def result(self):
        rng = np.random.default_rng(0)
        n, d = 200, 3
        samples = rng.standard_normal((n, d))
        log_weights = -np.log(n) * np.ones(n)  # equal weights
        return SamplerResult(
            samples=samples,
            log_weights=log_weights,
            log_evidence=-5.0,
            log_evidence_err=0.1,
            parameter_names=["a", "b", "c"],
        )

    def test_attributes(self, result):
        assert result.log_evidence == pytest.approx(-5.0)
        assert result.log_evidence_err == pytest.approx(0.1)
        assert result.samples.shape == (200, 3)

    def test_posterior_samples_shape(self, result):
        s = result.posterior_samples(n=50)
        assert s.shape == (50, 3)

    def test_posterior_dict_keys(self, result):
        d = result.posterior_dict(n=100)
        assert set(d.keys()) == {"a", "b", "c"}

    def test_posterior_dict_shape(self, result):
        d = result.posterior_dict(n=100)
        for arr in d.values():
            assert arr.shape == (100,)

    def test_posterior_median_keys(self, result):
        m = result.posterior_median()
        assert set(m.keys()) == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# DynestySampler — unit tests (no likelihood evaluation)
# ---------------------------------------------------------------------------

class TestDynestySampler:

    def test_instantiation(self):
        prior = _make_prior()
        s = DynestySampler(_gaussian_loglike, prior, nlive=50)
        assert s.prior is prior
        assert s.nlive == 50

    def test_loglike_array(self):
        prior = _make_prior()
        s = DynestySampler(_gaussian_loglike, prior)
        # At the midpoint of the prior, the loglike should be finite.
        theta = prior.transform(np.full(3, 0.5))
        ll = s.loglike_array(theta)
        assert np.isfinite(ll)

    def test_swappable_sampler(self):
        # The sampler is generic: passing a different callable should work.
        prior = UniformPrior([Parameter("x", -1.0, 1.0)])
        flat_like = lambda params: 0.0
        s = DynestySampler(flat_like, prior, nlive=10)
        theta = prior.transform(np.array([0.5]))
        assert s.loglike_array(theta) == 0.0


# ---------------------------------------------------------------------------
# Integration test: dynesty recovers a known Gaussian
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestDynestyIntegration:
    """Runs dynesty on a 3-D Gaussian. Marked slow; skipped in the fast suite."""

    @pytest.fixture(scope="class")
    def result(self):
        prior = _make_prior()
        sampler = DynestySampler(
            _gaussian_loglike, prior,
            nlive=200,
        )
        return sampler.run(dlogz=0.5, print_progress=False)

    def test_result_type(self, result):
        assert isinstance(result, SamplerResult)

    def test_log_evidence_finite(self, result):
        assert np.isfinite(result.log_evidence)

    def test_posterior_dict_has_correct_keys(self, result):
        d = result.posterior_dict(n=500)
        assert set(d.keys()) == {"mass_ratio", "tc", "ra"}

    def test_mass_ratio_posterior_near_truth(self, result):
        med = result.posterior_median()
        assert abs(med["mass_ratio"] - 0.75) < 0.05

    def test_tc_posterior_near_truth(self, result):
        med = result.posterior_median()
        assert abs(med["tc"] - 1187008882.4) < 0.05

    def test_ra_posterior_near_truth(self, result):
        med = result.posterior_median()
        assert abs(med["ra"] - 1.95) < 0.3
