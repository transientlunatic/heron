"""Tests for heron.inference.prior — priors and PriorDict."""
import numpy as np
import pytest
from scipy.integrate import quad

from heron.inference.prior import Uniform, Sine, Cosine, PowerLaw, PriorDict


class TestPriorNormalisation:
    """Each prior's density must integrate to 1 over its support."""

    @pytest.mark.parametrize("prior", [
        Uniform(0.4, 0.95),
        Sine(),
        Sine(0.0, np.pi / 2),
        Cosine(),
        PowerLaw(2.0, 10.0, 1000.0),
        PowerLaw(-1.0, 10.0, 1000.0),   # log-uniform special case
    ])
    def test_integrates_to_one(self, prior):
        integ, _ = quad(lambda v: np.exp(prior.ln_prob(v)), prior.minimum, prior.maximum)
        assert integ == pytest.approx(1.0, rel=1e-4)

    @pytest.mark.parametrize("prior", [
        Uniform(0.4, 0.95), Sine(), Cosine(), PowerLaw(2.0, 10.0, 1000.0),
    ])
    def test_outside_support_is_neg_inf(self, prior):
        assert prior.ln_prob(prior.minimum - 1e-3) == -np.inf
        assert prior.ln_prob(prior.maximum + 1e-3) == -np.inf


class TestRescale:
    """rescale is the inverse CDF: monotone, hits the bounds at u=0,1, and its
    empirical distribution matches ln_prob."""

    @pytest.mark.parametrize("prior", [
        Uniform(0.4, 0.95), Sine(), Cosine(), PowerLaw(2.0, 10.0, 1000.0),
    ])
    def test_bounds(self, prior):
        assert prior.rescale(0.0) == pytest.approx(prior.minimum, abs=1e-9)
        assert prior.rescale(1.0) == pytest.approx(prior.maximum, abs=1e-9)

    @pytest.mark.parametrize("prior", [Sine(), Cosine(), PowerLaw(2.0, 10.0, 1000.0)])
    def test_monotone(self, prior):
        u = np.linspace(0, 1, 50)
        vals = np.array([prior.rescale(ui) for ui in u])
        assert np.all(np.diff(vals) > 0)

    def test_powerlaw_samples_match_density(self):
        prior = PowerLaw(2.0, 10.0, 1000.0)
        rng = np.random.default_rng(0)
        samples = np.array([prior.rescale(u) for u in rng.random(20000)])
        # For p ∝ x², the CDF at the midpoint value should match.
        x = 500.0
        cdf_empirical = np.mean(samples <= x)
        cdf_analytic = (x**3 - 10.0**3) / (1000.0**3 - 10.0**3)
        assert cdf_empirical == pytest.approx(cdf_analytic, abs=0.02)


class TestPriorDict:

    def _prior(self):
        return PriorDict({
            "mass_ratio": Uniform(0.4, 0.95),
            "luminosity_distance": PowerLaw(2.0, 100.0, 2000.0),
            "dec": Cosine(),
            "theta_jn": Sine(),
            "psi": Uniform(0.0, np.pi, periodic=True),
        })

    def test_ndim_and_names(self):
        p = self._prior()
        assert p.ndim == 5
        assert p.parameter_names == [
            "mass_ratio", "luminosity_distance", "dec", "theta_jn", "psi",
        ]

    def test_periodic_parameters(self):
        assert self._prior().periodic_parameters == ["psi"]

    def test_bounds(self):
        b = self._prior().bounds()
        assert b["mass_ratio"] == (0.4, 0.95)
        assert b["dec"] == pytest.approx((-np.pi / 2, np.pi / 2))

    def test_transform_in_support(self):
        p = self._prior()
        theta = p.transform(np.full(p.ndim, 0.5))
        assert p.log_prior(theta) > -np.inf
        for i, pr in enumerate(p.priors.values()):
            assert pr.minimum <= theta[i] <= pr.maximum

    def test_transform_corners(self):
        p = self._prior()
        lo = p.transform(np.zeros(p.ndim))
        hi = p.transform(np.ones(p.ndim))
        for i, pr in enumerate(p.priors.values()):
            assert lo[i] == pytest.approx(pr.minimum, abs=1e-9)
            assert hi[i] == pytest.approx(pr.maximum, abs=1e-9)

    def test_to_dict(self):
        p = self._prior()
        theta = p.transform(np.full(p.ndim, 0.3))
        d = p.to_dict(theta)
        assert set(d) == set(p.parameter_names)

    def test_log_prior_out_of_range(self):
        p = self._prior()
        theta = p.transform(np.full(p.ndim, 0.5))
        theta[0] = 2.0  # mass_ratio out of [0.4, 0.95]
        assert p.log_prior(theta) == -np.inf

    def test_dropin_with_dynesty_sampler(self):
        """PriorDict must satisfy the BaseSampler interface DynestySampler uses."""
        from heron.sampling import DynestySampler
        p = self._prior()
        s = DynestySampler(lambda params: 0.0, p, nlive=10)
        theta = p.transform(np.full(p.ndim, 0.5))
        assert np.isfinite(s.loglike_array(theta))
