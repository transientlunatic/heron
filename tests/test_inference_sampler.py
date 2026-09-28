"""Tests for heron.inference.sampler.NessaiSampler."""
import numpy as np
import pytest

from heron.inference.prior import Uniform, PriorDict
from heron.inference.sampler import NessaiSampler


def _prior():
    return PriorDict({
        "x": Uniform(-5.0, 5.0),
        "y": Uniform(-5.0, 5.0),
        "psi": Uniform(0.0, np.pi, periodic=True),
    })


def _gaussian_loglike(params):
    return -0.5 * ((params["x"] - 1.0) ** 2 / 0.5**2
                   + (params["y"] + 0.5) ** 2 / 0.5**2
                   + (params["psi"] - 1.5) ** 2 / 0.3**2)


class TestConstruction:
    """These need no nessai import (BaseSampler + prior helpers only)."""

    def test_instantiation(self):
        s = NessaiSampler(_gaussian_loglike, _prior())
        assert s.prior.ndim == 3

    def test_bounds_from_priordict(self):
        s = NessaiSampler(_gaussian_loglike, _prior())
        assert s._bounds()["x"] == (-5.0, 5.0)

    def test_periodic_detected(self):
        s = NessaiSampler(_gaussian_loglike, _prior())
        assert s._periodic() == ["psi"]

    def test_bounds_from_uniformprior(self):
        """Falls back to heron.sampling.UniformPrior's Parameter list."""
        from heron.sampling import Parameter, UniformPrior
        up = UniformPrior([Parameter("a", 0.0, 1.0), Parameter("b", -1.0, 2.0)])
        s = NessaiSampler(_gaussian_loglike, up)
        assert s._bounds() == {"a": (0.0, 1.0), "b": (-1.0, 2.0)}
        assert s._periodic() == []


class TestModel:
    """Model construction and callables require nessai."""

    def test_build_model(self):
        pytest.importorskip("nessai")
        s = NessaiSampler(_gaussian_loglike, _prior())
        model = s._build_model()
        assert model.names == ["x", "y", "psi"]
        assert np.array_equal(np.asarray(model.bounds["x"]), [-5.0, 5.0])

    def test_model_log_prior_and_likelihood_vectorise(self):
        pytest.importorskip("nessai")
        s = NessaiSampler(_gaussian_loglike, _prior())
        model = s._build_model()
        # A structured array of two live points.
        x = np.array([(1.0, -0.5, 1.5), (0.0, 0.0, 0.0)],
                     dtype=[("x", float), ("y", float), ("psi", float)])
        lp = model.log_prior(x)
        ll = model.log_likelihood(x)
        assert lp.shape == (2,) and ll.shape == (2,)
        assert np.all(np.isfinite(lp))
        # Point on the gaussian peak has higher likelihood than the origin.
        assert ll[0] > ll[1]


@pytest.mark.slow
class TestNessaiIntegration:
    """A short end-to-end nessai run recovering a 3-D gaussian."""

    def test_recovers_gaussian(self, tmp_path):
        pytest.importorskip("nessai")
        s = NessaiSampler(_gaussian_loglike, _prior())
        result = s.run(output=str(tmp_path / "nessai"), nlive=200, seed=1234)
        med = result.posterior_median()
        assert abs(med["x"] - 1.0) < 0.15
        assert abs(med["y"] + 0.5) < 0.15
        assert np.isfinite(result.log_evidence)
