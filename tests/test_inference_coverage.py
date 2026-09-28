"""Tests for heron.inference.coverage — credible levels and PP-plots."""
import numpy as np
import pytest

from heron.inference.coverage import (
    credible_level_1d,
    credible_levels_from_grid,
    ks_uniform_pvalue,
    pp_plot,
)


class TestCredibleLevel1D:

    def test_gaussian_mean_is_half(self):
        grid = np.linspace(-6, 6, 601)
        log_density = -0.5 * grid**2
        assert credible_level_1d(log_density, grid, 0.0) == pytest.approx(0.5, abs=1e-3)

    def test_one_sigma(self):
        grid = np.linspace(-6, 6, 601)
        log_density = -0.5 * grid**2
        # CDF at +1σ of a standard normal ≈ 0.8413.
        assert credible_level_1d(log_density, grid, 1.0) == pytest.approx(0.8413, abs=5e-3)

    def test_monotone_in_truth(self):
        grid = np.linspace(-6, 6, 601)
        log_density = -0.5 * grid**2
        levels = [credible_level_1d(log_density, grid, t) for t in (-2, -1, 0, 1, 2)]
        assert np.all(np.diff(levels) > 0)


class TestCredibleLevelsFromGrid:

    def test_2d_gaussian_marginals(self):
        gx = np.linspace(-5, 5, 201)
        gy = np.linspace(-5, 5, 201)
        X, Y = np.meshgrid(gx, gy, indexing="ij")
        logL = -0.5 * (X**2 + (Y - 1.0) ** 2)
        cx, cy = credible_levels_from_grid(logL, [gx, gy], [0.0, 1.0])
        assert cx == pytest.approx(0.5, abs=1e-2)
        assert cy == pytest.approx(0.5, abs=1e-2)


class TestKSUniform:

    def test_uniform_high_p(self):
        # Deterministic uniform quantiles (seed-independent) => KS p ≈ 1.
        assert ks_uniform_pvalue(np.linspace(0.001, 0.999, 500)) > 0.5

    def test_clustered_low_p(self):
        rng = np.random.default_rng(0)
        clustered = rng.random(500) * 0.2  # all in [0, 0.2]
        assert ks_uniform_pvalue(clustered) < 1e-3


class TestPPPlot:

    def test_returns_pvalues(self):
        pytest.importorskip("matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        rng = np.random.default_rng(0)
        cls = {"q": rng.random(200), "tc": rng.random(200)}
        fig, ax, pvalues = pp_plot(cls, title="test")
        assert set(pvalues) == {"q", "tc"}
        assert all(0.0 <= p <= 1.0 for p in pvalues.values())
        import matplotlib.pyplot as plt
        plt.close(fig)
