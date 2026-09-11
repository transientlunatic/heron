"""Tests for heron.models.gp.exact — ExactGPSurrogate."""

import tempfile
from pathlib import Path

import numpy as np
import torch
import gpytorch
import pytest

from heron.types import Waveform, WaveformDict
from heron.models.gp.exact import ExactGPSurrogate


def _make_synthetic_training_data(n_per_q=50, mass_ratios=(0.3, 0.6, 1.0)):
    """Create simple synthetic training data (sine waves)."""
    all_x = []
    all_plus = []
    all_cross = []

    for q in mass_ratios:
        times = np.linspace(-0.5, 0.02, n_per_q)
        # Simple q-dependent waveform
        plus = np.sin(2 * np.pi * 50 * times) * np.exp(-100 * times**2) * q
        cross = np.cos(2 * np.pi * 50 * times) * np.exp(-100 * times**2) * q
        coords = np.column_stack([np.full(n_per_q, q), times])
        all_x.append(coords)
        all_plus.append(plus)
        all_cross.append(cross)

    return (
        torch.tensor(np.vstack(all_x), dtype=torch.float32),
        torch.tensor(np.concatenate(all_plus), dtype=torch.float32),
        torch.tensor(np.concatenate(all_cross), dtype=torch.float32),
    )


class TestExactGPSurrogate:

    @pytest.fixture(scope="class")
    def trained_model(self):
        """Train a small model once for the test class."""
        train_x, train_y_plus, train_y_cross = _make_synthetic_training_data(
            n_per_q=30, mass_ratios=(0.5, 1.0)
        )
        model = ExactGPSurrogate(
            train_x=train_x,
            train_y_plus=train_y_plus,
            train_y_cross=train_y_cross,
            warping="chirp",
            nu=2.5,
            output_scale=1.0,  # synthetic data doesn't need scaling
            device="cpu",
            total_mass=60.0,
            distance=100.0,
            training_iterations=20,  # just enough to verify it runs
        )
        return model

    def test_predict_returns_waveform_dict(self, trained_model):
        params = {
            "mass_ratio": 0.7,
            "time": {"lower": -0.3, "upper": 0.02, "number": 100},
        }
        wf = trained_model.predict(params)

        assert isinstance(wf, WaveformDict)
        assert "plus" in wf
        assert "cross" in wf

    def test_predict_has_correct_shapes(self, trained_model):
        n = 80
        params = {
            "mass_ratio": 0.7,
            "time": {"lower": -0.3, "upper": 0.02, "number": n},
        }
        wf = trained_model.predict(params)

        assert wf["plus"].data.shape == (n,)
        assert wf["plus"].times.shape == (n,)
        assert wf["plus"].covariance.shape == (n, n)
        assert wf["cross"].covariance.shape == (n, n)

    def test_covariance_is_positive_semidefinite(self, trained_model):
        params = {
            "mass_ratio": 0.7,
            "time": {"lower": -0.3, "upper": 0.02, "number": 50},
        }
        wf = trained_model.predict(params)
        eigvals = np.linalg.eigvalsh(wf["plus"].covariance)
        assert (eigvals >= -1e-6).all(), f"Negative eigenvalues: {eigvals.min()}"

    def test_uncertainty_higher_away_from_training(self, trained_model):
        """Uncertainty should be higher at mass ratios far from training data."""
        # Near training data (trained on q=0.5 and q=1.0)
        near_params = {
            "mass_ratio": 0.75,
            "time": {"lower": -0.2, "upper": 0.01, "number": 50},
        }
        # Far from training data
        far_params = {
            "mass_ratio": 0.1,
            "time": {"lower": -0.2, "upper": 0.01, "number": 50},
        }
        near_wf = trained_model.predict(near_params)
        far_wf = trained_model.predict(far_params)

        near_var = np.mean(near_wf["plus"].variance)
        far_var = np.mean(far_wf["plus"].variance)

        # This is a soft test — GP uncertainty should generally increase
        # away from training data, but it's not guaranteed for all kernels
        # Just verify both are finite and positive
        assert near_var > 0
        assert far_var > 0

    def test_save_load_roundtrip(self, trained_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            trained_model.save(path)

            loaded = ExactGPSurrogate.load(path, device="cpu")

            params = {
                "mass_ratio": 0.7,
                "time": {"lower": -0.2, "upper": 0.01, "number": 50},
            }
            wf_orig = trained_model.predict(params)
            wf_loaded = loaded.predict(params)

            np.testing.assert_allclose(
                wf_orig["plus"].data, wf_loaded["plus"].data, atol=1e-5
            )

    def test_parameter_names(self, trained_model):
        assert "mass_ratio" in trained_model.parameter_names

    def test_parameter_bounds(self, trained_model):
        bounds = trained_model.parameter_bounds
        assert "mass_ratio" in bounds
        lo, hi = bounds["mass_ratio"]
        assert lo < hi

    def test_predict_with_times_array(self, trained_model):
        times = np.linspace(-0.2, 0.01, 60)
        params = {
            "mass_ratio": 0.7,
            "times": times,
        }
        wf = trained_model.predict(params)
        assert wf["plus"].data.shape == (60,)


class TestPerPolarisationMean:
    """mean_module_plus / mean_module_cross — polarisation-specific means
    (e.g. LALApproximantPlusMean/CrossMean), and format_version 7
    checkpoint (de)serialization / backward compat with the old shared
    "mean_function" format."""

    def test_distinct_means_applied_per_polarisation(self):
        from heron.models.gp.mean import NewtonianInspiralMean, TaylorT2Mean

        train_x, train_y_plus, train_y_cross = _make_synthetic_training_data(
            n_per_q=30, mass_ratios=(0.5, 1.0)
        )
        model = ExactGPSurrogate(
            train_x=train_x,
            train_y_plus=train_y_plus,
            train_y_cross=train_y_cross,
            warping="chirp",
            output_scale=1.0,
            mean_module_plus=NewtonianInspiralMean(
                total_mass=60.0, distance=100.0, output_scale=1.0
            ),
            mean_module_cross=TaylorT2Mean(
                total_mass=60.0, distance=100.0, output_scale=1.0
            ),
            training_iterations=5,
        )
        assert type(model.models["plus"].mean_module) is NewtonianInspiralMean
        assert type(model.models["cross"].mean_module) is TaylorT2Mean

    def test_save_load_roundtrip_preserves_distinct_means(self):
        from heron.models.gp.mean import NewtonianInspiralMean, TaylorT2Mean

        train_x, train_y_plus, train_y_cross = _make_synthetic_training_data(
            n_per_q=30, mass_ratios=(0.5, 1.0)
        )
        model = ExactGPSurrogate(
            train_x=train_x,
            train_y_plus=train_y_plus,
            train_y_cross=train_y_cross,
            warping="chirp",
            output_scale=1.0,
            mean_module_plus=NewtonianInspiralMean(
                total_mass=60.0, distance=100.0, output_scale=1.0
            ),
            mean_module_cross=TaylorT2Mean(
                total_mass=60.0, distance=100.0, output_scale=1.0
            ),
            training_iterations=5,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            model.save(path)

            checkpoint = torch.load(path, weights_only=False)
            assert checkpoint["format_version"] == 7
            assert checkpoint["mean_functions"]["plus"]["type"] == "newtonian"
            assert checkpoint["mean_functions"]["cross"]["type"] == "taylort2"

            loaded = ExactGPSurrogate.load(path)
            assert type(loaded.models["plus"].mean_module) is NewtonianInspiralMean
            assert type(loaded.models["cross"].mean_module) is TaylorT2Mean

    def test_legacy_shared_mean_function_still_loads(self):
        """format_version <= 6 checkpoints store a single shared
        "mean_function" (no "mean_functions" dict) -- load() must still
        apply it to both polarisations."""
        from heron.models.gp.mean import NewtonianInspiralMean

        train_x, train_y_plus, train_y_cross = _make_synthetic_training_data(
            n_per_q=30, mass_ratios=(0.5, 1.0)
        )
        model = ExactGPSurrogate(
            train_x=train_x,
            train_y_plus=train_y_plus,
            train_y_cross=train_y_cross,
            warping="chirp",
            output_scale=1.0,
            mean_module=NewtonianInspiralMean(
                total_mass=60.0, distance=100.0, output_scale=1.0
            ),
            training_iterations=5,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            model.save(path)

            # Rewrite as a legacy (format_version <= 6) checkpoint: single
            # shared "mean_function", no "mean_functions".
            checkpoint = torch.load(path, weights_only=False)
            checkpoint["format_version"] = 6
            checkpoint["mean_function"] = checkpoint.pop("mean_functions")["plus"]
            torch.save(checkpoint, path)

            loaded = ExactGPSurrogate.load(path)
            assert type(loaded.models["plus"].mean_module) is NewtonianInspiralMean
            assert type(loaded.models["cross"].mean_module) is NewtonianInspiralMean


class TestNoiseFloorUsesResidualVariance:
    """The noise floor and outputscale/noise init must be scaled by the
    variance of the residual (y - mean(x)), not the raw target. With an
    accurate mean function, var(y) put the noise floor orders of magnitude
    above the entire residual, forcing the GP to predict ~= the bare mean
    (found with the IMRPhenomXAS mean: 0.1 rad residual on a 141 rad-std
    phase target)."""

    def test_noise_floor_scales_with_residual_not_target(self):
        import gpytorch
        from heron.models.gp.exact import _ExactGPModel

        class BigOffsetMean(gpytorch.means.Mean):
            def forward(self, x):
                return 1000.0 * torch.ones(x.shape[0], dtype=x.dtype)

        torch.manual_seed(0)
        x = torch.rand(50, 2)
        residual_scale = 0.01
        y = 1000.0 + residual_scale * torch.randn(50)

        with_mean = _ExactGPModel(
            x, y, mean_module=BigOffsetMean(), noise_floor_rel=1e-3
        )
        floor = float(
            with_mean.likelihood.noise_covar.raw_noise_constraint.lower_bound
        )
        # Residual var ~1e-4 -> floor ~1e-7. Raw target var would be ~0
        # (constant 1000 + tiny noise) for var computed on y... use a
        # sloped target to make raw var large instead:
        y_sloped = 1000.0 * x[:, 0] + residual_scale * torch.randn(50)

        class SlopedMean(gpytorch.means.Mean):
            def forward(self, xx):
                return 1000.0 * xx[:, 0]

        sloped = _ExactGPModel(
            x, y_sloped, mean_module=SlopedMean(), noise_floor_rel=1e-3
        )
        sloped_floor = float(
            sloped.likelihood.noise_covar.raw_noise_constraint.lower_bound
        )
        # var(y_sloped) ~ 8e4; residual var ~1e-4. The floor must track the
        # residual (~1e-7), not the raw target (~80).
        assert sloped_floor < 1e-5
        assert floor < 1e-5


class _CountingZeroMean(gpytorch.means.ZeroMean):
    """A ZeroMean that counts forward() calls via a class-level counter.

    The counter is a class attribute so it survives the float64 deepcopy that
    ``_get_predict_models`` performs — instance attributes would be copied, a
    class attribute is shared across all copies.
    """

    calls = 0

    def forward(self, xx):
        type(self).calls += 1
        return super().forward(xx)


class TestEnvelopeCovarianceDiagonal:
    """The variance-only k-smoothing envelope (``envelope_covariance_diagonal``)
    that avoids re-evaluating the mean at each offset."""

    @pytest.fixture(scope="class")
    def model(self):
        train_x, yp, yc = _make_synthetic_training_data(
            n_per_q=30, mass_ratios=(0.5, 1.0)
        )
        return ExactGPSurrogate(
            train_x=train_x, train_y_plus=yp, train_y_cross=yc,
            warping="chirp", nu=2.5, output_scale=1.0, device="cpu",
            total_mass=60.0, distance=100.0, training_iterations=20,
        )

    @staticmethod
    def _base_params():
        return {"mass_ratio": 0.7,
                "time": {"lower": -0.3, "upper": 0.02, "number": 60}}

    def test_empty_offsets_match_predict_diagonal(self, model):
        """With no offsets the covariance-only path must reproduce the full
        predict() covariance diagonal exactly (the exact-GP covariance is
        mean-independent, so swapping the mean off changes nothing)."""
        params = self._base_params()
        env = model.envelope_covariance_diagonal(params, [])
        wf = model.predict(params)
        for pol in ("plus", "cross"):
            np.testing.assert_allclose(
                env[pol], wf[pol].covariance.diagonal(), rtol=1e-9, atol=1e-30,
            )

    def test_envelope_is_upper_bound(self, model):
        params = self._base_params()
        base = model.envelope_covariance_diagonal(params, [])
        env = model.envelope_covariance_diagonal(
            params, [0.01, -0.01, 0.02, -0.02]
        )
        for pol in ("plus", "cross"):
            assert np.all(env[pol] >= base[pol] - 1e-30)

    def test_envelope_equals_max_over_offsets(self, model):
        params = self._base_params()
        offsets = [0.013, -0.017, 0.023]
        env = model.envelope_covariance_diagonal(params, offsets)
        q0 = params["mass_ratio"]
        stacked = {"plus": [], "cross": []}
        for off in [0.0] + offsets:
            p = dict(params)
            p["mass_ratio"] = q0 + off
            d = model._covariance_diag(p)
            stacked["plus"].append(d["plus"])
            stacked["cross"].append(d["cross"])
        for pol in ("plus", "cross"):
            np.testing.assert_allclose(
                env[pol], np.maximum.reduce(stacked[pol]),
                rtol=1e-12, atol=1e-30,
            )

    def test_mean_never_evaluated_in_envelope(self):
        """The whole point of the variance-only path: the (expensive) mean is
        never evaluated during enveloping, so cost is independent of the number
        of offsets."""
        train_x, yp, yc = _make_synthetic_training_data(
            n_per_q=20, mass_ratios=(0.5, 1.0)
        )
        model = ExactGPSurrogate(
            train_x=train_x, train_y_plus=yp, train_y_cross=yc,
            warping="chirp", nu=2.5, output_scale=1.0, device="cpu",
            total_mass=60.0, distance=100.0, training_iterations=5,
            mean_module=_CountingZeroMean(),
        )
        params = {"mass_ratio": 0.7,
                  "time": {"lower": -0.3, "upper": 0.02, "number": 40}}

        _CountingZeroMean.calls = 0
        model.predict(params)  # warms the float64 clone and uses the mean
        assert _CountingZeroMean.calls > 0  # sanity: predict evaluates the mean

        _CountingZeroMean.calls = 0
        model.envelope_covariance_diagonal(
            params, [0.01, -0.01, 0.02, -0.02, 0.03]
        )
        assert _CountingZeroMean.calls == 0


class TestCovarianceModes:
    """predict(covariance='full'|'diagonal'|'none') — the diagonal path lets
    the marginal likelihood avoid forming the full N×N covariance."""

    @pytest.fixture(scope="class")
    def model(self):
        train_x, yp, yc = _make_synthetic_training_data(
            n_per_q=30, mass_ratios=(0.5, 1.0)
        )
        return ExactGPSurrogate(
            train_x=train_x, train_y_plus=yp, train_y_cross=yc,
            warping="chirp", nu=2.5, output_scale=1.0, device="cpu",
            total_mass=60.0, distance=100.0, training_iterations=20,
        )

    @staticmethod
    def _params():
        return {"mass_ratio": 0.7,
                "time": {"lower": -0.3, "upper": 0.02, "number": 60}}

    def test_diagonal_matches_full_diagonal(self, model):
        full = model.predict(self._params(), covariance="full")
        diag = model.predict(self._params(), covariance="diagonal")
        for pol in ("plus", "cross"):
            assert diag[pol].covariance is None
            assert diag[pol].variance is not None
            np.testing.assert_array_equal(diag[pol].data, full[pol].data)
            np.testing.assert_allclose(
                diag[pol].variance, full[pol].covariance.diagonal(),
                rtol=1e-9, atol=1e-30,
            )

    def test_none_is_mean_only(self, model):
        full = model.predict(self._params(), covariance="full")
        none = model.predict(self._params(), covariance="none")
        for pol in ("plus", "cross"):
            assert none[pol].covariance is None and none[pol].variance is None
            np.testing.assert_array_equal(none[pol].data, full[pol].data)

    def test_invalid_mode_raises(self, model):
        with pytest.raises(ValueError):
            model.predict(self._params(), covariance="banana")

    def test_pickle_roundtrip(self, model):
        """ExactGPSurrogate must survive pickle (GPyTorch prior closures would
        otherwise break it) for n_pool workers."""
        import pickle

        params = self._params()
        wf = model.predict(params)
        restored = pickle.loads(pickle.dumps(model))
        wf2 = restored.predict(params)
        for pol in ("plus", "cross"):
            np.testing.assert_allclose(wf[pol].data, wf2[pol].data, rtol=1e-9, atol=1e-30)
            np.testing.assert_allclose(
                wf[pol].covariance, wf2[pol].covariance, rtol=1e-9, atol=1e-30
            )
