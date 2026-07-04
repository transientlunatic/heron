"""Tests for heron.models.gp.sparse — SparseGPSurrogate."""

import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest

from heron.types import WaveformDict
from heron.models.gp.sparse import SparseGPSurrogate


def _make_synthetic_training_data(n_per_q=50, mass_ratios=(0.3, 0.6, 1.0)):
    """Create simple synthetic training data (sine waves)."""
    all_x = []
    all_plus = []
    all_cross = []

    for q in mass_ratios:
        times = np.linspace(-0.5, 0.02, n_per_q)
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


class TestSparseGPSurrogate:

    @pytest.fixture(scope="class")
    def trained_model(self):
        """Train a small SVGP once for the test class."""
        train_x, train_y_plus, train_y_cross = _make_synthetic_training_data(
            n_per_q=30, mass_ratios=(0.5, 1.0)
        )
        model = SparseGPSurrogate(
            train_x=train_x,
            train_y_plus=train_y_plus,
            train_y_cross=train_y_cross,
            n_inducing=15,
            warping="chirp",
            nu=2.5,
            output_scale=1.0,  # synthetic data doesn't need scaling
            device="cpu",
            total_mass=60.0,
            distance=100.0,
            training_iterations=30,  # just enough to verify it runs
            ls_min_q=0.1,
            ls_min_time=0.05,
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

    def test_covariance_nonzero_at_realistic_output_scale(self):
        """covariance must not silently underflow to zero at output_scale=1e27.

        Regression test: dividing a float32 covariance by output_scale**2
        (~1e54) underflows to exactly 0 (float32 min ~1.2e-38) unless cast
        to float64 first. output_scale=1.0 (used by the other fixtures)
        can't catch this.
        """
        train_x, train_y_plus, train_y_cross = _make_synthetic_training_data(
            n_per_q=30, mass_ratios=(0.5, 1.0)
        )
        model = SparseGPSurrogate(
            train_x=train_x,
            train_y_plus=train_y_plus * 1e-21,  # realistic strain magnitude
            train_y_cross=train_y_cross * 1e-21,
            n_inducing=15,
            output_scale=1e27,
            training_iterations=10,
            ls_min_q=0.1,
            ls_min_time=0.05,
        )
        params = {
            "mass_ratio": 0.7,
            "time": {"lower": -0.3, "upper": 0.02, "number": 50},
        }
        wf = model.predict(params)
        assert np.diagonal(wf["plus"].covariance).max() > 0

    def test_lengthscale_floor_respected(self, trained_model):
        """Learned lengthscales must not collapse below the configured floors."""
        m = trained_model.models["plus"]
        ls_q = float(m.covar_module.base_kernel.kernels[0].lengthscale[0, 0])
        ls_t = float(m.covar_module.base_kernel.kernels[1].lengthscale[0, 0])
        assert ls_q >= 0.1 - 1e-6
        assert ls_t >= 0.05 - 1e-6

    def test_save_load_roundtrip(self, trained_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            trained_model.save(path)

            loaded = SparseGPSurrogate.load(path, device="cpu")

            params = {
                "mass_ratio": 0.7,
                "time": {"lower": -0.2, "upper": 0.01, "number": 50},
            }
            wf_orig = trained_model.predict(params)
            wf_loaded = loaded.predict(params)

            np.testing.assert_allclose(
                wf_orig["plus"].data, wf_loaded["plus"].data, atol=1e-5
            )
            # Lengthscale floors must also survive the round-trip.
            assert loaded.ls_min_q == trained_model.ls_min_q
            assert loaded.ls_min_time == trained_model.ls_min_time

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

    def test_predict_uses_latent_not_predictive_covariance(self, trained_model):
        """predict() must not inflate K with the learned observation noise."""
        params = {
            "mass_ratio": 0.7,
            "time": {"lower": -0.2, "upper": 0.01, "number": 40},
        }
        points_warped = None
        model = trained_model.models["plus"]
        likelihood = trained_model.likelihoods["plus"]

        wf = trained_model.predict(params)

        times = torch.linspace(-0.2, 0.01, 40, dtype=torch.float32)
        points = torch.column_stack([
            torch.full((40,), 0.7, dtype=torch.float32), times,
        ])
        points_warped = points.clone()
        points_warped[:, -1] = trained_model.warping.warp(points_warped[:, -1])

        with torch.no_grad():
            latent = model(points_warped)
            predictive = likelihood(model(points_warped))

        latent_diag = latent.variance.numpy()
        predictive_diag = predictive.variance.numpy()

        # If predict() used the predictive distribution, its variance would
        # match predictive_diag (which is strictly larger due to noise).
        # It should instead match the latent variance.
        returned_diag = np.diagonal(wf["plus"].covariance) * trained_model.output_scale**2
        assert np.allclose(returned_diag, latent_diag, rtol=1e-4)
        assert not np.allclose(returned_diag, predictive_diag, rtol=1e-4)
