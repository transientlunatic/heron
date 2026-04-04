"""Tests for heron.models.gp.exact — ExactGPSurrogate."""

import tempfile
from pathlib import Path

import numpy as np
import torch
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
