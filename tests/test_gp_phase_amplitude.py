"""Tests for heron.models.gp.phase_amplitude — PhaseAmplitudeGPSurrogate."""

import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest

from heron.types import Waveform, WaveformDict
from heron.models.gp.phase_amplitude import (
    strain_to_amplitude_phase,
    PhaseAmplitudeGPSurrogate,
)


def _make_synthetic_training_data(n_per_q=50, mass_ratios=(0.3, 0.6, 1.0)):
    """Create simple synthetic training data (sine waves), same shape as
    tests/test_gp_exact.py's fixture so the two models are directly
    comparable."""
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


class TestStrainToAmplitudePhase:

    def test_recovers_known_amplitude_and_phase(self):
        """Constant amplitude, linear phase ramp at a single mass ratio —
        unwrap should recover the input phase exactly (small enough steps
        that consecutive differences stay under pi)."""
        n = 200
        t = np.linspace(0.0, 1.0, n)
        amplitude = 3.7
        phase = -10.0 * t  # steps of ~0.05 rad, well under the pi Nyquist limit

        y_plus = amplitude * np.cos(phase)
        y_cross = amplitude * np.sin(phase)

        x = torch.tensor(
            np.column_stack([np.full(n, 0.5), t]), dtype=torch.float64
        )
        y_plus_t = torch.tensor(y_plus, dtype=torch.float64)
        y_cross_t = torch.tensor(y_cross, dtype=torch.float64)

        x_out, logA_out, phase_out = strain_to_amplitude_phase(x, y_plus_t, y_cross_t)

        # Rows come back sorted by time (already sorted here, single q group).
        np.testing.assert_allclose(x_out.numpy(), x.numpy())
        np.testing.assert_allclose(
            logA_out.numpy(), np.full(n, np.log(amplitude)), atol=1e-8
        )
        np.testing.assert_allclose(phase_out.numpy(), phase, atol=1e-8)

    def test_groups_by_mass_ratio_independently(self):
        """Two mass ratios with different phase ramps must not leak into
        each other's unwrap."""
        n = 50
        t = np.linspace(0.0, 1.0, n)
        x_list, yp_list, yc_list = [], [], []
        for q, rate in [(0.3, -5.0), (0.8, -20.0)]:
            phase = rate * t
            x_list.append(np.column_stack([np.full(n, q), t]))
            yp_list.append(np.cos(phase))
            yc_list.append(np.sin(phase))

        x = torch.tensor(np.vstack(x_list), dtype=torch.float64)
        y_plus = torch.tensor(np.concatenate(yp_list), dtype=torch.float64)
        y_cross = torch.tensor(np.concatenate(yc_list), dtype=torch.float64)

        x_out, logA_out, phase_out = strain_to_amplitude_phase(x, y_plus, y_cross)

        assert x_out.shape[0] == 2 * n
        # log-amplitude should be ~0 everywhere (unit amplitude both groups)
        np.testing.assert_allclose(logA_out.numpy(), np.zeros(2 * n), atol=1e-8)


class TestPhaseAmplitudeGPSurrogate:

    @pytest.fixture(scope="class")
    def trained_model(self):
        train_x, train_y_plus, train_y_cross = _make_synthetic_training_data(
            n_per_q=30, mass_ratios=(0.5, 1.0)
        )
        model = PhaseAmplitudeGPSurrogate(
            train_x=train_x,
            train_y_plus=train_y_plus,
            train_y_cross=train_y_cross,
            warping="chirp",
            nu=2.5,
            device="cpu",
            total_mass=60.0,
            distance=100.0,
            training_iterations=20,
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
        for pol in ("plus", "cross"):
            eigvals = np.linalg.eigvalsh(wf[pol].covariance)
            assert (eigvals >= -1e-6).all(), f"{pol}: negative eigenvalues {eigvals.min()}"

    def test_save_load_roundtrip(self, trained_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            trained_model.save(path)

            loaded = PhaseAmplitudeGPSurrogate.load(path, device="cpu")

            params = {
                "mass_ratio": 0.7,
                "time": {"lower": -0.2, "upper": 0.01, "number": 50},
            }
            wf_orig = trained_model.predict(params)
            wf_loaded = loaded.predict(params)

            np.testing.assert_allclose(
                wf_orig["plus"].data, wf_loaded["plus"].data, atol=1e-5
            )
            np.testing.assert_allclose(
                wf_orig["cross"].data, wf_loaded["cross"].data, atol=1e-5
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


class TestMeanReferencedUnwrap:
    """strain_to_amplitude_phase with a phase_reference — recovers targets
    the plain sparse-grid unwrap corrupts."""

    def _undersampled_chirp(self, n=80):
        """A quadratic phase whose per-sample advance grows past pi (plain
        unwrap silently deletes cycles), sampled at a single mass ratio."""
        t = np.linspace(0.0, 1.0, n)
        phase = -400.0 * t**2  # final step ~10 rad >> pi
        y_plus = np.cos(phase)
        y_cross = np.sin(phase)
        x = torch.tensor(np.column_stack([np.full(n, 0.5), t]), dtype=torch.float64)
        return (
            x,
            torch.tensor(y_plus, dtype=torch.float64),
            torch.tensor(y_cross, dtype=torch.float64),
            phase,
        )

    def test_plain_unwrap_corrupts_undersampled_phase(self):
        x, yp, yc, true_phase = self._undersampled_chirp()
        _, _, phase_out = strain_to_amplitude_phase(x, yp, yc)
        assert np.abs(phase_out.numpy() - true_phase).max() > 2 * np.pi

    def test_referenced_unwrap_recovers_undersampled_phase(self):
        x, yp, yc, true_phase = self._undersampled_chirp()
        # Reference slightly wrong (constant 0.3 rad offset), as a real
        # approximant mean would be — recovery must still be exact.
        ref = true_phase + 0.3
        _, _, phase_out = strain_to_amplitude_phase(
            x, yp, yc, phase_reference=ref
        )
        np.testing.assert_allclose(phase_out.numpy(), true_phase, atol=1e-8)

    def test_branch_canonicalised_against_reference(self):
        """A reference offset by ~2*pi*k + eps must not leave the target on
        a far 2*pi branch (the median-canonicalisation step)."""
        x, yp, yc, true_phase = self._undersampled_chirp()
        ref = true_phase + 3 * 2 * np.pi + 0.3
        _, _, phase_out = strain_to_amplitude_phase(
            x, yp, yc, phase_reference=ref
        )
        # Recovered phase equals truth up to a 2*pi multiple close to the
        # reference's branch; the residual to the REFERENCE must be small.
        resid = phase_out.numpy() - ref
        assert np.abs(resid - np.median(resid)).max() < 1.0
        assert np.abs(np.median(resid)) < np.pi


class TestMeanFunctionSaveLoad:
    """Mean functions must survive checkpoint round-trips (format_version 2);
    with format_version 1 they were silently dropped on load."""

    def test_newtonian_means_roundtrip(self):
        from heron.models.gp.mean import (
            NewtonianInspiralAmplitudeMean,
            NewtonianInspiralPhaseMean,
        )
        from heron.models.warping import get_warping

        train_x, train_y_plus, train_y_cross = _make_synthetic_training_data(
            n_per_q=30, mass_ratios=(0.5, 1.0)
        )
        warping = get_warping("chirp")
        model = PhaseAmplitudeGPSurrogate(
            train_x=train_x,
            train_y_plus=train_y_plus,
            train_y_cross=train_y_cross,
            warping=warping,
            device="cpu",
            mean_module_amplitude=NewtonianInspiralAmplitudeMean(
                total_mass=60.0, distance=100.0, warping=warping
            ),
            mean_module_phase=NewtonianInspiralPhaseMean(
                total_mass=60.0, distance=100.0, warping=warping
            ),
            # Serialization test, not an optimization test: skip training
            # (the PN phase mean on this synthetic sine data is numerically
            # hostile to L-BFGS and can NotPSD, which is irrelevant here).
            training_iterations=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            model.save(path)
            loaded = PhaseAmplitudeGPSurrogate.load(path)

        assert type(loaded.models["log_amplitude"].mean_module).__name__ == (
            "NewtonianInspiralAmplitudeMean"
        )
        assert type(loaded.models["phase"].mean_module).__name__ == (
            "NewtonianInspiralPhaseMean"
        )

        # Predictions must match the in-memory model.
        params = {"mass_ratio": 0.7, "time": {"lower": -0.4, "upper": 0.0, "number": 40}}
        ref = model.predict(params)["plus"].data
        got = loaded.predict(params)["plus"].data
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-30)

    def test_zero_mean_roundtrip_unaffected(self):
        train_x, train_y_plus, train_y_cross = _make_synthetic_training_data(
            n_per_q=30, mass_ratios=(0.5, 1.0)
        )
        model = PhaseAmplitudeGPSurrogate(
            train_x=train_x,
            train_y_plus=train_y_plus,
            train_y_cross=train_y_cross,
            device="cpu",
            training_iterations=5,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            model.save(path)
            loaded = PhaseAmplitudeGPSurrogate.load(path)
        assert type(loaded.models["phase"].mean_module).__name__ == "ZeroMean"
