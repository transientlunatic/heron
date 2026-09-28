"""Tests for the mass-ratio input-warped kernel (q_warping).

Covers heron.models.gp.kernels.WarpedMaternKernel / symmetric_mass_ratio_warp
directly, and their integration into ExactGPSurrogate / PhaseAmplitudeGPSurrogate
via q_warping="eta".
"""

import tempfile
from pathlib import Path

import numpy as np
import torch
import gpytorch
import pytest

from heron.models.gp.exact import ExactGPSurrogate
from heron.models.gp.phase_amplitude import PhaseAmplitudeGPSurrogate
from heron.models.gp.mean import NewtonianInspiralAmplitudeMean
from heron.models.gp.kernels import (
    WarpedMaternKernel,
    symmetric_mass_ratio_warp,
    Q_WARP_FUNCTIONS,
)


class TestSymmetricMassRatioWarp:

    def test_matches_known_values(self):
        # eta(q) = q/(1+q)^2; eta(1) = 0.25 (equal mass, the maximum).
        q = torch.tensor([0.1, 0.5, 1.0])
        eta = symmetric_mass_ratio_warp(q)
        expected = torch.tensor([0.1 / 1.21, 0.5 / 2.25, 0.25])
        torch.testing.assert_close(eta, expected)

    def test_monotonic_on_unit_interval(self):
        q = torch.linspace(0.01, 1.0, 200)
        eta = symmetric_mass_ratio_warp(q)
        assert (torch.diff(eta) > 0).all()

    def test_registered_in_warp_function_table(self):
        assert Q_WARP_FUNCTIONS["eta"] is symmetric_mass_ratio_warp


class TestWarpedMaternKernel:

    @pytest.fixture
    def kernel(self):
        k = WarpedMaternKernel(warp_fn=symmetric_mass_ratio_warp, nu=2.5)
        k.lengthscale = 0.05
        return k

    def test_matches_plain_matern_on_prewarped_input(self, kernel):
        # k(x,x') via WarpedMaternKernel on raw q must equal an ordinary
        # MaternKernel (same lengthscale) evaluated directly on eta(q) --
        # this is the whole point of the construction.
        q = torch.linspace(0.1, 0.97, 20).unsqueeze(-1)
        K_warped = kernel(q).to_dense()

        plain = gpytorch.kernels.MaternKernel(nu=2.5)
        plain.lengthscale = 0.05
        K_plain_on_eta = plain(symmetric_mass_ratio_warp(q)).to_dense()

        torch.testing.assert_close(K_warped, K_plain_on_eta)

    def test_covariance_symmetric_and_psd(self, kernel):
        q = torch.linspace(0.1, 0.97, 25).unsqueeze(-1)
        K = kernel(q).to_dense()
        assert torch.allclose(K, K.T, atol=1e-6)
        eigvals = torch.linalg.eigvalsh(K + 1e-6 * torch.eye(25))
        assert (eigvals >= -1e-6).all()

    def test_diagonal_is_unity(self, kernel):
        q = torch.linspace(0.1, 0.97, 25).unsqueeze(-1)
        K = kernel(q).to_dense()
        np.testing.assert_allclose(torch.diagonal(K).detach().numpy(), 1.0, atol=1e-5)

    def test_gradient_flows_to_lengthscale(self, kernel):
        q = torch.linspace(0.1, 0.97, 10).unsqueeze(-1)
        loss = kernel(q).to_dense().sum()
        loss.backward()
        assert kernel.raw_lengthscale.grad is not None
        assert torch.isfinite(kernel.raw_lengthscale.grad).all()

    def test_effective_lengthscale_longer_near_q_one(self):
        # eta'(q) -> 0 as q -> 1, so the effective lengthscale in raw q
        # units (ls_eta / eta'(q)) should be much larger near q=1 than
        # near q=0 -- i.e. correlation should fall off much more slowly
        # near q=1 than near q=0, for the same raw-q step.
        kernel = WarpedMaternKernel(warp_fn=symmetric_mass_ratio_warp, nu=2.5)
        kernel.lengthscale = 0.02
        step = 0.03
        near_zero = kernel(torch.tensor([[0.10], [0.10 + step]])).to_dense()[0, 1]
        near_one = kernel(torch.tensor([[0.90], [0.90 + step]])).to_dense()[0, 1]
        assert near_one > near_zero


def _make_synthetic_training_data(n_per_q=15, mass_ratios=(0.3, 0.4, 0.5, 0.6, 0.7)):
    all_x, all_plus, all_cross = [], [], []
    for q in mass_ratios:
        times = np.linspace(-0.3, 0.05, n_per_q)
        plus = np.sin(2 * np.pi * 20 * times) * np.exp(-50 * times**2) * q
        cross = np.cos(2 * np.pi * 20 * times) * np.exp(-50 * times**2) * q
        coords = np.column_stack([np.full(n_per_q, q), times])
        all_x.append(coords)
        all_plus.append(plus)
        all_cross.append(cross)
    return (
        torch.tensor(np.vstack(all_x), dtype=torch.float32),
        torch.tensor(np.concatenate(all_plus), dtype=torch.float32),
        torch.tensor(np.concatenate(all_cross), dtype=torch.float32),
    )


class TestExactGPSurrogateQWarping:

    @pytest.fixture(scope="class")
    def trained_model(self):
        train_x, train_y_plus, train_y_cross = _make_synthetic_training_data()
        return ExactGPSurrogate(
            train_x=train_x,
            train_y_plus=train_y_plus,
            train_y_cross=train_y_cross,
            warping="simple",
            nu=2.5,
            output_scale=1.0,
            device="cpu",
            total_mass=60.0,
            distance=100.0,
            training_iterations=20,
            ls_min_time=0.02,
            ls_min_q=0.02,
            noise_floor_rel=1e-3,
            q_warping="eta",
        )

    def test_predict_returns_finite_waveform(self, trained_model):
        params = {"mass_ratio": 0.55, "time": {"lower": -0.25, "upper": 0.0, "number": 30}}
        wf = trained_model.predict(params)
        assert np.isfinite(wf["plus"].data).all()
        assert np.isfinite(wf["plus"].covariance).all()

    def test_covariance_is_positive_semidefinite(self, trained_model):
        params = {"mass_ratio": 0.55, "time": {"lower": -0.25, "upper": 0.0, "number": 30}}
        wf = trained_model.predict(params)
        eigvals = np.linalg.eigvalsh(wf["plus"].covariance)
        assert (eigvals >= -1e-6).all()

    def test_mass_ratio_kernel_is_warped(self, trained_model):
        q_kernel = trained_model.models["plus"].covar_module.base_kernel.kernels[0]
        assert isinstance(q_kernel, WarpedMaternKernel)

    def test_save_load_roundtrip(self, trained_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            trained_model.save(path)
            loaded = ExactGPSurrogate.load(path, device="cpu")

            assert loaded.q_warping == "eta"

            params = {"mass_ratio": 0.55, "time": {"lower": -0.25, "upper": 0.0, "number": 30}}
            wf_orig = trained_model.predict(params)
            wf_loaded = loaded.predict(params)
            np.testing.assert_allclose(
                wf_orig["plus"].data, wf_loaded["plus"].data, atol=1e-5
            )

    def test_old_checkpoints_default_to_no_q_warping(self):
        train_x, train_y_plus, train_y_cross = _make_synthetic_training_data(n_per_q=8)
        plain_model = ExactGPSurrogate(
            train_x=train_x,
            train_y_plus=train_y_plus,
            train_y_cross=train_y_cross,
            warping="simple",
            output_scale=1.0,
            training_iterations=5,
            ls_min_time=0.02,
            ls_min_q=0.1,
            noise_floor_rel=1e-3,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            plain_model.save(path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            del checkpoint["q_warping"]
            torch.save(checkpoint, path)

            loaded = ExactGPSurrogate.load(path, device="cpu")
            assert loaded.q_warping is None

    def test_mean_function_unaffected_by_q_warping(self):
        # The whole design point of WarpedMaternKernel: only the KERNEL's
        # own distance computation sees the warped mass ratio. The mean
        # function must receive the same (raw) x regardless of q_warping.
        train_x, train_y_plus, train_y_cross = _make_synthetic_training_data(n_per_q=8)
        common_kwargs = dict(
            train_x=train_x, train_y_plus=train_y_plus, train_y_cross=train_y_cross,
            warping="simple", output_scale=1.0, training_iterations=0,
            ls_min_time=0.02, ls_min_q=0.1, noise_floor_rel=1e-3,
        )
        mean = NewtonianInspiralAmplitudeMean(total_mass=60.0, distance=100.0)
        model_plain = ExactGPSurrogate(**common_kwargs, mean_module=mean)
        model_warped = ExactGPSurrogate(**common_kwargs, mean_module=mean, q_warping="eta")

        x_probe = torch.tensor([[0.55, -0.1], [0.3, -0.2]], dtype=torch.float32)
        with torch.no_grad():
            mean_plain = model_plain.models["plus"].mean_module(x_probe)
            mean_warped = model_warped.models["plus"].mean_module(x_probe)
        torch.testing.assert_close(mean_plain, mean_warped)


class TestPhaseAmplitudeGPSurrogateQWarping:

    @pytest.fixture(scope="class")
    def trained_model(self):
        train_x, train_y_plus, train_y_cross = _make_synthetic_training_data()
        return PhaseAmplitudeGPSurrogate(
            train_x=train_x,
            train_y_plus=train_y_plus,
            train_y_cross=train_y_cross,
            warping="simple",
            nu=2.5,
            device="cpu",
            total_mass=60.0,
            distance=100.0,
            training_iterations=20,
            ls_min_time=0.02,
            ls_min_q=0.02,
            noise_floor_rel=1e-3,
            q_warping="eta",
        )

    def test_predict_returns_finite_waveform(self, trained_model):
        params = {"mass_ratio": 0.55, "time": {"lower": -0.25, "upper": 0.0, "number": 30}}
        wf = trained_model.predict(params)
        assert np.isfinite(wf["plus"].data).all()
        assert np.isfinite(wf["plus"].covariance).all()

    def test_save_load_roundtrip(self, trained_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            trained_model.save(path)
            loaded = PhaseAmplitudeGPSurrogate.load(path, device="cpu")

            assert loaded.q_warping == "eta"

            params = {"mass_ratio": 0.55, "time": {"lower": -0.25, "upper": 0.0, "number": 30}}
            wf_orig = trained_model.predict(params)
            wf_loaded = loaded.predict(params)
            np.testing.assert_allclose(
                wf_orig["plus"].data, wf_loaded["plus"].data, atol=1e-5
            )
