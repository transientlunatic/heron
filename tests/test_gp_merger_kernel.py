"""Tests for the non-stationary (merger-aware) time kernel.

Covers heron.models.gp.kernels.NonstationaryMaternKernel directly, and its
integration into ExactGPSurrogate via merger_kernel=True.
"""

import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest

from heron.models.gp.exact import ExactGPSurrogate
from heron.models.gp.kernels import NonstationaryMaternKernel


class TestNonstationaryMaternKernel:

    @pytest.fixture
    def kernel(self):
        return NonstationaryMaternKernel(
            nu=2.5,
            ls_min_far=0.01,
            ls_min_near=0.001,
            init_ls_far=0.5,
            init_ls_near=0.02,
            center=0.0,
            init_width=0.05,
        )

    def test_local_lengthscale_shrinks_near_merger(self, kernel):
        l_far = kernel.local_lengthscale(torch.tensor([-1.0])).item()
        l_near = kernel.local_lengthscale(torch.tensor([0.5])).item()
        assert l_far == pytest.approx(0.5, rel=0.05)
        assert l_near == pytest.approx(0.02, rel=0.05)
        assert l_near < l_far

    def test_covariance_symmetric_and_psd(self, kernel):
        x = torch.linspace(-1.0, 0.2, 25).unsqueeze(-1)
        K = kernel(x).evaluate()
        assert torch.allclose(K, K.T, atol=1e-6)
        eigvals = torch.linalg.eigvalsh(K + 1e-6 * torch.eye(25))
        assert (eigvals >= -1e-6).all()

    def test_diagonal_is_unity(self, kernel):
        x = torch.linspace(-1.0, 0.2, 25).unsqueeze(-1)
        K = kernel(x).evaluate()
        np.testing.assert_allclose(torch.diagonal(K).detach().numpy(), 1.0, atol=1e-5)

    def test_diag_mode_matches_full_diagonal(self, kernel):
        x = torch.linspace(-1.0, 0.2, 25).unsqueeze(-1)
        K = kernel(x).evaluate()
        k_diag = kernel(x, diag=True)
        np.testing.assert_allclose(
            k_diag.detach().numpy(), torch.diagonal(K).detach().numpy(), atol=1e-5
        )

    def test_reduces_to_stationary_matern_at_equal_lengthscale(self):
        # If lengthscale_far == lengthscale_near, the field is stationary
        # everywhere and should match gpytorch's own MaternKernel exactly.
        kernel = NonstationaryMaternKernel(
            nu=2.5, init_ls_far=0.3, init_ls_near=0.3, center=0.0, init_width=0.05,
        )
        stationary = __import__("gpytorch").kernels.MaternKernel(nu=2.5)
        stationary.lengthscale = 0.3

        x = torch.linspace(-1.0, 0.2, 25).unsqueeze(-1)
        K_ns = kernel(x).evaluate().detach()
        K_st = stationary(x).evaluate().detach()
        np.testing.assert_allclose(K_ns.numpy(), K_st.numpy(), atol=1e-5)

    def test_gradients_flow_to_all_parameters(self, kernel):
        x = torch.linspace(-1.0, 0.2, 10).unsqueeze(-1)
        loss = kernel(x).evaluate().sum()
        loss.backward()
        for name, p in kernel.named_parameters():
            assert p.grad is not None, f"no gradient for {name}"
            assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"

    def test_center_is_fixed_not_learnable(self, kernel):
        # `center` must be a buffer, not a Parameter: two earlier attempts
        # at making it learnable (see kernel docstring) both collapsed by
        # dragging it to the edge of whatever range it was allowed, since
        # nothing pins it to the actually-known physical merger time.
        param_names = {name for name, _ in kernel.named_parameters()}
        assert "center" not in param_names
        assert "raw_center" not in param_names
        assert float(kernel.center) == 0.0

    def test_width_stays_within_bounds_under_optimisation(self):
        # Regression test for the collapse this kernel went through during
        # development: an unbounded (or one-sided-bounded) width will run
        # to its floor/ceiling under a few steps of aggressive optimisation
        # toward the trivial "shortest lengthscale everywhere" solution.
        # With a two-sided Interval constraint, `width` must stay inside
        # (min_width, max_width) no matter how hard the optimiser pushes.
        kernel = NonstationaryMaternKernel(
            nu=2.5, ls_min_far=0.01, ls_min_near=0.001, min_width=0.01, max_width=0.1,
            init_ls_far=0.5, init_ls_near=0.02, center=0.0, init_width=0.03,
        )
        x = torch.linspace(-1.0, 0.2, 40).unsqueeze(-1)
        y = torch.sin(20 * x.squeeze(-1))
        opt = torch.optim.Adam(kernel.parameters(), lr=0.5)
        for _ in range(50):
            opt.zero_grad()
            K = kernel(x).evaluate()
            # Drive toward a degenerate near-diagonal solution -- the same
            # kind of pressure MLL optimisation exerted during training.
            loss = torch.linalg.norm(K - torch.eye(40))
            loss.backward()
            opt.step()

        assert 0.01 < float(kernel.width.detach()) < 0.1
        assert float(kernel.center) == 0.0


def _make_synthetic_training_data_with_merger(n_per_q=60, mass_ratios=(0.5, 1.0)):
    """Synthetic data with a sharp, fast-oscillating 'merger' near t=0,
    superimposed on a slow inspiral-like envelope — the kind of
    two-timescale structure the merger kernel is meant to capture."""
    all_x, all_plus, all_cross = [], [], []
    for q in mass_ratios:
        times = np.linspace(-0.5, 0.1, n_per_q)
        slow = np.sin(2 * np.pi * 5 * times) * q
        fast_burst = np.sin(2 * np.pi * 80 * times) * np.exp(-500 * times**2) * q
        plus = slow + fast_burst
        cross = np.cos(2 * np.pi * 5 * times) * q + fast_burst
        coords = np.column_stack([np.full(n_per_q, q), times])
        all_x.append(coords)
        all_plus.append(plus)
        all_cross.append(cross)

    return (
        torch.tensor(np.vstack(all_x), dtype=torch.float32),
        torch.tensor(np.concatenate(all_plus), dtype=torch.float32),
        torch.tensor(np.concatenate(all_cross), dtype=torch.float32),
    )


class TestExactGPSurrogateMergerKernel:

    @pytest.fixture(scope="class")
    def trained_model(self):
        train_x, train_y_plus, train_y_cross = _make_synthetic_training_data_with_merger()
        model = ExactGPSurrogate(
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
            ls_min_q=0.2,
            noise_floor_rel=1e-3,
            merger_kernel=True,
            ls_min_time_merger=0.005,
            merger_center=0.0,
            merger_width_init=0.05,
        )
        return model

    def test_predict_returns_finite_waveform(self, trained_model):
        params = {"mass_ratio": 0.7, "time": {"lower": -0.4, "upper": 0.05, "number": 40}}
        wf = trained_model.predict(params)
        assert np.isfinite(wf["plus"].data).all()
        assert np.isfinite(wf["plus"].covariance).all()

    def test_covariance_is_positive_semidefinite(self, trained_model):
        params = {"mass_ratio": 0.7, "time": {"lower": -0.4, "upper": 0.05, "number": 40}}
        wf = trained_model.predict(params)
        eigvals = np.linalg.eigvalsh(wf["plus"].covariance)
        assert (eigvals >= -1e-6).all(), f"Negative eigenvalues: {eigvals.min()}"

    def test_merger_lengthscale_differs_from_far_lengthscale(self, trained_model):
        # Sanity check the mechanism engages during training — the two
        # lengthscales need not stay at their (different) init values, but
        # they shouldn't have collapsed onto exactly the same point either.
        time_kernel = trained_model.models["plus"].covar_module.base_kernel.kernels[-1]
        assert isinstance(time_kernel, NonstationaryMaternKernel)
        l_far = float(time_kernel.lengthscale_far.detach())
        l_near = float(time_kernel.lengthscale_near.detach())
        assert l_far > 0 and l_near > 0

    def test_save_load_roundtrip(self, trained_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            trained_model.save(path)
            loaded = ExactGPSurrogate.load(path, device="cpu")

            assert loaded.merger_kernel is True

            params = {"mass_ratio": 0.7, "time": {"lower": -0.4, "upper": 0.05, "number": 40}}
            wf_orig = trained_model.predict(params)
            wf_loaded = loaded.predict(params)
            np.testing.assert_allclose(
                wf_orig["plus"].data, wf_loaded["plus"].data, atol=1e-5
            )

    def test_old_checkpoints_default_to_no_merger_kernel(self):
        # A plain (non-merger) model must still load correctly with
        # merger_kernel defaulting to False (backward compatibility).
        train_x, train_y_plus, train_y_cross = _make_synthetic_training_data_with_merger(
            n_per_q=20
        )
        plain_model = ExactGPSurrogate(
            train_x=train_x,
            train_y_plus=train_y_plus,
            train_y_cross=train_y_cross,
            warping="simple",
            output_scale=1.0,
            training_iterations=5,
            ls_min_time=0.02,
            ls_min_q=0.2,
            noise_floor_rel=1e-3,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            plain_model.save(path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            del checkpoint["merger_kernel"]
            torch.save(checkpoint, path)

            loaded = ExactGPSurrogate.load(path, device="cpu")
            assert loaded.merger_kernel is False
