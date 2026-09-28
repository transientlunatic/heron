"""Tests for the additive long-lengthscale mass-ratio 'floor' kernel.

Covers heron.models.gp.kernels.build_additive_floor_kernel directly, and its
integration into ExactGPSurrogate / PhaseAmplitudeGPSurrogate via
q_floor_kernel=True. See the kernel's docstring and CLAUDE.md's "Log-det
bias" notes for the motivation: a plain product-Matern kernel's posterior
variance collapses to ~0 at training nodes, oscillating with the training
grid's period in a way that biases the GW likelihood's log-det term toward
training nodes. This is a complementary, kernel-level mechanism to the
runtime `k_smoothing_offsets` patch in heron/gw_likelihood.py.
"""

import tempfile
from pathlib import Path

import numpy as np
import torch
import gpytorch
import pytest

from heron.models.gp.exact import ExactGPSurrogate
from heron.models.gp.phase_amplitude import PhaseAmplitudeGPSurrogate
from heron.models.gp.kernels import build_additive_floor_kernel


class TestBuildAdditiveFloorKernel:

    @pytest.fixture
    def kernel(self):
        return build_additive_floor_kernel(
            nu=2.5,
            active_dims=[0],
            ls_min=0.01,
            init_ls=0.05,
            floor_lengthscale=3.0,
            floor_outputscale_min=0.05,
            floor_outputscale_init=0.1,
        )

    def test_is_additive_of_two_kernels(self, kernel):
        assert isinstance(kernel, gpytorch.kernels.AdditiveKernel)
        assert len(kernel.kernels) == 2

    def test_covariance_symmetric_and_psd(self, kernel):
        x = torch.linspace(0.1, 0.9, 20).unsqueeze(-1)
        K = kernel(x).evaluate()
        assert torch.allclose(K, K.T, atol=1e-6)
        eigvals = torch.linalg.eigvalsh(K + 1e-6 * torch.eye(20))
        assert (eigvals >= -1e-6).all()

    def test_long_lengthscale_is_fixed_not_learnable(self, kernel):
        long_kernel = kernel.kernels[1].base_kernel
        assert long_kernel.raw_lengthscale.requires_grad is False
        assert float(long_kernel.lengthscale) == pytest.approx(3.0)

    def test_short_lengthscale_is_learnable(self, kernel):
        short_kernel = kernel.kernels[0]
        assert short_kernel.raw_lengthscale.requires_grad is True
        assert float(short_kernel.lengthscale) == pytest.approx(0.05, rel=0.05)

    def test_floor_outputscale_respects_min_bound_under_optimisation(self, kernel):
        # Regression test for the same "collapse toward the floor" pathology
        # documented for NonstationaryMaternKernel's width: drive the
        # optimiser hard toward a near-diagonal (locally-independent)
        # solution, which favours shrinking the long component's amplitude
        # toward zero. The GreaterThan constraint must hold regardless.
        x = torch.linspace(0.1, 0.9, 20).unsqueeze(-1)
        opt = torch.optim.Adam(kernel.parameters(), lr=0.5)
        for _ in range(100):
            opt.zero_grad()
            K = kernel(x).evaluate()
            loss = torch.linalg.norm(K - torch.eye(20))
            loss.backward()
            opt.step()
        long_scaled = kernel.kernels[1]
        assert float(long_scaled.outputscale) >= 0.05 - 1e-6

    def test_gradients_flow_to_learnable_parameters_only(self, kernel):
        x = torch.linspace(0.1, 0.9, 10).unsqueeze(-1)
        loss = kernel(x).evaluate().sum()
        loss.backward()
        short = kernel.kernels[0]
        long_scaled = kernel.kernels[1]
        assert short.raw_lengthscale.grad is not None
        assert torch.isfinite(short.raw_lengthscale.grad).all()
        assert long_scaled.raw_outputscale.grad is not None
        assert torch.isfinite(long_scaled.raw_outputscale.grad).all()
        assert long_scaled.base_kernel.raw_lengthscale.grad is None


def _make_synthetic_training_data_dense_q(n_per_q=15, mass_ratios=(0.3, 0.4, 0.5, 0.6, 0.7)):
    """Denser, evenly-spaced mass-ratio grid (spacing 0.1) than the other
    GP test fixtures — needed so a training-grid-periodic variance
    oscillation actually exists to test against."""
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


class TestExactGPSurrogateQFloorKernel:

    @pytest.fixture(scope="class")
    def trained_model(self):
        train_x, train_y_plus, train_y_cross = _make_synthetic_training_data_dense_q()
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
            ls_min_q=0.1,
            noise_floor_rel=1e-3,
            q_floor_kernel=True,
            q_floor_outputscale_min=0.05,
            q_floor_outputscale_init=0.1,
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
        assert (eigvals >= -1e-6).all(), f"Negative eigenvalues: {eigvals.min()}"

    def test_mass_ratio_kernel_is_additive(self, trained_model):
        # dim 0 (mass_ratio) should be the additive floor kernel; dim 1
        # (time) stays a plain MaternKernel since q_floor_kernel only
        # applies to non-time dims.
        product_kernel = trained_model.models["plus"].covar_module.base_kernel
        q_kernel = product_kernel.kernels[0]
        assert isinstance(q_kernel, gpytorch.kernels.AdditiveKernel)

    def test_save_load_roundtrip(self, trained_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            trained_model.save(path)
            loaded = ExactGPSurrogate.load(path, device="cpu")

            assert loaded.q_floor_kernel is True

            params = {"mass_ratio": 0.55, "time": {"lower": -0.25, "upper": 0.0, "number": 30}}
            wf_orig = trained_model.predict(params)
            wf_loaded = loaded.predict(params)
            np.testing.assert_allclose(
                wf_orig["plus"].data, wf_loaded["plus"].data, atol=1e-5
            )

    def test_old_checkpoints_default_to_no_q_floor_kernel(self):
        train_x, train_y_plus, train_y_cross = _make_synthetic_training_data_dense_q(
            n_per_q=8
        )
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
            del checkpoint["q_floor_kernel"]
            torch.save(checkpoint, path)

            loaded = ExactGPSurrogate.load(path, device="cpu")
            assert loaded.q_floor_kernel is False

    def test_does_not_flatten_variance_oscillation_relative_to_plain_kernel(self):
        """Documents a real negative result, not a code contract to defend.

        The original hypothesis (see kernels.py module docstring) was that
        an additive long-lengthscale component would flatten the training-
        grid-periodic collapse of posterior variance at training nodes
        (the mechanism behind the log-det bias -- see CLAUDE.md and
        heron/gw_likelihood.py's k_smoothing_offsets). It doesn't: the
        posterior variance at a training node is, to leading order,
        Var_post(x_i) ~ sigma^2 regardless of kernel shape (conditioning on
        near-noiseless data collapses the posterior to a point mass no
        matter what prior kernel produced it), and any long-lengthscale
        component stable enough not to itself collapse toward `ls_min` is
        also long enough to correlate ~0.98-1.0 over just half a training-
        grid spacing -- so it barely changes the antinode value either.

        This test is isolated from MLL training convergence (which on easy
        synthetic data can collapse outputscale near zero and make K
        unmeasurably small everywhere, masking any structural difference)
        by building two untrained `_ExactGPModel`s directly with identical
        fixed noise/outputscale, differing only in plain-Matern vs.
        additive-floor-kernel for the mass-ratio dimension. Asserts the
        ratio stays *comparable* (within 10%) rather than improving --
        a regression guard against silently reintroducing the original,
        disproven "flattens the oscillation" claim."""
        from heron.models.gp.exact import _ExactGPModel

        q_train = [0.3, 0.4, 0.5, 0.6, 0.7]
        train_x = torch.tensor([[q, 0.0] for q in q_train], dtype=torch.float64)
        train_y = torch.zeros(len(q_train), dtype=torch.float64)

        def build(q_floor_kernel):
            model = _ExactGPModel(
                train_x, train_y,
                nu=2.5,
                ls_min_per_dim=[0.1, 0.1],
                noise_floor_rel=1e-3,
                q_floor_kernel=q_floor_kernel,
                q_floor_outputscale_min=0.1,
                q_floor_outputscale_init=0.2,
            ).double()
            # Posterior variance depends only on X, the kernel and the
            # noise -- not on y. Fix noise/outputscale identically across
            # both models so the comparison isolates kernel structure, not
            # data-driven initialisation.
            model.likelihood.noise = 1e-4
            model.covar_module.outputscale = 1.0
            model.eval()
            model.likelihood.eval()
            return model

        plain = build(False)
        floored = build(True)

        def variance_at(model, q):
            x = torch.tensor([[q, 0.0]], dtype=torch.float64)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                return float(model(x).variance.item())

        def node_antinode_ratio(model):
            return variance_at(model, 0.55) / max(variance_at(model, 0.5), 1e-300)

        ratio_plain = node_antinode_ratio(plain)
        ratio_floored = node_antinode_ratio(floored)
        assert ratio_floored == pytest.approx(ratio_plain, rel=0.1), (
            f"expected the floor kernel to leave the node/antinode ratio "
            f"essentially unchanged (verified negative result), but it "
            f"moved by >10%: floored={ratio_floored}, plain={ratio_plain}"
        )


class TestPhaseAmplitudeGPSurrogateQFloorKernel:

    @pytest.fixture(scope="class")
    def trained_model(self):
        train_x, train_y_plus, train_y_cross = _make_synthetic_training_data_dense_q()
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
            ls_min_q=0.1,
            noise_floor_rel=1e-3,
            q_floor_kernel=True,
            q_floor_outputscale_min=0.05,
            q_floor_outputscale_init=0.1,
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

    def test_save_load_roundtrip(self, trained_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            trained_model.save(path)
            loaded = PhaseAmplitudeGPSurrogate.load(path, device="cpu")

            assert loaded.q_floor_kernel is True

            params = {"mass_ratio": 0.55, "time": {"lower": -0.25, "upper": 0.0, "number": 30}}
            wf_orig = trained_model.predict(params)
            wf_loaded = loaded.predict(params)
            np.testing.assert_allclose(
                wf_orig["plus"].data, wf_loaded["plus"].data, atol=1e-5
            )

    def test_old_checkpoints_default_to_no_q_floor_kernel(self):
        train_x, train_y_plus, train_y_cross = _make_synthetic_training_data_dense_q(
            n_per_q=8
        )
        plain_model = PhaseAmplitudeGPSurrogate(
            train_x=train_x,
            train_y_plus=train_y_plus,
            train_y_cross=train_y_cross,
            warping="simple",
            training_iterations=5,
            ls_min_time=0.02,
            ls_min_q=0.1,
            noise_floor_rel=1e-3,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            plain_model.save(path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            del checkpoint["q_floor_kernel"]
            torch.save(checkpoint, path)

            loaded = PhaseAmplitudeGPSurrogate.load(path, device="cpu")
            assert loaded.q_floor_kernel is False
