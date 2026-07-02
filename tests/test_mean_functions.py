"""
Tests for custom mean functions.

Run with: pytest tests/test_mean_functions.py -v
"""

import pytest
import torch
import numpy as np

# Try importing the mean function
try:
    from heron.models.mean_functions import (
        IMRPhenomDMeanFunction,
        ZeroMeanWithApproximant,
        RIPPLE_AVAILABLE
    )
    HERON_MEAN_FUNCTIONS_AVAILABLE = True
except ImportError:
    HERON_MEAN_FUNCTIONS_AVAILABLE = False
    RIPPLE_AVAILABLE = False


@pytest.mark.skipif(
    not HERON_MEAN_FUNCTIONS_AVAILABLE,
    reason="Mean functions module not available"
)
class TestMeanFunctions:
    """Test suite for custom mean functions."""

    def test_import(self):
        """Test that mean functions can be imported."""
        from heron.models.mean_functions import IMRPhenomDMeanFunction
        assert IMRPhenomDMeanFunction is not None

    @pytest.mark.skipif(not RIPPLE_AVAILABLE, reason="Ripple not installed")
    def test_imrphenomd_initialization(self):
        """Test IMRPhenomD mean function initialization."""
        mean_fn = IMRPhenomDMeanFunction(
            total_mass=20.0,
            distance=100.0,
            delta_t=1.0/4096,
            f_lower=20.0,
            f_ref=20.0,
            device=torch.device('cpu')
        )

        assert mean_fn.total_mass == 20.0
        assert mean_fn.distance == 100.0
        assert mean_fn.delta_t == 1.0/4096
        assert mean_fn.f_lower == 20.0
        assert mean_fn.f_ref == 20.0

    @pytest.mark.skipif(not RIPPLE_AVAILABLE, reason="Ripple not installed")
    def test_imrphenomd_forward(self):
        """Test IMRPhenomD mean function forward pass."""
        mean_fn = IMRPhenomDMeanFunction(
            total_mass=20.0,
            distance=100.0,
            device=torch.device('cpu')
        )

        # Create test input: 10 points with mass_ratio=0.8, times from -0.05 to 0.01
        n_points = 10
        mass_ratios = torch.ones(n_points) * 0.8
        times = torch.linspace(-0.05, 0.01, n_points)
        x = torch.stack([mass_ratios, times], dim=1)

        # Evaluate mean function
        output = mean_fn(x, polarization='plus')

        # Check output shape and type
        assert output.shape == (n_points,)
        assert isinstance(output, torch.Tensor)
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"

    @pytest.mark.skipif(not RIPPLE_AVAILABLE, reason="Ripple not installed")
    def test_imrphenomd_polarizations(self):
        """Test both polarizations of IMRPhenomD."""
        mean_fn = IMRPhenomDMeanFunction(
            total_mass=20.0,
            distance=100.0,
            device=torch.device('cpu')
        )

        # Create test input
        x = torch.tensor([[0.8, -0.02], [0.8, 0.0], [0.8, 0.01]])

        # Evaluate both polarizations
        h_plus = mean_fn(x, polarization='plus')
        h_cross = mean_fn(x, polarization='cross')

        # Check both are valid
        assert h_plus.shape == (3,)
        assert h_cross.shape == (3,)
        assert not torch.isnan(h_plus).any()
        assert not torch.isnan(h_cross).any()

        # They should be different (not identical)
        assert not torch.allclose(h_plus, h_cross), "Plus and cross polarizations should differ"

    @pytest.mark.skipif(not RIPPLE_AVAILABLE, reason="Ripple not installed")
    def test_imrphenomd_caching(self):
        """Test waveform caching mechanism."""
        mean_fn = IMRPhenomDMeanFunction(
            total_mass=20.0,
            distance=100.0,
            device=torch.device('cpu')
        )

        # First evaluation
        x = torch.tensor([[0.8, -0.02], [0.8, 0.0]])
        output1 = mean_fn(x, polarization='plus')

        # Check cache is populated
        assert len(mean_fn._waveform_cache) > 0

        # Second evaluation with same mass ratio should use cache
        x2 = torch.tensor([[0.8, -0.01], [0.8, 0.005]])
        output2 = mean_fn(x2, polarization='plus')

        assert output2.shape == (2,)

        # Clear cache
        mean_fn.clear_cache()
        assert len(mean_fn._waveform_cache) == 0

    @pytest.mark.skipif(not RIPPLE_AVAILABLE, reason="Ripple not installed")
    def test_imrphenomd_multiple_mass_ratios(self):
        """Test evaluation with multiple mass ratios."""
        mean_fn = IMRPhenomDMeanFunction(
            total_mass=20.0,
            distance=100.0,
            device=torch.device('cpu')
        )

        # Create input with different mass ratios
        x = torch.tensor([
            [0.5, -0.02],
            [0.5, 0.0],
            [0.8, -0.02],
            [0.8, 0.0],
            [1.0, -0.02],
            [1.0, 0.0],
        ])

        output = mean_fn(x, polarization='plus')

        # Check output
        assert output.shape == (6,)
        assert not torch.isnan(output).any()

        # Should have cached multiple mass ratios
        assert len(mean_fn._waveform_cache) == 3  # 3 unique mass ratios

    def test_zero_mean_wrapper(self):
        """Test ZeroMeanWithApproximant wrapper."""
        # Test with no approximant (should behave like zero mean)
        zero_mean = ZeroMeanWithApproximant(approximant_mean=None)
        x = torch.randn(10, 2)
        output = zero_mean(x)

        assert output.shape == (10,)
        assert torch.allclose(output, torch.zeros(10))

    @pytest.mark.skipif(not RIPPLE_AVAILABLE, reason="Ripple not installed")
    def test_zero_mean_wrapper_with_approximant(self):
        """Test ZeroMeanWithApproximant with actual approximant."""
        mean_fn = IMRPhenomDMeanFunction(
            total_mass=20.0,
            distance=100.0,
            device=torch.device('cpu')
        )

        wrapper = ZeroMeanWithApproximant(approximant_mean=mean_fn)
        assert wrapper.use_approximant is True

        # Should produce non-zero output
        x = torch.tensor([[0.8, -0.02], [0.8, 0.0]])
        output = wrapper(x)

        assert output.shape == (2,)
        assert not torch.all(output == 0), "Should produce non-zero values with approximant"

    @pytest.mark.skipif(not RIPPLE_AVAILABLE, reason="Ripple not installed")
    def test_imrphenomd_with_different_masses(self):
        """Test IMRPhenomD with different total masses."""
        for total_mass in [10.0, 20.0, 50.0]:
            mean_fn = IMRPhenomDMeanFunction(
                total_mass=total_mass,
                distance=100.0,
                device=torch.device('cpu')
            )

            x = torch.tensor([[0.8, -0.02], [0.8, 0.0]])
            output = mean_fn(x, polarization='plus')

            assert output.shape == (2,)
            assert not torch.isnan(output).any()

    @pytest.mark.skipif(not RIPPLE_AVAILABLE, reason="Ripple not installed")
    def test_imrphenomd_with_different_distances(self):
        """Test IMRPhenomD with different distances (should scale amplitude)."""
        x = torch.tensor([[0.8, -0.02], [0.8, 0.0]])

        # Close distance
        mean_fn_close = IMRPhenomDMeanFunction(
            total_mass=20.0,
            distance=50.0,
            device=torch.device('cpu')
        )
        output_close = mean_fn_close(x, polarization='plus')

        # Far distance
        mean_fn_far = IMRPhenomDMeanFunction(
            total_mass=20.0,
            distance=200.0,
            device=torch.device('cpu')
        )
        output_far = mean_fn_far(x, polarization='plus')

        # Closer should have larger amplitude
        assert torch.abs(output_close).max() > torch.abs(output_far).max()


@pytest.mark.skipif(not RIPPLE_AVAILABLE, reason="Ripple not installed")
class TestIMRPhenomDIntegration:
    """Integration tests with GPyTorch models."""

    def test_with_gp_model(self):
        """Test mean function integration with GP model."""
        import gpytorch
        from heron.models.gpytorch import ExactGPModelKeOpsWithMean

        # Create synthetic training data
        n_train = 20
        train_x = torch.stack([
            torch.ones(n_train) * 0.8,
            torch.linspace(-0.05, 0.01, n_train)
        ], dim=1)
        train_y = torch.sin(train_x[:, 1] * 100)  # Synthetic data

        # Create mean function
        mean_fn = IMRPhenomDMeanFunction(
            total_mass=20.0,
            distance=100.0,
            device=torch.device('cpu')
        )

        # Create GP model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModelKeOpsWithMean(
            train_x=train_x,
            train_y=train_y,
            likelihood=likelihood,
            mean_function=mean_fn,
            polarization='plus'
        )

        # Test forward pass
        model.train()
        output = model(train_x)

        assert isinstance(output, gpytorch.distributions.MultivariateNormal)
        assert output.mean.shape == (n_train,)

    def test_training_with_mean_function(self):
        """Test training a GP model with mean function."""
        import gpytorch
        from heron.models.gpytorch import ExactGPModelKeOpsWithMean

        # Create synthetic data
        n_train = 20
        train_x = torch.stack([
            torch.ones(n_train) * 0.8,
            torch.linspace(-0.05, 0.01, n_train)
        ], dim=1)
        train_y = torch.sin(train_x[:, 1] * 100)

        # Create model
        mean_fn = IMRPhenomDMeanFunction(
            total_mass=20.0,
            distance=100.0,
            device=torch.device('cpu')
        )
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModelKeOpsWithMean(
            train_x=train_x,
            train_y=train_y,
            likelihood=likelihood,
            mean_function=mean_fn,
            polarization='plus'
        )

        # Train for a few iterations
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        initial_loss = None
        final_loss = None

        for i in range(10):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)

            if i == 0:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

            if i == 9:
                final_loss = loss.item()

        # Loss should decrease
        assert final_loss < initial_loss, "Training should reduce loss"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
