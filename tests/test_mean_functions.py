"""Tests for heron.models.gp.mean — PN mean functions."""

import torch
import numpy as np
import pytest

from heron.models.gp.mean import ZeroMean, NewtonianInspiralMean, TaylorT2Mean


class TestZeroMean:

    def test_returns_zeros(self):
        mean = ZeroMean()
        x = torch.randn(20, 2)
        out = mean(x)
        assert out.shape == (20,)
        assert (out == 0).all()


class TestNewtonianInspiralMean:

    def test_output_shape(self):
        mean = NewtonianInspiralMean(total_mass=60.0, distance=100.0)
        x = torch.tensor([
            [0.5, -0.1],
            [0.5, -0.05],
            [0.5, -0.01],
            [0.5, 0.0],
            [0.5, 0.01],
        ])
        out = mean(x)
        assert out.shape == (5,)

    def test_zero_after_merger(self):
        """Post-merger (t > 0) should be zero in the inspiral approximation."""
        mean = NewtonianInspiralMean(total_mass=60.0, distance=100.0, output_scale=1.0)
        x = torch.tensor([[0.5, 0.01], [0.5, 0.1], [0.5, 1.0]])
        out = mean(x)
        assert (out == 0).all()

    def test_nonzero_during_inspiral(self):
        """Pre-merger (t < 0) should produce non-zero values."""
        mean = NewtonianInspiralMean(total_mass=60.0, distance=100.0, output_scale=1.0)
        x = torch.tensor([[0.5, -0.5], [0.5, -0.1], [0.5, -0.01]])
        out = mean(x)
        assert not (out == 0).all()

    def test_amplitude_scales_with_distance(self):
        """Closer distance should give larger amplitude."""
        x = torch.tensor([[0.5, -0.1]])
        near = NewtonianInspiralMean(total_mass=60.0, distance=50.0, output_scale=1.0)
        far = NewtonianInspiralMean(total_mass=60.0, distance=200.0, output_scale=1.0)
        assert abs(float(near(x))) > abs(float(far(x)))

    def test_output_scale_applied(self):
        """Output should scale with output_scale."""
        x = torch.tensor([[0.5, -0.1]])
        m1 = NewtonianInspiralMean(total_mass=60.0, distance=100.0, output_scale=1.0)
        m2 = NewtonianInspiralMean(total_mass=60.0, distance=100.0, output_scale=1e27)
        ratio = float(m2(x)) / float(m1(x))
        assert ratio == pytest.approx(1e27, rel=1e-3)

    def test_with_warping(self):
        """Should unwarp time before evaluating."""
        from heron.models.warping import ChirpTimeWarping

        warping = ChirpTimeWarping(alpha=0.625)
        mean = NewtonianInspiralMean(
            total_mass=60.0, distance=100.0, output_scale=1.0, warping=warping
        )
        # Warped time coordinates
        raw_times = torch.tensor([-0.5, -0.1, -0.01])
        warped_times = warping.warp(raw_times)
        x_warped = torch.stack([torch.full_like(warped_times, 0.5), warped_times], dim=1)
        out = mean(x_warped)
        assert out.shape == (3,)
        assert not (out == 0).all()


class TestTaylorT2Mean:

    def test_output_shape(self):
        mean = TaylorT2Mean(total_mass=60.0, distance=100.0)
        x = torch.tensor([[0.5, -0.1], [0.8, -0.05], [1.0, -0.01]])
        out = mean(x)
        assert out.shape == (3,)

    def test_differs_from_newtonian(self):
        """1PN corrections should make TaylorT2 differ from Newtonian."""
        x = torch.tensor([[0.5, -0.1]])
        newt = NewtonianInspiralMean(total_mass=60.0, distance=100.0, output_scale=1.0)
        t2 = TaylorT2Mean(total_mass=60.0, distance=100.0, output_scale=1.0)
        # They should differ (1PN phase correction)
        assert abs(float(newt(x)) - float(t2(x))) > 1e-30
